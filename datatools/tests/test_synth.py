import unittest

from datatools.build_shared_data import Context, yargi_record, yasama_record
from datatools.normalize import load_aliases
from datatools.synth import (
    BIG_RATE,
    chat_respond_rows,
    exchange,
    helpsteer_chat,
    history_rate,
    needs_tool,
    with_history,
)

CTX = Context(load_aliases(None), queries={}, calc_results={}, new_writes=set())


class NeedsToolTests(unittest.TestCase):
    def test_tool_requests_are_not_respond_examples(self) -> None:
        for prompt in ("125 * 4 kaç eder?", "Bugün İstanbul'da hava durumu nasıl?", "Dolar kuru ne kadar?",
                       "main.py dosyasını oku", "https://ornek.com sayfasını aç", "ali@ornek.com adresine mail at",
                       "2026 seçim sonuçları", "internette Python 3.13'ü araştır"):
            with self.subTest(prompt=prompt):
                self.assertTrue(needs_tool(prompt))

    def test_plain_chat_is_kept(self) -> None:
        for prompt in ("Sosyal anksiyete ile başa çıkmak için ne yapabilirim?",
                       "Toskana'da otantik bir bağ bozumu için nereleri önerirsin?", "Merhaba, nasılsın?"):
            with self.subTest(prompt=prompt):
                self.assertFalse(needs_tool(prompt))


class ChatSourceTests(unittest.TestCase):
    def test_helpsteer_takes_the_preferred_response_and_history(self) -> None:
        row = {"domain": "general", "overall_preference": 2, "response1": "kötü", "response2": "iyi",
               "context": [{"role": "user", "content": "Selam"}, {"role": "assistant", "content": "Merhaba!"},
                           {"role": "user", "content": "Bir şiir önerir misin?"}]}
        history, prompt, answer = helpsteer_chat(row)
        self.assertEqual((prompt, answer), ("Bir şiir önerir misin?", "iyi"))
        self.assertEqual(len(history), 2)
        self.assertIsNone(helpsteer_chat({**row, "domain": "code"}))
        self.assertIsNone(helpsteer_chat({**row, "overall_preference": 0}))

    def test_respond_rows_build_for_yasama_and_yargi(self) -> None:
        dpo = iter([{"prompt": "Kahve mi çay mı daha sağlıklı?", "chosen": "<think>x</think>İkisinin de faydaları var."},
                    {"prompt": "250 / 5 kaç?", "chosen": "50"}])
        human = iter([{"prompt": "Karaoke sever misin?", "chosen": "Bayılırım!"}])
        yasama = list(chat_respond_rows("yasama", dpo, human, iter([]), dpo_rate=1.0))
        self.assertEqual([r["user_prompt"] for r in yasama], ["Kahve mi çay mı daha sağlıklı?", "Karaoke sever misin?"])
        label, messages, _ = yasama_record(yasama[0], "chat_respond", CTX)
        self.assertEqual((label, messages[-1]["content"]), ("RESPOND", "RESPOND"))

        dpo = iter([{"prompt": "Kahve mi çay mı daha sağlıklı?", "chosen": "<think>x</think>İkisinin de faydaları var."}])
        yargi = list(chat_respond_rows("yargi", dpo, iter([{"prompt": "a", "chosen": "b"}]), iter([]), dpo_rate=1.0))
        self.assertEqual(len(yargi), 1)  # Human-Like answers are not used for Yargi
        label, messages, _ = yargi_record(yargi[0], "chat_respond", CTX)
        self.assertEqual(label, "RESPOND")
        self.assertIn("<tool_observations>\n[]\n</tool_observations>", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "İkisinin de faydaları var.")


class HistoryMixTests(unittest.TestCase):
    def test_rates_favor_rare_actions(self) -> None:
        self.assertEqual(history_rate("WEB_SEARCH"), BIG_RATE)
        self.assertEqual(history_rate("FILE_MOVE"), 1.0)
        self.assertGreater(history_rate("OPEN_TOOL"), BIG_RATE)

    def test_earlier_turn_reaches_the_request_but_not_the_label(self) -> None:
        earlier = exchange("Azelaik asit nedir?", "Azelaik asit bir dikarboksilik asittir.")
        row = {"source_id": "g-1", "step_index": 0, "user_prompt": "Merhaba, nasılsın?",
               "conversation_history": [], "state": {"completed_actions": [], "last_observation": None},
               "target_action": "RESPOND"}
        mixed = with_history(row, earlier)
        label, messages, meta = yasama_record(mixed, "history_mix", CTX)
        self.assertEqual((label, meta["has_history"]), ("RESPOND", True))
        self.assertIn("CONVERSATION_HISTORY:", messages[0]["content"])
        self.assertIsNone(exchange("soru", "x" * 5000))


if __name__ == "__main__":
    unittest.main()
