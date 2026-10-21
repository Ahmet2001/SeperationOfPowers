import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datatools import build_mix
from datatools.build_mix import grounded_answer_ok, search_prompt_ok, stratum
from yasama.yasama import request_message as yasama_request
from yargi.yargi import request_messages as yargi_requests

STATE = {"completed_actions": [], "last_observation": None, "recent_observations": [],
         "step_index": 0, "turn_index": 0}


def yasama_record(index, prompt, action, source, done=()):
    state = {**STATE, "completed_actions": list(done),
             "recent_observations": [{"action": a, "status": "success", "error": None} for a in done]}
    return {"id": f"{source}:{index}", "source": source, "action": action, "split": "train",
            "messages": [yasama_request(user_prompt=prompt, conversation_history=[], state=state),
                         {"role": "assistant", "content": action}]}


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class QualityTests(unittest.TestCase):
    def test_search_prompts_must_be_whole_questions(self):
        for prompt in ("Hamile kadın grip aşısı olabilir mi?", "İtalya'da Füniküler Nedir",
                       "Ameliyat sırasında tansiyon neden yükselir"):
            self.assertTrue(search_prompt_ok(prompt), prompt)
        for prompt in ("ms bir nörodejeneratif hastalık", "Oyun Grubu", "görüntü tanımı edebi nedir",
                       "Kırmızı boyunlu wallaby bu İngiliz"):
            self.assertFalse(search_prompt_ok(prompt), prompt)

    def test_answers_must_come_from_the_results(self):
        observations = '{"snippet": "Sarıhumma salgını 1793 yılında Philadelphia şehrinde meydana geldi."}'
        self.assertTrue(grounded_answer_ok("Sarıhumma salgını 1793 yılında Philadelphia'da meydana geldi.", observations))
        self.assertFalse(grounded_answer_ok("Kendi kendine yardımın yeniden ele geçirilmesi tanımlanır.", observations))
        self.assertFalse(grounded_answer_ok("1793", observations))

    def test_strata(self):
        good, bad = "Sukraloz kan şekerini etkiler mi?", "sukraloz kan şekeri"
        self.assertEqual(stratum("yasama", yasama_record(1, good, "WEB_SEARCH", "arac_kullanimi")), "WEB_SEARCH/ms-marco")
        self.assertIsNone(stratum("yasama", yasama_record(2, bad, "WEB_SEARCH", "arac_kullanimi")))
        self.assertEqual(stratum("yasama", yasama_record(3, good, "FINISH", "arac_kullanimi", ["WEB_SEARCH"])),
                         "FINISH/after-WEB_SEARCH")
        self.assertIsNone(stratum("yasama", yasama_record(4, bad, "FINISH", "arac_kullanimi", ["WEB_SEARCH"])))
        self.assertEqual(stratum("yasama", yasama_record(5, "Kahve mi çay mı?", "RESPOND", "chat_respond")), "RESPOND/chat")
        self.assertEqual(stratum("yasama", yasama_record(6, "Dosyaları listele", "FILE_LIST", "hunter_tr")),
                         "FILE_LIST/hunter_tr")
        messages = yargi_requests(user_prompt=bad, conversation_history=[], observations=[
            {"action": "WEB_SEARCH", "data": {"query": bad, "results": []}, "status": "success", "error": None}])
        record = {"id": "a:1", "source": "arac_kullanimi", "action": "WEB_SEARCH",
                  "messages": [*messages, {"role": "assistant", "content": "Evet."}]}
        self.assertIsNone(stratum("yargi", record))


class BuildTests(unittest.TestCase):
    def test_caps_sample_deterministically_and_keep_small_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared, out = Path(tmp, "shared"), Path(tmp, "mix")
            records = [yasama_record(i, f"Ankara {i}. caddesi nerede bulunur?", "WEB_SEARCH", "arac_kullanimi") for i in range(50)]
            records += [yasama_record(100 + i, "Klasörü listeler misin?", "FILE_LIST", "hunter_tr") for i in range(3)]
            records.append(yasama_record(200, "kısa", "WEB_SEARCH", "arac_kullanimi"))
            for role in build_mix.ROLES:
                for split in build_mix.SPLITS:
                    write(shared / role / f"{split}.jsonl.gz", records if (role, split) == ("yasama", "train") else [])
            with patch.dict(build_mix.CAPS, {("yasama", "WEB_SEARCH/ms-marco"): 10}):
                first = build_mix.build(shared, out, lambda _: None)
                chosen = [json.loads(line)["id"] for line in gzip.open(out / "yasama" / "train.jsonl.gz", "rt")]
                build_mix.build(shared, out, lambda _: None)
                again = [json.loads(line)["id"] for line in gzip.open(out / "yasama" / "train.jsonl.gz", "rt")]
            self.assertEqual(chosen, again)
            counts = first["counts"]["yasama"]["train"]
            self.assertEqual(counts["by_action"], {"WEB_SEARCH": 10, "FILE_LIST": 3})
            self.assertEqual(counts["strata"]["WEB_SEARCH/ms-marco"], {"available": 50, "kept": 10})
            self.assertEqual(counts["quality_rejects"], {"WEB_SEARCH/arac_kullanimi": 1})
            record = json.loads(gzip.open(out / "yasama" / "train.jsonl.gz", "rt").readline())
            self.assertEqual(record["messages"][0], build_mix.SYSTEM_MESSAGES["yasama"])
            self.assertEqual([m["role"] for m in record["messages"]], ["system", "user", "assistant"])


if __name__ == "__main__":
    unittest.main()
