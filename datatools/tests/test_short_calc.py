import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from datatools import short_calc
from datatools.build_shared_data import BUILDERS, Context, reused_record, reused_rows
from datatools.normalize import Reject, check_answer_form, load_aliases


class ShortCalcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.items = list(short_calc.episodes(160))
        cls.ctx = Context(load_aliases(None), queries={}, new_writes=set(),
                          calc_results={i["source_id"]: [i["result"]] for i in cls.items})

    def test_episodes_are_deterministic_and_cover_every_template(self) -> None:
        self.assertEqual([i["user_prompt"] for i in short_calc.episodes(160)], [i["user_prompt"] for i in self.items])
        names = Counter(i["source_id"].split("-")[1] for i in self.items)
        self.assertEqual(set(names), {t.name for t in short_calc.TEMPLATES})

    def test_every_role_row_passes_the_shared_data_builder(self) -> None:
        labels = Counter()
        for role in ("yasama", "yurutme", "yargi"):
            for row in short_calc.rows(role, self.items):
                label, messages, _ = BUILDERS[role](row, short_calc.SOURCE, self.ctx)
                labels[role, label] += 1
        self.assertEqual(labels["yasama", "CALCULATE"], len(self.items))
        self.assertEqual(labels["yasama", "FINISH"], len(self.items))
        self.assertEqual(labels["yurutme", "CALCULATE"], len(self.items))
        self.assertTrue(0 < labels["yargi", "CALCULATE"] < len(self.items))

    def test_answer_numbers(self) -> None:
        self.assertEqual(short_calc.answer_number(156.25), "156,25")
        self.assertEqual(short_calc.answer_number(81.6666666667), "81,67")
        self.assertEqual(short_calc.answer_number(500), "500")
        self.assertEqual(short_calc.answer_number(396.8), "396,8")


class AnswerFormTests(unittest.TestCase):
    def test_labels_truncations_and_bare_numbers_are_rejected(self) -> None:
        for text in ("Uygulanamaz", "Not Applicable", "Geçerli Değil", "Bu nedenle cevap", "Bu nedenle, fark", "3", "-4.0"):
            with self.subTest(text=text), self.assertRaises(Reject):
                check_answer_form(text)

    def test_answers_are_kept(self) -> None:
        for text in ("Sonuç -4.0'dır.", "Kişi başı 156,25 TL düşer.", "Bu nedenle cevap 12'dir."):
            with self.subTest(text=text):
                check_answer_form(text)


class ReuseTests(unittest.TestCase):
    def test_reused_records_keep_id_and_split_and_pass_the_answer_filter(self) -> None:
        def record(id_, source, answer):
            return {"id": id_, "source": source, "source_dataset": "d", "split": "test", "action": "RESPOND",
                    "has_history": False, "messages": [{"role": "user", "content": "x"},
                                                       {"role": "assistant", "content": answer}]}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "yargi").mkdir()
            for split in ("train", "validation", "test"):
                with gzip.open(Path(tmp) / "yargi" / f"{split}.jsonl.gz", "wt") as handle:
                    if split == "test":
                        for item in (record("a", "chat_respond", "Merhaba!"), record("b", "chat_respond", "Uygulanamaz"),
                                     record("c", "arac_kullanimi", "Tamam.")):
                            handle.write(json.dumps(item) + "\n")
            rows = list(reused_rows(Path(tmp), "yargi", "chat_respond"))
        self.assertEqual([r["_record"]["id"] for r in rows], ["a", "b"])
        label, messages, meta = reused_record(rows[0], "yargi")
        self.assertEqual((label, messages[-1]["content"], meta), ("RESPOND", "Merhaba!", {"has_history": False}))
        with self.assertRaises(Reject):
            reused_record(rows[1], "yargi")


if __name__ == "__main__":
    unittest.main()
