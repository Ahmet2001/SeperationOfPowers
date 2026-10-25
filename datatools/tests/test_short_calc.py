import unittest
from collections import Counter

from datatools import short_calc
from datatools.build_shared_data import BUILDERS, Context
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


if __name__ == "__main__":
    unittest.main()
