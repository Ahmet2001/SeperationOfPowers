from __future__ import annotations

import unittest

from runtime.output_parsing import trim_trailing_loop
from yargi.yargi import YargiRuntime


class TrailingLoopTests(unittest.TestCase):
    def test_loop_to_the_end_keeps_one_copy(self):
        text = "Site hazır. Özellikler: - Parşömen mumlar - Paralaks" + " - Parşömen mumlar - Paralaks" * 12
        self.assertEqual(trim_trailing_loop(text), "Site hazır. Özellikler: - Parşömen mumlar - Paralaks")

    def test_cut_off_last_copy_and_case(self):
        self.assertEqual(trim_trailing_loop("Menü: Alt " + "alt " * 30 + "Al"), "Menü: Alt")
        text = "Renkler: " + "**Krem** (#c9a962) - " * 9 + "**Krem** (#c9"
        self.assertEqual(trim_trailing_loop(text), "Renkler: **Krem** (#c9a962)")

    def test_ordinary_answers_are_untouched(self):
        for text in ("Toplam 156,25 lira. Kişi başı 156,25 lira.",
                     "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |",
                     "Çok çok çok teşekkürler, iyi günler!",
                     "ha ha ha",
                     "Adım 1: oku. Adım 2: oku. Adım 3: oku. Sonra bitti."):
            self.assertEqual(trim_trailing_loop(text), text)

    def test_yargi_trims_its_answer(self):
        runtime = YargiRuntime(lambda payload: "Dosya oluşturuldu." + " Dosya oluşturuldu." * 10)
        self.assertEqual(runtime.run(user_prompt="x", conversation_history=[], observations=[]), "Dosya oluşturuldu.")


if __name__ == "__main__":
    unittest.main()
