"""Torch-free parts of the role trainer: serialization, packing, schedule."""

from __future__ import annotations

import unittest

from datatools.build_mix import TokenLength
from training.train_role import (
    EOS_ID,
    MESSAGE_END_ID,
    MESSAGE_START_ID,
    encode_record,
    lr_at,
    pack,
    parse_args,
)


def fake_encode(text: str) -> list[int]:
    # One id per character keeps spans easy to read; "\n" is 13 as in NDSRF004.
    return [13 if c == "\n" else 100 + (ord(c) % 1000) for c in text]


class FakeTokenizer:
    def encode_ids(self, data: bytes):
        return fake_encode(data.decode("utf-8"))


MESSAGES = [
    {"role": "system", "content": "Kural"},
    {"role": "user", "content": "Önceki"},
    {"role": "assistant", "content": "Eski cevap"},
    {"role": "user", "content": "Soru"},
    {"role": "assistant", "content": "Yanıt"},
]


class SerializationTests(unittest.TestCase):
    def test_only_the_last_assistant_message_is_the_target(self):
        ids, start, end = encode_record(fake_encode, MESSAGES)
        self.assertEqual(ids[start:end], [*fake_encode("Yanıt"), MESSAGE_END_ID])
        self.assertEqual(ids[start - len(fake_encode("asistan\n")) - 1], MESSAGE_START_ID)
        self.assertEqual(ids[end:], [13, EOS_ID])
        # The earlier answer is in the context, outside the span.
        earlier = fake_encode("Eski cevap")
        self.assertTrue(any(ids[i:i + len(earlier)] == earlier for i in range(start)))
        self.assertEqual(ids.count(MESSAGE_START_ID), len(MESSAGES))

    def test_mix_length_limit_counts_the_same_tokens(self):
        counter = object.__new__(TokenLength)
        counter._tokenizer = FakeTokenizer()
        ids, _, _ = encode_record(fake_encode, MESSAGES)
        # At these limits the byte shortcut does not apply, so the exact count decides.
        counter.max_tokens = len(ids)
        self.assertFalse(counter({"messages": MESSAGES}))
        counter.max_tokens = len(ids) - 1
        self.assertTrue(counter({"messages": MESSAGES}))

    def test_a_record_must_end_with_the_target(self):
        with self.assertRaises(ValueError):
            encode_record(fake_encode, MESSAGES[:-1])


class PackingTests(unittest.TestCase):
    def test_rows_keep_order_and_never_overflow(self):
        lengths = [5, 4, 3, 6, 2]
        rows = pack(lengths, [0, 1, 2, 3, 4], capacity=10)
        self.assertEqual(rows, [[0, 1], [2, 3], [4]])
        self.assertEqual(sorted(i for row in rows for i in row), [0, 1, 2, 3, 4])
        with self.assertRaises(ValueError):
            pack([11], [0], capacity=10)

    def test_schedule_warms_up_then_decays_to_the_floor(self):
        self.assertAlmostEqual(lr_at(0, 100, 1e-5, 10, 0.1), 1e-6)
        self.assertAlmostEqual(lr_at(9, 100, 1e-5, 10, 0.1), 1e-5)
        self.assertAlmostEqual(lr_at(100, 100, 1e-5, 10, 0.1), 1e-6)
        self.assertGreater(lr_at(50, 100, 1e-5, 10, 0.1), lr_at(80, 100, 1e-5, 10, 0.1))

    def test_role_defaults(self):
        args = parse_args(["--role", "yargi", "--mix", "m", "--run-dir", "r"])
        self.assertEqual((args.lr, args.epochs, args.select_by), (5e-6, 2, "loss"))
        args = parse_args(["--role", "yasama", "--mix", "m", "--run-dir", "r", "--lr", "2e-5"])
        self.assertEqual((args.lr, args.select_by), (2e-5, "exact_match"))


if __name__ == "__main__":
    unittest.main()
