"""Run the NedoLM .mercan converter with libmercan-safe token pieces.

    python training/mercan_convert_escaped.py <mercan_convert_32002.py> <checkpoint> --output ... [converter args]

libmercan's llama fork decodes NedoLM token pieces as RWKV vocabulary, which
unescapes ``\\t``, ``\\n``, ``\\r``, ``\\xHH`` and ``\\<c>``. The converter stores
raw bytes, so the lone backslash token (id 95) decodes to nothing: generated
JSON loses every escape (``"a\\nb"`` comes out as ``"anb"``). Escaping each
backslash in the stored pieces makes the unescape return the raw bytes.
Encoding is unaffected: NDSRF004 text goes through the exact external encoder.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def escape_piece(piece: bytes) -> bytes:
    return piece.replace(b"\\", b"\\\\")


def main() -> None:
    converter = Path(sys.argv.pop(1))
    spec = importlib.util.spec_from_file_location("mercan_convert", converter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parse_vocab_txt = module.parse_vocab_txt

    def escaped(path, vocab_size):
        tokens, token_types = parse_vocab_txt(path, vocab_size)
        changed = sum(b"\\" in token for token in tokens)
        print(f"escaped backslashes in {changed} token pieces", flush=True)
        return [escape_piece(token) for token in tokens], token_types

    module.parse_vocab_txt = escaped
    module.main()


if __name__ == "__main__":
    main()
