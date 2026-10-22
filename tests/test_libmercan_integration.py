"""Real SDK/native role-adapter smoke; no model downloads or tool execution."""
import os
from contextlib import ExitStack

import pytest

from runtime import GenerationConfig, LibMercanBackend


@pytest.mark.skipif(
    not (os.environ.get("ETHOSOFT_MERCAN_LIBRARY")
         and os.environ.get("ETHOSOFT_MERCAN_TEST_MODEL")),
    reason="set ETHOSOFT_MERCAN_LIBRARY and ETHOSOFT_MERCAN_TEST_MODEL",
)
def test_three_role_adapters_use_sdk_and_release_independently():
    model_path = os.environ["ETHOSOFT_MERCAN_TEST_MODEL"]
    library = os.environ["ETHOSOFT_MERCAN_LIBRARY"]
    with ExitStack() as stack:
        backends = [stack.enter_context(LibMercanBackend(
            model_path, library=library, n_ctx=1024, threads=2,
        )) for _ in range(3)]
        for role, backend in zip(("Yasama", "Yürütme", "Yargı"), backends):
            answer = backend.generate([
                {"role": "system", "content": f"Sen {role} rolündesin. Türkçe kısa cevap ver."},
                {"role": "user", "content": "Merhaba, nasılsın?"},
            ], config=GenerationConfig(max_tokens=8))
            assert answer.strip(), role
            assert backend._sdk_model.history == ()
            print(role, repr(answer))
        # Closing a role must not free the process-wide backend of other roles.
        backends[0].close()
        assert backends[1].generate(
            [{"role": "user", "content": "Merhaba"}],
            config=GenerationConfig(max_tokens=8),
        ).strip()
    # Initialization also works after all previous runtime handles were closed.
    with LibMercanBackend(model_path, library=library, threads=2) as backend:
        assert backend.version
