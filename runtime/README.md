# Runtime backends

The role runtimes depend on a shared `ChatBackend` protocol. The orchestration code does not need to know how a model is executed.

For terminal launcher/configuration usage, see [`CLI.md`](../CLI.md).

Supported adapters:

- `OllamaBackend`: Ollama native `/api/chat`
- `LlamaCppBackend`: llama.cpp OpenAI-compatible `/v1/chat/completions`
- `MercanCliBackend`: the `mercan` executable from MercanRuntime
- `LibMercanBackend`: in-process EthosoftLib SDK adapter over `libmercan`

## Ollama

```python
from runtime import OllamaBackend
from yasama.yasama import YasamaRuntime

backend = OllamaBackend(
    model="mercan-yasama",
    base_url="http://127.0.0.1:11434",
)
yasama = YasamaRuntime(backend=backend)
```

## llama.cpp server

Start a llama.cpp server separately, then point the adapter at it:

```python
from runtime import LlamaCppBackend
from yurutme.yurutme import YurutmeRuntime

backend = LlamaCppBackend(
    base_url="http://127.0.0.1:8080",
    model="mercan-yurutme",
)
yurutme = YurutmeRuntime(backend=backend)
```

## libmercan through the Mercan CLI

`MercanCliBackend` runs the installed MercanRuntime CLI. The executable can be linked to static `libmercan`.

```python
from runtime import MercanCliBackend
from yargi.yargi import YargiRuntime

backend = MercanCliBackend(
    model="/models/yargi.mercan",
    binary="/opt/mercan/bin/mercan",
    threads=8,
    gpu_layers=0,
)
yargi = YargiRuntime(backend=backend)
```

A Hugging Face model reference accepted by the Mercan CLI can also be used as `model`, for example `owner/repo:model-q4.mercan`.

## Direct in-process libmercan

`LibMercanBackend` delegates to `ethosoftlib.mercan.Model.generate(messages)`.
EthosoftLib owns the native bindings, ABI validation, model/context lifetime,
formatting and sampling. Requests contain the complete role transcript and
never read or update SDK chat history. There is no subprocess or HTTP server.

Install the SDK (0.6.0 or later) and build MercanRuntime (0.2.0 or later) as a shared library:

```bash
# For local development, install the sibling SDK checkout first:
python -m pip install -e /path/to/EthosoftLib
python -m pip install -e '.[mercan]'

git clone https://github.com/Ahmet2001/MercanRuntime.git
cd MercanRuntime
./scripts/prepare_llama.sh
cmake -S . -B build -DMERCAN_BUILD_SHARED=ON
cmake --build build -j
```

Then point the Python backend at the produced shared library and a local `.mercan` model:

```python
from runtime import LibMercanBackend
from yasama.yasama import YasamaRuntime

backend = LibMercanBackend(
    model="/models/yasama.mercan",
    library="/path/to/libmercan.so",   # macOS: .dylib, Windows: .dll
    threads=8,
    gpu_layers=0,
)

yasama = YasamaRuntime(backend=backend)
```

The SDK calls the versioned native chat API; the complete generation loop
and sampling live in C++, shared with the CLI:

```text
mercan_chat_begin_v1 -> mercan_generation_run_v1
  -> tokenize -> batched prompt decode -> native sample -> token decode
  -> output byte callback -> ... -> stop/limit/cancel
mercan_generation_free_v1

# The low-level decode/logits API remains available for custom integrations:
mercan_model_load
  -> mercan_context_create
  -> mercan_tokenize
  -> mercan_decode(prompt)
  -> mercan_logits
  -> sample token
  -> mercan_decode(token)
  -> ...
```

It uses the same ChatML-style role formatting as the current Mercan CLI by default:

```text
system    -> sistem
user      -> kullanici
assistant -> asistan
```

The role names can be overridden with `role_system`, `role_user`, and `role_assistant` if a checkpoint uses a different template.

External Mercan architecture/tokenizer plugins can be loaded before model load:

```python
backend = LibMercanBackend(
    model="/models/model.mercan",
    library="/opt/mercan/lib/libmercan.so",
    plugins=["/opt/mercan/plugins/libmercan_arch_anka.so"],
)
```

Unlike `MercanCliBackend`, the direct backend expects a local model path; Hugging Face pull/cache resolution belongs outside the C ABI adapter.

The `library` setting is optional when EthosoftLib bundles the shared library
or `ETHOSOFT_MERCAN_LIBRARY` points to it. Existing `threads`, `threads_batch`,
`gpu_layers`, `n_ctx`, `n_batch`, model options, plugins and custom role names
are forwarded to the SDK. `timeout_seconds` is not enforced for in-process
generation; cancellation in the SDK is cooperative between decode calls.

For all three roles, copy `sop.libmercan.example.json`, replace the model
paths with checkpoints trained for the respective role contracts, and run
`mercan-sop chat --config your-config.json`. The native smoke test uses one
general SFT model for all adapters to check transport and lifetime only; it
does not establish role-specific routing or tool-argument accuracy.

For a local build without altering an existing runtime checkout, run from
EthosoftLib:

```bash
MERCAN_USE_LOCAL_SOURCE=1 MERCAN_SRC=/path/to/MercanRuntime \
  MERCAN_JOBS=1 bash scripts/build_mercan_native_linux.sh
```

Run native adapter regressions with existing model artifacts:

```bash
ETHOSOFT_MERCAN_LIBRARY=/path/to/libmercan.so \
ETHOSOFT_MERCAN_TEST_MODEL=/path/to/model.mercan \
  python -m pytest -q tests/test_libmercan_integration.py
```

Use `close()` or a context manager when a model should be released explicitly:

```python
with LibMercanBackend(
    model="/models/yargi.mercan",
    library="/opt/mercan/lib/libmercan.so",
) as backend:
    yargi = YargiRuntime(backend=backend)
    # run pipeline here
```

## Mixed backends

Each role owns its own runtime, so backends can be mixed without changing `main.py`:

```python
from runtime import LibMercanBackend, LlamaCppBackend, OllamaBackend
from yasama.yasama import YasamaRuntime
from yurutme.yurutme import YurutmeRuntime
from yargi.yargi import YargiRuntime

yasama = YasamaRuntime(backend=OllamaBackend("mercan-yasama"))
yurutme = YurutmeRuntime(
    backend=LlamaCppBackend(model="mercan-yurutme")
)
yargi = YargiRuntime(
    backend=LibMercanBackend(
        model="/models/yargi.mercan",
        library="/opt/mercan/lib/libmercan.so",
    )
)
```

The same backend instance may also be shared when the same model/runtime serves multiple roles.

## Prompt serialization contract

Role prompt serialization lives in `runtime/serialization.py`. This keeps backend transport separate from the training/runtime data contract.

Current layouts are:

```text
Yasama:
USER:
<request>
[CONVERSATION_HISTORY: ...]
STATE:
{...}

Yurutme:
USER:
<request>
[CONVERSATION_HISTORY: ...]
STATE:
{...}
ACTION:
<action>
TOOL_SCHEMA:
{...}
PLACEHOLDER:
{...}

Yargi:
<request>
[<conversation_history>...</conversation_history>]
<tool_observations>
[...]
</tool_observations>
```

Yasama still outputs only a canonical action string. Yurutme still outputs only the arguments JSON object. Yargi still outputs only natural language.

`conversation_history` is documented in `MasterFormat.txt` as conceptual input, but the report does not define one exact serialization example for it. The current deterministic history blocks are therefore a runtime convention, not a finalized training standard; update only `runtime/serialization.py` if the training formatter later establishes another representation.
