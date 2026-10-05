# Runtime backends

The role runtimes depend on a shared `ChatBackend` protocol. The orchestration code does not need to know how a model is executed.

Supported adapters:

- `OllamaBackend`: Ollama native `/api/chat`
- `LlamaCppBackend`: llama.cpp OpenAI-compatible `/v1/chat/completions`
- `MercanCliBackend`: the `mercan` executable from `mercanApp-test1`
- `LibMercanBackend`: direct in-process ctypes binding to the custom `libmercan` C ABI

## Training-contract serialization

`runtime/serialization.py` is the single runtime source of truth for what each role model sees. Backend choice must not change this serialization.

Yasama receives the documented conceptual input in a deterministic layout:

```text
USER:
<current user request>
CONVERSATION_HISTORY:
<JSON, only when non-empty>
STATE:
<JSON>
```

Its model output remains a raw canonical action string such as `FILE_READ` or `FINISH`; arguments are never part of the Yasama target.

Yurutme receives:

```text
USER:
<current user request>
CONVERSATION_HISTORY:
<JSON, only when non-empty>
STATE:
<JSON>
ACTION:
<selected action>
TOOL_SCHEMA:
<JSON>
PLACEHOLDER:
<JSON or null>
```

Its model output is the arguments JSON object only.

Yargi follows the documented ChatML example. The user message contains the raw current request followed by real executor observations:

```text
<current user request>
<tool_observations>
[
  ... real observations ...
]
</tool_observations>
```

When relevant conversation history is non-empty, it is serialized before the observation block. The final Yargi model output is natural language only; it is not wrapped in JSON.

The exact history serialization is an explicit runtime convention because `MasterFormat.txt` defines `conversation_history` conceptually but does not provide a concrete serialized example for it. If training later adopts another history encoding, change `runtime/serialization.py` and the matching formatter together.

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

`MercanCliBackend` is the simplest portable boundary. It works with the normal `mercanApp-test1` build where `libmercan` is static and the `mercan` executable is linked against it.

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

`LibMercanBackend` talks directly to the public `mercan.h` C ABI with Python `ctypes`. There is no subprocess and no HTTP server in this path.

Build `mercanApp-test1` as a shared library first:

```bash
git clone https://github.com/Ahmet2001/mercanApp-test1.git
cd mercanApp-test1
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

The direct backend mirrors the generation loop in `mercanApp-test1/cli/main.cpp`:

```text
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
