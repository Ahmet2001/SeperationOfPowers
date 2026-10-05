# Runtime backends

The role runtimes depend on a shared `ChatBackend` protocol. The orchestration code does not need to know how a model is executed.

Supported adapters:

- `OllamaBackend`: Ollama native `/api/chat`
- `LlamaCppBackend`: llama.cpp OpenAI-compatible `/v1/chat/completions`
- `MercanCliBackend`: the `mercan` executable from `mercanApp-test1`, which is linked against the custom `libmercan` runtime

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

## libmercan / Mercan runtime

`mercanApp-test1` builds a `mercan` CLI executable linked against `libmercan`. The adapter deliberately uses that executable as the stable process boundary so it works whether `libmercan` is built static (the current default) or shared.

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

## Mixed backends

Each role owns its own runtime, so backends can be mixed without changing `main.py`:

```python
from runtime import LlamaCppBackend, MercanCliBackend, OllamaBackend
from yasama.yasama import YasamaRuntime
from yurutme.yurutme import YurutmeRuntime
from yargi.yargi import YargiRuntime

 yasama = YasamaRuntime(backend=OllamaBackend("mercan-yasama"))
 yurutme = YurutmeRuntime(
     backend=LlamaCppBackend(model="mercan-yurutme")
 )
 yargi = YargiRuntime(
     backend=MercanCliBackend("/models/yargi.mercan")
 )
```

The same backend instance may also be shared when the same model/server serves multiple roles.
