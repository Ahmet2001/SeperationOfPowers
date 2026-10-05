# mercan-sop CLI

The repository can be installed as a local command-line tool:

```bash
pip install -e .
```

After that, use `mercan-sop` directly from the terminal.

## Direct model/backend test

Use `model-test` when you want to verify that a model runtime works before entering the Yasama/Yurutme/Yargi pipeline.

Ollama example:

```bash
mercan-sop model-test ollama qwen2.5:7b
```

Custom prompt:

```bash
mercan-sop model-test ollama qwen2.5:7b \
  --prompt "Merhaba, kendini tek cümlede tanıt."
```

llama.cpp server example:

```bash
mercan-sop model-test llamacpp mercan-test \
  --base-url http://127.0.0.1:8080
```

Mercan CLI example:

```bash
mercan-sop model-test mercan-cli /models/model.mercan \
  --binary /opt/mercan/bin/mercan \
  --threads 8
```

Direct libmercan example:

```bash
mercan-sop model-test libmercan /models/model.mercan \
  --library /opt/mercan/lib/libmercan.so \
  --threads 8 \
  --gpu-layers -1
```

`model-test` sends one system message and one user message directly to the selected backend. It does not invoke Yasama, Yurutme, Yargi, the tool registry, or the executor. This makes it useful for separating model/server problems from pipeline problems.

## Create a starter config

```bash
mercan-sop init-config sop.json
```

## Inspect the resolved config

```bash
mercan-sop show-config --config sop.json
```

Any config key can be overridden from the terminal with `--set`:

```bash
mercan-sop show-config \
  --config sop.json \
  --set roles.yasama.model=my-router \
  --set roles.yurutme.threads=8
```

## One-shot run

```bash
mercan-sop run \
  --config sop.json \
  "main.py dosyasini oku"
```

You can also configure the most common role settings without a config file:

```bash
mercan-sop run \
  --yasama-backend ollama \
  --yasama-model mercan-yasama \
  --yurutme-backend llamacpp \
  --yurutme-model mercan-yurutme \
  --yurutme-base-url http://127.0.0.1:8080 \
  --yargi-backend libmercan \
  --yargi-model /models/yargi.mercan \
  --set roles.yargi.library=/opt/mercan/lib/libmercan.so \
  "Merhaba"
```

## Interactive chat

```bash
mercan-sop chat --config sop.json
```

Inside the interactive terminal:

```text
/config   print active configuration
/exit     exit
/quit     exit
```

Conversation history is kept across terminal turns and passed back into the role runtimes.

## Runtime health check

```bash
mercan-sop doctor --config sop.json
```

`doctor` checks role/backend/model configuration and local paths required by direct `libmercan` setups. It does not send an inference request to remote Ollama or llama.cpp servers. Use `model-test` for a real inference check.

## Backend names

The CLI accepts these backend identifiers:

```text
ollama
llamacpp
mercan-cli
libmercan
```

Each role may use a different backend.

## Example libmercan settings

```bash
mercan-sop chat \
  --set roles.yasama.backend=libmercan \
  --set roles.yasama.model=/models/yasama.mercan \
  --set roles.yasama.library=/opt/mercan/lib/libmercan.so \
  --set roles.yasama.threads=8 \
  --set roles.yasama.gpu_layers=-1 \
  --set roles.yurutme.backend=libmercan \
  --set roles.yurutme.model=/models/yurutme.mercan \
  --set roles.yurutme.library=/opt/mercan/lib/libmercan.so \
  --set roles.yargi.backend=libmercan \
  --set roles.yargi.model=/models/yargi.mercan \
  --set roles.yargi.library=/opt/mercan/lib/libmercan.so
```

## Tool registry and executor hooks

The CLI launches the model pipeline; actual tool execution remains an explicit runtime dependency.

A registry file can be attached with:

```bash
mercan-sop chat --tool-registry tools.json
```

An executor callable can be loaded dynamically with `module:function` syntax:

```bash
mercan-sop chat \
  --tool-registry tools.json \
  --executor my_project.executor:execute
```

The executor callable must accept:

```python
execute(action: str, arguments: dict) -> dict
```

and return the canonical observation shape:

```json
{
  "action": "FILE_READ",
  "data": {},
  "status": "success",
  "error": null
}
```

If Yasama selects a tool action while no executor is configured, the CLI stops that turn with an explicit error instead of pretending the tool ran.
