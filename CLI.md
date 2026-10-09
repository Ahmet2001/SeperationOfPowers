# mercan-sop CLI

The repository can be installed as a local command-line tool:

```bash
pip install -e .
```

After that, use `mercan-sop` directly from the terminal.

## Optional Turkish voice output: EMA Lightning

Use `--speech` to read **the final user-visible answer** aloud after
Yasama/Yurutme/Yargi produces it. The Yurutme role emits JSON tool arguments,
so those are **not** read aloud; Yargi's answer (or an ASK_CLARIFICATION
question) is voiced instead.

Speech uses [canberkkkkkk/ema-lightning](https://huggingface.co/canberkkkkkk/ema-lightning),
a small Turkish TTS model. Its upstream `ema-lightning` Python package
downloads model weights (~34 MB) on first use and caches them locally; after
that it can run offline. It does not use the Ollama server for TTS.

### Install speech with CPU-only PyTorch (recommended)

EMA Lightning uses **CPU inference** in this CLI: `EMA(device="cpu")`.
PyTorch must also be a CPU-only wheel. The upstream TTS package requires
`torch>=2.1`; a plain `pip install -e ".[speech]"` **may install a CUDA-enabled
PyTorch build on Linux**, so use the installer below instead.

EMA Lightning supports **Python 3.11–3.13**. Activate a virtual environment
created with one of these versions, then run the installer:

```bash
cd ~/Masaüstü/SeperationOfPowers
source .venv/bin/activate
python --version
bash scripts/install_speech_cpu.sh
```

The script installs/reinstalls `torch>=2.1` using PyTorch's **official CPU wheel
index** (`https://download.pytorch.org/whl/cpu`), installs the optional
`ema-lightning` and `sounddevice` dependencies, then **verifies**
`torch.version.cuda is None` and `torch.version.hip is None`. It uses your
active virtual environment and never uses `sudo`.

Equivalent manual installation:

```bash
python -m pip install --upgrade --force-reinstall "torch>=2.1" \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[speech]"
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

On Ubuntu, if PortAudio is missing, install the system library separately:

```bash
sudo apt install libportaudio2
```

**If CUDA PyTorch was already installed:** switching to a CPU wheel removes
the GPU-enabled `torch` build but may leave unused `nvidia-*` CUDA dependency
packages in that virtual environment. For maximum disk savings, create a
**fresh Python 3.11–3.13 virtual environment** and run the installer there;
do not indiscriminately uninstall CUDA packages used by other projects.

Speech remains optional: `python -m pip install -e .` alone does **not** add
PyTorch. Enabling `--speech` with a GPU-enabled PyTorch build prints a
`[speech:error]` explaining how to switch, while the text response remains
available. There is no download of the EMA model weights until speech's
first use.

Example chat with the models installed in Ollama:

```bash
mercan-sop chat --speech --debug \
  --yasama-backend ollama --yasama-model qwen3:1.7b \
  --yurutme-backend ollama --yurutme-model qwen3:1.7b \
  --yargi-backend ollama --yargi-model qwen3:1.7b
```

One-shot mode:

```bash
mercan-sop run --speech \
  --yasama-model qwen3:1.7b \
  --yurutme-model qwen3:1.7b \
  --yargi-model qwen3:1.7b \
  "Merhaba, nasılsın?"
```

`--speech` is opt-in; plain runs do not import audio/TTS packages or download
model weights. You can also enable it in JSON config with
`{"pipeline":{"speech":true}}` or `--set pipeline.speech=true`.
A TTS import, download, or speaker error is printed as `[speech:error]` and
does not prevent successful text answers or the next chat turn. Audio playback
requires a working local output device.

Synthetic speech should be disclosed to listeners when used in a service.

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
