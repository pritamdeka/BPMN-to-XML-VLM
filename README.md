# BPMN Redrawer Suite (code-only, multi-model)

This repository contains **only Python code** (no notebooks) for converting BPMN **diagram images** → **BPMN XML**
using multiple backends: **OpenAI (GPT‑4o)**, **Mistral**, **Gemini**, **Gemma 3**, and **Qwen 2.5‑VL**.

## Structure
```
src/
  pipelines/
    gpt4o_redrawer.py
    mistral_redrawer.py
    gemini_redrawer.py
    gemma_redrawer.py
    qwen_redrawer.py
  utils.py
  cli.py
examples/
  prompts
  Camunda-test-set/
	images
	xml
  hdBPMN-test-subset/
	images
	xml 

```

## Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
Prepare a prompt and images:
```bash
echo "Write the BPMN XML strictly between triple backticks..." > examples/prompts/prompt.txt
# add BPMN images to examples/images/
```

Run with the unified CLI:
```bash
# OpenAI (uses OPENAI_API_KEY unless --api-key is passed)
python -m src.cli --engine openai --image-folder examples/images --prompt-file examples/prompts/prompt.txt --model gpt-4o --out-dir outputs

# Mistral (uses MISTRAL_API_KEY)
python -m src.cli --engine mistral --image-folder examples/images --prompt-file examples/prompts/prompt.txt --model mistral-small-2503 --out-dir outputs

# Gemini (uses GOOGLE_API_KEY)
python -m src.cli --engine gemini --image-folder examples/images --prompt-file examples/prompts/prompt.txt --model gemini-2.5-flash --out-dir outputs

# Gemma (local HF model)
python -m src.cli --engine gemma --image-folder examples/images --prompt-file examples/prompts/prompt.txt --model google/gemma-3-4b-it --out-dir outputs

# Qwen (local HF model)
python -m src.cli --engine qwen --image-folder examples/images --prompt-file examples/prompts/prompt.txt --model Qwen/Qwen2.5-VL-3B-Instruct --out-dir outputs
```

**Outputs:** `.bpmn` files per image, plus `metrics_*.json`, `metrics_*.csv`, and `logs_*.json` in `outputs/`.

## Security
- **No secrets are stored.** Set keys via env vars or `--api-key` flags:
  - `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `GOOGLE_API_KEY`
  - For Hugging Face gated models, set: `HF_TOKEN` as needed (login separately if required).
- The example keys in your original snippets have been **removed** to avoid accidental exposure.

## Requirements
See `requirements.txt`. GPU acceleration is recommended for Gemma and Qwen.

## License
MIT © 2025 Pritam Deka


## Evaluation scripts
Two evaluation modules are included (converted from your scripts, now with CLIs):

### 1) Flow-based (strict + partial matches)
```bash
python -m src.eval.flow_fair_eval \
  --gold-folder /path/to/gold_folder \
  --pred-folder /path/to/pred_folder \
  --cutoff 0.7 \
  --out-dir outputs/eval_flows
```

This produces CSVs:
- `bpmn_comparison_name_only.csv`
- `bpmn_comparison_type_only.csv`
- `bpmn_comparison_relations.csv`
- `bpmn_comparison_name_type.csv`

### 2) Graph-based (degree histograms, LCS, semantic similarity)
```bash
python -m src.eval.graph_eval \
  --gold-folder /path/to/gold_folder \
  --pred-folder /path/to/pred_folder \
  --out-csv outputs/bpmn_eval_fair.csv
```
