# Language model preference optimization

Two independent adaptations of **Qwen3-0.6B** explore how the training signal shapes the model's responses. DPO learns from preferred and rejected French answers. GRPO samples arithmetic solutions and rewards both a machine-readable answer and its correctness.

[Open the notebook](language_model_preference_optimization.ipynb) · [Recorded training results](results/recorded-results.json)

| Experiment | Training data | LoRA rank | Optimizer steps |
|---|---|---:|---:|
| French DPO | 981 preference pairs after length filtering | 32 | 246 |
| GSM8K GRPO | 200 arithmetic problems, two completions per prompt | 64 | 50 |

## What the examples show

On an age problem, the GRPO model computes the correct answer, 19, but encloses it in a LaTeX box. The reward function expects an `<answer>` tag and misses it. On a shopping problem it uses the expected tag but gives the wrong change. These examples illustrate why response format and arithmetic accuracy are scored separately.

The notebook contains the base and adapted generations and an evaluation on the first 200 GSM8K test items. The saved run ends before either final test score. The recorded training losses come from different objectives.

## Run

From this directory with a CUDA GPU and a compatible PyTorch installation:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab language_model_preference_optimization.ipynb
```

The notebook downloads the base model and both datasets. Running the training cells writes adapters and merged checkpoints locally. The evaluation cells save predictions in `evaluations/`. Hosted experiment tracking is disabled.

## Sources

The experiment design draws on material by Amr Mohamed and Yang Zhang, with Michalis Vazirgiannis. The model and datasets are available from their publishers. [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · [French Orca preference pairs](https://huggingface.co/datasets/AIffl/french_orca_dpo_pairs) · [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
