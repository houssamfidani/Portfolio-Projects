# LLM fine-tuning and tool calling

Two separate LoRA adapters for Qwen3.5-0.8B-Base: French news summarization and structured function calling. The project covers quantized loading, prompt formatting, response-only loss, training, and application-side tool execution.

**[Open the notebook](llm_finetuning_and_tool_calling.ipynb)** · [Recorded metrics](results/recorded-metrics.json) · [Tool outputs before](results/tool-calling-before.txt) / [after](results/tool-calling-after.txt)

## Recorded results

| Experiment | Setup | Observation |
|---|---|---|
| Summarization | 100 optimizer steps; 50 held-out examples | Validation loss **2.331 → 2.171** |
| Tool calling | Separate adapter; 50 optimizer steps | In three examples, the model used a weather tool and declined an unsupported booking request |

Both runs used a single NVIDIA T4, NF4 4-bit weights, FP16 computation, and rank-32 LoRA. The adapters had 21.65M trainable parameters, about 2.80% of the combined parameter count. Training processed approximately 800 summarization examples and 400 tool conversations, less than one epoch in each case.

The tool returns fixed demonstration data. Evaluation is limited to 50 summarization examples and three tool prompts; the umbrella advice remained weak despite correct tool use. A larger held-out benchmark would be needed to assess factual accuracy and tool-use reliability.

## Run

Use a fresh Python 3.11+ environment with a CUDA GPU. From this directory:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab llm_finetuning_and_tool_calling.ipynb
```

Run cells in order to download the model and datasets, train both adapters, and compare generations. The main training-library pins match the original setup. Saved outputs can be read without a GPU; trained adapters are not bundled. New weights are written to `outputs/`, which is ignored by Git. Optional FP16 merging requires additional memory and is disabled by default.
