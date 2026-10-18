# Machine learning portfolio

**Houssam Fidani** · MVA, ENS Paris-Saclay

Four projects covering neural language models, reinforcement learning, and Bayesian inference. Each includes the implementation, recorded results, and a short explanation of what those results establish.

| Project | Focus | Recorded result |
|---|---|---|
| [LLM fine-tuning and tool calling](llm-finetuning-and-tool-calling/) | LoRA, 4-bit quantization, response masking, function execution | Summarization validation loss **2.331 → 2.171** on 50 examples; separate tool-calling demonstrations |
| [AlphaZero-inspired Connect Four](alphazero-connect-four/) | PyTorch policy/value network, MCTS, self-play | **70 wins in 100 games** against a tactical heuristic, balanced across player positions |
| [Attention-based translation](attention-based-translation/) | GRU encoder-decoder, global attention, error analysis | Validation loss **4.60 → 2.76** over 20 epochs; attention analysis using a separate reference checkpoint |
| [Bayesian mosquito modeling](bayesian-mosquito-modeling/) | Hierarchical Poisson model, custom HMC, posterior predictive checks | 20,000 HMC iterations; predictive checks identify limited fit at the observation level |

Start with a project's README, then open its notebook or PDF report. Setup commands are listed per project. The notebooks retain selected outputs so they can be reviewed without running training.

[LinkedIn](https://www.linkedin.com/in/houssam-fidani)
