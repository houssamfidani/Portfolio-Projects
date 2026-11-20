# Machine learning portfolio

**Houssam Fidani** · MVA, ENS Paris-Saclay · [LinkedIn](https://www.linkedin.com/in/houssam-fidani)

Eight projects across computer vision, language modeling, reinforcement learning, and Bayesian inference. Each project includes its implementation, recorded results, and instructions for running it.

## Computer vision

| Project | Method | Recorded result |
|---|---|---|
| [Visual instance retrieval](visual-instance-retrieval/) | SIFT, bag of visual words, geometric verification | For one query against 1,611 paintings, the two matching images had **69 and 67 inliers**. The closest incorrect candidate had 14. |
| [Neural networks for image classification](neural-networks-for-image-classification/) | NumPy backpropagation, decision boundaries, PyTorch classifiers | On CIFAR-10, a CNN reached **65.08%** test accuracy after 20 epochs. A linear pixel classifier reached **41.36%** after 10. |

## Natural language processing and language models

| Project | Method | Recorded result |
|---|---|---|
| [Legal document classification](legal-text-classification/) | TF-IDF baselines, full JuriBERT fine-tuning, zero-shot classification | On 5,571 held-out decisions, LinearSVC reached **0.970 macro-F1** and JuriBERT reached **0.958**. |
| [LLM fine-tuning and tool calling](llm-finetuning-and-tool-calling/) | Quantized LoRA training, response masking, application-side tool execution | Summarization validation loss fell from **2.331 to 2.171** on 50 examples. A separate adapter was tested on three tool prompts. |
| [Language model preference optimization](language-model-preference-optimization/) | DPO on French response pairs and GRPO on GSM8K | The runs completed **246 DPO steps** and **50 GRPO steps**. Saved examples show how a correct arithmetic answer can miss the format reward. |
| [Attention-based translation](attention-based-translation/) | English-to-French GRU encoder-decoder with global attention | Validation loss fell from **4.60 to 2.76** over 20 epochs. Alignment plots use a separate reference checkpoint. |

## Reinforcement learning and planning

| Project | Method | Recorded result |
|---|---|---|
| [AlphaZero-inspired Connect Four](alphazero-connect-four/) | Policy/value network, self-play, Monte Carlo tree search | The strongest configuration won **70 of 100 games** against a tactical heuristic, playing 50 games from each starting position. |

## Bayesian inference

| Project | Method | Recorded result |
|---|---|---|
| [Bayesian mosquito modeling](bayesian-mosquito-modeling/) | Hierarchical Poisson model, Hamiltonian Monte Carlo, posterior predictive checks | The 95% posterior predictive intervals covered **all 13 annual totals** and **46 of 84 region-year observations**. |
