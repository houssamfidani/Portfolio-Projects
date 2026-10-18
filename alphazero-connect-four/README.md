# AlphaZero-inspired Connect Four

A Connect Four agent combining a PyTorch policy/value network with Monte Carlo Tree Search. Self-play generates training examples; the trained network then guides search against random and tactical opponents. A Pygame interface allows human play.

**Read:** [report (French)](rapport.pdf) · [experiment results](src/resultats_experiences.txt) · [network](src/model.py) · [MCTS](src/mcts.py)

## Recorded results

Each configuration was evaluated against a tactical heuristic over 100 games: 50 as the first player and 50 as the second. Evaluation used 50 MCTS simulations per move.

| Configuration | Self-play games | Training simulations/move | Epochs | Wins vs. heuristic |
|---|---:|---:|---:|---:|
| 1 | 100 | 50 | 10 | 51/100 |
| 2 | 200 | 100 | 20 | 67/100 |
| 3 | 300 | 150 | 50 | **70/100** |

The heuristic takes an immediate win, blocks an immediate loss, and otherwise chooses randomly. It is not an optimal opponent. The implementation uses one self-play generation stage followed by training, rather than repeated AlphaZero training cycles. Several settings change together, so these runs do not isolate each setting's effect.

## Run

From this directory, install dependencies in a Python environment:

```bash
python -m pip install -r requirements.txt
cd src
python interface.py
```

Click a column to play; press **R** to restart. The interface loads the included `model_3.pth` checkpoint.

To evaluate that checkpoint:

```bash
python -c "import test; test.tester_model('model_3.pth', 50, 50, 'ameliore')"
```

`python experiences.py` reruns all three training/evaluation configurations and overwrites their checkpoints, plots, and result file. Randomness is not fixed, so scores can vary. Original code and report are preserved.
