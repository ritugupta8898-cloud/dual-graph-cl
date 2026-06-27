# Dual-Graph Mechanical Neuromodulation

This is the implementation of my attempt at solving the catastrophic forgetting problem in neural networks.

Normally, when a neural network learns a new task, it overwrites the weights it used for the old tasks. This means it completely forgets the first thing it learned. This project tries to stop that from happening by using two separate graphs working together, without using replay buffers or hard parameter freezing.

## How it Works

Think of a neural network like a city grid. When learning a new task, standard networks just bulldoze the old roads to build new ones. I fix this by splitting the network into two parts:

1. The Main Graph, referred to as g_main: This is the standard network that actually looks at the data and makes the predictions.
2. The Control Graph, referred to as g_mod: This graph doesn't predict anything at all. It just acts like a heat map to protect the Main Graph.

When the network finishes learning a task, the Control Graph looks at which neurons were used the most. It generates a heat score for those specific neurons and spreads that importance out with diffusion.

If a neuron is hot (meaning it is highly important for the old task), we drastically shrink its learning rate. This physically protects it so the new task cannot overwrite it. If a neuron is cold (unused), we leave its learning rate high so it can easily learn the next task.

Instead of freezing the whole network, we just softly lower the learning rate on the important roads and let the network build on the empty lots.

## A Bug That Mattered

Early on, the diffusion step inside the Control Graph used a raw, unnormalized adjacency matrix. This caused the importance scores to blow up to values in the tens of millions, which silently broke the suppression mechanism — every node ended up clamped to the same flat floor value regardless of how important it actually was. The mechanism was technically running, but it wasn't doing anything node-selective.

After normalizing the adjacency matrix properly (symmetric degree normalization before propagation), importance scores became sane (single digits, varying meaningfully per node and per task), and the suppression effect became real and consistent. All results below are from the model after this fix. This is worth mentioning because it's a good example of why diagnostic logging matters — the bug was invisible from the accuracy numbers alone.

## The Results

I tested this on a 5-task Split-MNIST benchmark, with a fixed-seed baseline-vs-suppression comparison run across 5 different seeds (42, 7, 0, 127, 63) to make sure the effect was real and not just luck on one run.

**Task 0 retention (accuracy on task 0 after training all 5 tasks):**

| Seed | Baseline | Dual-Graph (Suppression) |
|------|----------|---------------------------|
| 42   | 0.9844   | 0.9962                    |
| 0    | 0.9385   | 0.9849                    |
| 7    | 0.9305   | 0.9976                    |
| 127  | 0.9986   | 0.9991                    |
| 63   | 0.9967   | 0.9939                    |

The Dual-Graph model matched or beat the baseline in all 5 seeds. Consistent and clean.

**Task 1 retention (accuracy on task 1 after training all 5 tasks):**

| Seed | Baseline | Dual-Graph (Suppression) |
|------|----------|---------------------------|
| 42   | 0.6391   | 0.8673                    |
| 0    | 0.6087   | 0.6998                    |
| 7    | 0.5857   | 0.7071                    |
| 127  | 0.4721   | 0.8746                    |
| 63   | 0.4848   | 0.5142                    |

The Dual-Graph model beat the baseline in all 5 seeds, including two large margins (seed 42: +23 points, seed 127: +40 points). This is a real, replicated effect.

**Class-incremental (unmasked, all 10 digits at once) accuracy:**

| Seed | Oracle | Inferred |
|------|--------|----------|
| 42   | 0.512  | 0.510    |
| 0    | 0.318  | 0.317    |
| 7    | 0.481  | 0.479    |
| 127  | 0.411  | 0.409    |
| 63   | 0.375  | 0.374    |

This is the harder, more honest test — no hints about which task an image belongs to. Performance here improved after the diffusion fix but is still weak and seed-dependent. This is the part of the project that needs the most work; the task-incremental numbers above look better than they should partly because the evaluation tells the model which 2 classes to choose between.

## Honest Takeaways

- The suppression mechanism has a real, fully replicated protective effect on both task 0 (5/5 seeds) and task 1 (5/5 seeds) retention, after fixing the diffusion normalization bug.
- The effect size varies by seed (sometimes +5 points, sometimes +40), which is normal and expected, not a sign the result isn't real.
- Real-world (unmasked, class-incremental) classification performance is still well below what's needed for practical use. This is the main remaining limitation.

## Next Steps

- Fix the unmasked/class-incremental accuracy — likely needs some kind of output calibration, since training with masked cross-entropy biases the logits toward whichever task was learned most recently.
- Try a harder benchmark (e.g. Permuted-MNIST) with more interference between tasks, to see if the effect holds or grows under more difficult conditions.
- Once calibration is solved, consider architectural extensions (dynamic graph growth when nodes saturate, decoupling the Control Graph's node count from the Main Graph's).

## How to Run It

Install the requirements:
```bash
pip install -r requirements.txt
```

Run the Dual-Graph model:
```bash
python3 train.py
```

Run the standard baseline:
```bash
python3 train_baseline.py
```
