# Dual-Graph Mechanical Neuromodulation

This is the implementation of my attempt at solving the catastrophic forgetting problem in neural networks. 

Normally, when a neural network learns a new task, it overwrites the weights it used for the old tasks. This means it completely forgets the first thing it learned. This project tries to stop that from happening by using two separate graphs working together, without using replay buffers or hard parameter freezing.

---

## How it Works

Think of a neural network like a city grid. When learning a new task, standard networks just bulldoze the old roads to build new ones. I fix this by splitting the network into two parts:

* **The Main Graph (`g_main`)**: This is the standard network that actually looks at the data and makes the predictions.
* **The Control Graph (`g_mod`)**: This graph doesn't predict anything at all. It just acts like a heat map to protect the Main Graph.

When the network finishes learning a task, the Control Graph looks at which neurons were used the most. It generates a heat score for those specific neurons and spreads that importance out with diffusion. 

If a neuron is hot (meaning it is highly important for the old task), we drastically shrink its learning rate. This physically protects it so the new task cannot overwrite it. If a neuron is cold (unused), we leave its learning rate high so it can easily learn the next task. 

Instead of freezing the whole network, we just softly lower the learning rate on the important roads and let the network build on the empty lots.

---

## The Results

I tested this on a 5-task Split-MNIST benchmark, with a fixed-seed baseline-vs-suppression comparison run across 5 different seeds (42, 7, 0, 127, 63) to make sure the effect was real and not just luck on one run.

### Task 0 Retention
*(Accuracy on task 0 after training all 5 tasks)*

| Seed | Baseline | Dual-Graph (Suppression) |
| :--- | :------- | :----------------------- |
| 42   | 0.9754   | 0.9792                   |
| 7    | 0.5480   | 0.5489                   |
| 0    | 0.9915   | 0.9939                   |
| 127  | 0.9872   | 0.9976                   |
| 63   | 0.9872   | 0.9891                   |

The Dual-Graph model matched or beat the baseline on task 0 retention in all 5 seeds tested. The effect is small but consistent.

### Task 1 Retention
*(Accuracy on task 1 after training all 5 tasks)*

| Seed | Baseline | Dual-Graph (Suppression) |
| :--- | :------- | :----------------------- |
| 42   | 0.7267   | 0.7380                   |
| 7    | 0.6817   | 0.6944                   |
| 0    | 0.5485   | 0.7365                   |
| 127  | 0.8408   | 0.7708                   |
| 63   | 0.6479   | 0.8820                   |

Here the Dual-Graph model won in 4 out of 5 seeds, including two large gains (seed 0: **+19 points**, seed 63: **+23 points**). Seed 127 went the other way, where the baseline actually did better. I haven't figured out why that seed behaves differently yet — that's an open question, not something I'm sweeping under the rug.

### Class-Incremental Accuracy
*(Unmasked, all 10 digits at once)*

| Seed | Oracle | Inferred |
| :--- | :----- | :------- |
| 7    | 0.185  | 0.190    |
| 0    | 0.370  | 0.376    |
| 127  | 0.466  | 0.462    |
| 63   | 0.454  | 0.451    |

This is the harder, more honest test — no hints about which task an image belongs to. Performance here is weak and bounces around a lot depending on seed. This is the part of the project that still needs real work; the task-incremental numbers above look better than they should partly because the evaluation tells the model which 2 classes to choose between.

---

## Honest Takeaways

* The suppression mechanism has a real, replicated protective effect on the earliest task, and a mostly-consistent (4/5 seed) effect on the next one.
* It is not yet a complete solution to catastrophic forgetting — later tasks and unmasked classification still need work.
* Seed 127 breaking the pattern on task 1 is unexplained and worth digging into before claiming anything stronger.

## Next Steps

* Figure out why seed 127 behaves differently on task 1.
* Try a harder benchmark (e.g., Permuted-MNIST) with more interference between tasks, where a real protective effect should be easier to see.
* Fix the unmasked/class-incremental accuracy — likely needs some kind of output calibration, since training with masked cross-entropy biases the logits toward whichever task was learned most recently.
* Hold off on bigger architecture changes (dynamic graph growth, decoupling the control graph) until the above is sorted out.

---

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
