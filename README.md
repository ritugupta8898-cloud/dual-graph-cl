# Dual-Graph Mechanical-CL

This is the implementation of my attempt at solving catastrophic forgetting in neural networks. 

Normally, when a neural network learns a new task, it overwrites the weights used for previous tasks. This project attempts to prevent this by using two separate graphs that work together to protect important weights, without relying on replay buffers or freezing the entire network.

## How it Works

The network is split into two distinct parts:

1. **The Main Graph (`g_main`)**: The primary network (an MLP with hidden dimensions [128, 64]) that processes data and makes predictions.
2. **The Control Graph (`g_mod`)**: A secondary graph with 192 control nodes that does not make predictions. Its role is to track node usage and protect the Main Graph.

After the network finishes learning a task, the Control Graph checks which neurons were most active. It assigns a per-node importance score based on the maximum of its historical importance and the average activation for the current task.

If a neuron is important for an old task, we reduce its learning rate on the Main Graph by scaling its gradient updates by `clamp(1 - importance, min=0.1)`. This protects the neuron from being overwritten by the new task. If a neuron is unused, its learning rate remains high so it can learn the new data.

## Experimental Setup

The mechanism was tested against a standard baseline using a multi-seed evaluation.

* **Benchmark:** Split-MNIST, 5 tasks, 2 classes/task.
* **Setting:** Task-incremental (task ID known at evaluation via masking).
* **Optimizer:** SGD (lr=0.01, momentum=0.9), 15 epochs/task.
* **Seeds:** 5 fixed seeds (42, 7, 0, 127, 63). The baseline and the suppression models used the same seeds for a direct comparison.

## v1 Evaluation Results

Multi-seed testing showed consistent retention on early tasks, though unmasked class-incremental performance needs improvement.

### Task 0 Retention
*Accuracy on Task 0 after sequentially training on tasks 0 through 4.*

| Seed | Baseline (Final) | Suppression (Final) |
| :--- | :--- | :--- |
| 42 | 0.9754 | 0.9792 |
| 7 | 0.5480 | 0.5489 |
| 0 | 0.9915 | 0.9939 |
| 127 | 0.9872 | 0.9976 |
| 63 | 0.9872 | 0.9891 |

**Finding:** Consistent, small, real positive effect. Suppression matches or exceeds the baseline on Task 0 retention across all 5 out of 5 seeds.

### Task 1 Retention
*Accuracy on Task 1 after sequentially training on tasks 1 through 4.*

| Seed | Baseline (Final) | Suppression (Final) |
| :--- | :--- | :--- |
| 42 | 0.7267 | 0.7380 |
| 7 | 0.6817 | 0.6944 |
| 0 | 0.5485 | 0.7365 |
| 127 | 0.8408 | 0.7708 |
| 63 | 0.6479 | 0.8820 |

**Finding:** A mostly-consistent positive effect. Suppression wins on 4 out of 5 seeds (only seed 127 favors the baseline). 

### Mixed (Unmasked, 10-way) Accuracy
*Class-incremental setting without handing the model the task ID (Seed 42 omitted as the print statement was added later).*

| Seed | Oracle | Inferred |
| :--- | :--- | :--- |
| 7 | 0.1850 | 0.1900 |
| 0 | 0.3700 | 0.3760 |
| 127 | 0.4660 | 0.4620 |
| 63 | 0.4540 | 0.4510 |

**Finding:** Weak and seed-dependent. Without the masking shortcut (which zeroes out 8 of 10 classes at eval time), real 10-way performance is variable and low. 

## Honest Conclusion

1.  **Clear Retention Benefits:** The suppression mechanism helps retain early tasks. It protected Task 0 in all 5 seeds and Task 1 in 4 out of 5 seeds compared to the baseline.
2.  **Class-Incremental Weakness:** Unmasked classification performance remains weak. While the mechanism mitigates forgetting within the task-incremental framework, it does not yet solve catastrophic forgetting in a fully unmasked setting.

## Next Steps 

* **Harder Benchmarks:** Test on a dataset with more inter-task interference (e.g., Permuted-MNIST) where a real protective effect would be easier to magnify and detect.
* **Investigate Task Saturation:** Analyze the node importance scores to see if the memory is saturating too quickly, which might be suppressing later tasks' learning rather than protecting them.
* **Address Class-Incremental Performance:** Look into output-layer calibration, as masked cross-entropy training currently induces recency-biased logits.
* **Scope Management:** Pause architectural expansions (like dynamic graph growth) until the class-incremental performance is addressed.

## How to Run It

1. Install the requirements:
pip install -r requirements.txt

2. Run the Experimental Dual-Graph Model:
python3 train.py

3. Run the Standard Baseline (To watch it fail):
python3 train_baseline.py
