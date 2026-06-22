dual-graph-neuromodulation/
│
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── train.py
├── evaluate.py
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── split_mnist.py
│   ├── permuted_mnist.py
│   └── transforms.py
│
├── models/
│   ├── __init__.py
│   ├── graph_builder.py      👈 START HERE
│   ├── diffusion.py
│   ├── control_graph.py
│   ├── main_graph.py
│   ├── dual_graph.py
│   └── layers.py
│
├── losses/
│   ├── __init__.py
│   ├── task_loss.py
│   ├── sparsity_loss.py
│   └── total_loss.py
│
├── engine/
│   ├── __init__.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── continual.py
│
├── metrics/
│   ├── __init__.py
│   ├── accuracy.py
│   ├── forgetting.py
│   ├── entropy.py
│   ├── cosine.py
│   ├── sparsity.py
│   └── utilization.py
│
├── visualization/
│   ├── __init__.py
│   ├── heatmaps.py
│   ├── routing.py
│   ├── diffusion.py
│   └── plots.py
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── checkpoint.py
│   ├── seed.py
│   └── graph_utils.py
│
├── experiments/
│   ├── experiment_v0.py
│   ├── split_mnist.py
│   ├── permuted_mnist.py
│   └── configs.py
│
├── docs/
│   ├── architecture.md
│   ├── research_log.md
│   ├── design_decisions.md   👈 (We'll add this)
│   └── todo.md
│
├── outputs/
│   ├── figures/
│   ├── heatmaps/
│   ├── modulation/
│   └── routing/
│
├── checkpoints/
├── logs/
├── configs/
├── notebooks/
├── papers/
├── scripts/
└── tests/