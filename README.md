# Dual-Graph Mechanical Neuromodulation

This is the implementation of my attempt at solving the catastrophic forgetting problem in neural networks. 

Normally, when a neural network learns a new task, it overwrites the weights it used for the old tasks. This means it completely forgets the first thing it learned. This project  tries stops that from happening by using two separate graphs working together, without using replay buffers or hard parameter freezing.

## How it Works

Think of a neural network like a city grid. When learning a new task, standard networks just bulldoze the old roads to build new ones. I fix this by splitting the network into two parts:

1. The Main Graph or referred as g_main: This is the standard network that actually looks at the data and makes the predictions.
2. The Control Graph or referred as g_mod: This graph doesn't predict anything at all. It just acts like a heat map to protect the Main Graph.

When the network finishes learning a task, the Control Graph looks at which neurons were used the most. It generates a heat score for those specific neurons and spreads that importance out with diffusion . 

If a neuron is hot (meaning it is highly important for the old task), we drastically shrink its learning rate. This physically protects it so the new task cannot overwrite it. If a neuron is cold (unused), we leave its learning rate high so it can easily learn the next task. 

Instead of freezing the whole network, we just softly lower the learning rate on the important roads and let the network build on the empty lots.

## The Results

I tested this on a 5-task Split-MNIST benchmark. 

Dual-Graph Model (High Memory):
- Task 0 started at 96.88% and ended at 96.74% even after learning 4 completely new tasks. 
- However this was at this stage not consistent across all tests that problem will be solved later 

Standard Baseline Network (Failed):
- Task 0 got stuck at 79%.
- Task 4 got stuck at 58%.
- These were found to ve consistent across various reruns
- Without our Control Graph, the network just overwrote its own weights and became mediocre at everything.


## How to Run It

1. Install the requirements:
pip install -r requirements.txt

2. Run the Experimental Dual-Graph Model:
python3 train.py

3. Run the Standard Baseline (To watch it fail):
python3 train_baseline.py