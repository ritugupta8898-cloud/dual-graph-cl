import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

from models.dual_graph import DualGraphNetwork
from models.graph_builder import build_graph
#setting the seeds 
import random
import numpy as np

SEED = 63

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # harmless if no GPU

set_seed(SEED)
# ---------- data ----------
transform = transforms.ToTensor()
train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

NUM_TASKS = 5
classes_per_task = 10 // NUM_TASKS
def split_tasks(dataset, num_tasks=NUM_TASKS):
    tasks = []
    classes_per_task = 10 // num_tasks
    for t in range(num_tasks):
        lo, hi = t * classes_per_task, (t + 1) * classes_per_task
        idx = [i for i, (_, y) in enumerate(dataset) if lo <= y < hi]
        tasks.append(Subset(dataset, idx))
    return tasks
def masked_logits(logits, task_id, classes_per_task):
    mask = torch.full_like(logits, float('-inf'))
    lo, hi = task_id * classes_per_task, (task_id + 1) * classes_per_task
    mask[:, lo:hi] = 0
    return logits + mask
train_task = split_tasks(train_full)
test_task = split_tasks(test_full)

train_loaders = [DataLoader(t, batch_size=128, shuffle=True) for t in train_task]
test_loaders = [DataLoader(t, batch_size=256, shuffle=False) for t in test_task]

# ---------- model ----------
input_dim = 28 * 28
hidden_dims = [128, 64]
output_dim = 10
context_dim = NUM_TASKS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

W = build_graph(hidden_dims)
W = W.to(device)

model = DualGraphNetwork(input_dim, hidden_dims, output_dim, context_dim, W,
                          lambda_diff=0.5, K=5).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ---------- eval function ----------
@torch.no_grad()
def evaluate(task_id, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        z = torch.zeros(x.size(0), NUM_TASKS, device=device)
        z[:, task_id] = 1.0
        out, AK, final_h = model(x, z)
        out = masked_logits(out, task_id, classes_per_task=classes_per_task)
        pred = out.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

# ---------- training loop ----------
EPOCHS_PER_TASK = 15
acc_matrix = torch.zeros(NUM_TASKS, NUM_TASKS)
importance_memory = torch.zeros(192, device=device)

supression = None
for task_id, loader in enumerate(train_loaders):
    model.train()
    Ak_averaged = torch.zeros(192,device=device)
    for epoch in range(EPOCHS_PER_TASK):
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            z = torch.zeros(x.size(0), NUM_TASKS, device=device)
            z[:, task_id] = 1.0
            out, AK, final_h = model(x, z)
            out = masked_logits(out, task_id, classes_per_task=classes_per_task)
            loss = F.cross_entropy(out, y)
            loss.backward()
            if supression is not None:
             with torch.no_grad():
              offset = 0
              for layer in model.g_main.layers:
                h_dim = layer.out_features
                s_slice = supression[offset : offset + h_dim].unsqueeze(1)
                if layer.weight.grad is not None:
                    layer.weight.grad *= s_slice
                if layer.bias.grad is not None:
                    layer.bias.grad *= s_slice.squeeze()
                offset += h_dim

            optimizer.step()
    

    print(f"task {task_id} done, last batch loss: {loss.item():.4f}")

    # eval on all tasks seen so far
    for eval_task_id in range(task_id + 1):
        acc_matrix[task_id, eval_task_id] = evaluate(eval_task_id, test_loaders[eval_task_id])

print(acc_matrix)