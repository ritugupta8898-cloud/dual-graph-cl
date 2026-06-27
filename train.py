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

SEED = 7


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

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
            Ak_averaged += AK.mean(dim=0).detach()
    Ak_averaged/=len(loader)
    importance_memory = torch.maximum(importance_memory,Ak_averaged).detach()
    normalized_importance = importance_memory / (importance_memory.max() + 1e-8)
    supression = torch.clamp(1.0 - normalized_importance, min=0.1).detach()
    print(f"Task {task_id} | importance_memory: min={importance_memory.min():.4f} max={importance_memory.max():.4f} mean={importance_memory.mean():.4f}")
    print(f"Task {task_id} | supression: min={supression.min():.4f} max={supression.max():.4f} mean={supression.mean():.4f}")

    print(f"task {task_id} done, last batch loss: {loss.item():.4f}")

    # eval on all tasks seen so far
    for eval_task_id in range(task_id + 1):
        acc_matrix[task_id, eval_task_id] = evaluate(eval_task_id, test_loaders[eval_task_id])

print(acc_matrix)

@torch.no_grad()
def evaluate_all_mixed(model, full_test_dataset, device, num_tasks, classes_per_task):
    model.eval()
    loader = DataLoader(full_test_dataset, batch_size=256, shuffle=False)
    correct_oracle = 0
    correct_inferred = 0
    total = 0

    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        batch_size = x.size(0)

        true_task_ids = y // classes_per_task
        z_oracle = torch.zeros(batch_size, num_tasks, device=device)
        z_oracle[torch.arange(batch_size), true_task_ids] = 1.0

        out_oracle, _, _ = model(x, z_oracle)
        pred_oracle = out_oracle.argmax(dim=-1)
        correct_oracle += (pred_oracle == y).sum().item()

        all_logits = torch.full((batch_size, num_tasks * classes_per_task), float('-inf'), device=device)

        for t in range(num_tasks):
            z_infer = torch.zeros(batch_size, num_tasks, device=device)
            z_infer[:, t] = 1.0
            out_infer, _, _ = model(x, z_infer)
            
            lo, hi = t * classes_per_task, (t + 1) * classes_per_task
            all_logits[:, lo:hi] = out_infer[:, lo:hi]

        pred_inferred = all_logits.argmax(dim=-1)
        correct_inferred += (pred_inferred == y).sum().item()
        total += y.size(0)

    acc_oracle = correct_oracle / total
    acc_inferred = correct_inferred / total
    
    print("\n=== Global Mixed Accuracy (All 10 Classes) ===")
    print(f"Oracle Accuracy:     {acc_oracle:.4f}")
    print(f"Inferred Accuracy:   {acc_inferred:.4f}")
    
    return acc_oracle, acc_inferred
evaluate_all_mixed(model, test_full, device, NUM_TASKS, classes_per_task)