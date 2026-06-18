import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ── 复用 train_pinn.py 里已有的东西 ──
from models.pinn import PINNMultiClass   # 只是借用 FaultDataset 的写法


class FaultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ── MLP 基线（无状态头、无物理损失，ReLU）──
class MLPBaseline(nn.Module):
    def __init__(self, in_dim=156, hidden=128, depth=3, n_classes=5, dropout=0.1):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * depth
        for i in range(depth):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_v, y_v in loader:
            X_v, y_v = X_v.to(device), y_v.to(device)
            preds    = torch.argmax(model(X_v), dim=1)
            correct += (preds == y_v).sum().item()
            total   += len(y_v)
    return correct / total


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[设备] {device}")
    os.makedirs('outputs/checkpoints', exist_ok=True)

    # ── 加载数据（与 train_pinn.py 完全相同）──
    data_path = 'final_data/ieee39_final_dataset.csv'
    print(f"\n加载 {data_path} ...")
    df = pd.read_csv(data_path)

    fault_type_map = {'Normal': 0, '3LG': 1, 'LG': 2, 'LLG': 3, 'LL': 4}
    vm_cols = [f'vm_{i}' for i in range(39)]
    va_cols = [f'va_{i}' for i in range(39)]
    p_cols  = [f'p_{i}'  for i in range(39)]
    q_cols  = [f'q_{i}'  for i in range(39)]
    fcols   = vm_cols + va_cols + p_cols + q_cols

    X_all = df[fcols].values.astype(np.float32)
    y_all = df['fault_type'].map(fault_type_map).values.astype(np.int64)
    print(f"样本数: {len(X_all):,}  特征: {X_all.shape[1]}")

    # ── 与 train_pinn.py 完全相同的切分 ──
    np.random.seed(42)
    idx    = np.random.permutation(len(X_all))
    n_val  = int(0.1 * len(X_all))
    n_test = int(0.1 * len(X_all))
    v_idx  = idx[:n_val]
    te_idx = idx[n_val:n_val + n_test]
    t_idx  = idx[n_val + n_test:]

    BATCH = 1024
    train_loader = DataLoader(FaultDataset(X_all[t_idx], y_all[t_idx]),
                              batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(FaultDataset(X_all[v_idx], y_all[v_idx]),
                              batch_size=2048, shuffle=False, num_workers=0)
    print(f"训练: {len(t_idx):,}  验证: {len(v_idx):,}  测试: {len(te_idx):,}")

    # ── 模型 ──
    model     = MLPBaseline(in_dim=156, hidden=128, depth=3, n_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                      factor=0.5, min_lr=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    # ── 训练 ──
    EPOCHS   = 100
    PATIENCE = 20
    best_val_acc = 0.0
    wait = 0
    t0   = time.time()

    print("\n" + "=" * 55)
    print("MLP 基线训练（无物理损失）")
    print("=" * 55)

    for epoch in range(EPOCHS):
        model.train()
        sum_loss = n_correct = n_total = 0

        if TQDM_AVAILABLE:
            bar = tqdm(train_loader, desc=f"E{epoch+1:3d}/{EPOCHS}",
                       ncols=90, leave=False)
        else:
            bar = train_loader

        for X_b, y_b in bar:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sum_loss  += loss.item()
            n_correct += (logits.argmax(1) == y_b).sum().item()
            n_total   += len(y_b)

        avg_loss  = sum_loss / len(train_loader)
        train_acc = n_correct / n_total
        val_acc   = evaluate(model, val_loader, device)
        scheduler.step(avg_loss)
        lr_now = optimizer.param_groups[0]['lr']

        elapsed = time.time() - t0
        eta     = elapsed / (epoch + 1) * (EPOCHS - epoch - 1)
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.4f} | "
              f"Val: {val_acc:.4f} | "
              f"lr: {lr_now:.2e} | "
              f"ETA: {eta/60:.1f}min")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(),
                       'outputs/checkpoints/mlp_baseline.pth')
            print(f"         [SAVED] val acc: {best_val_acc:.4f}")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\n早停（连续 {PATIENCE} 轮无提升）")
                break

    total = time.time() - t0
    print("\n" + "=" * 55)
    print(f"训练完成！用时 {total/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"已保存至: outputs/checkpoints/mlp_baseline.pth")
    print("=" * 55)
    print("\n现在可以运行: python evaluate.py")


if __name__ == '__main__':
    train()
