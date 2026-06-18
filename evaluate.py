"""
evaluate.py — 独立评估脚本
用途：加载已训练好的 PINN/MLP 模型，生成并保存：
  1. 混淆矩阵（图 + CSV）
  2. 每类 Precision / Recall / F1（图 + CSV）
  3. 消融实验：PINN vs MLP，不同噪声 σ 下准确率（图 + CSV）
  4. 汇总 JSON

运行方式（放在项目根目录，和 train_pinn.py 同级）：
  python evaluate.py
  python evaluate.py --mlp_model outputs/checkpoints/mlp_baseline.pth
  python evaluate.py --skip_ablation   # 没有 MLP 模型时跳过消融
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 中文字体配置 ──
def _set_chinese_font():
    # 按优先级依次尝试常见中文字体
    candidates = [
        "SimHei", "Microsoft YaHei", "STHeiti", "PingFang SC",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    else:
        # 都没有就下载一个（Linux 服务器常用方案）
        print("[WARN] 未找到中文字体，图中中文可能显示为方块")
        print("       Linux 可运行: apt-get install fonts-wqy-microhei")
        print("       或: pip install matplotlib 后手动放字体")
    matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示

_set_chinese_font()
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

# ── 把项目根目录加到 path，保证能 import 自己的模块 ──
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from models.pinn import PINNMultiClass
from utils import build_Ybus          # 只用来加载 Ybus，evaluate 本身不算物理 loss

# ──────────────────────────────────────────────
# 常量（与 train_pinn.py 保持完全一致）
# ──────────────────────────────────────────────
FAULT_CLASSES   = ["Normal", "3LG", "LG", "LLG", "LL"]
N_CLASSES       = len(FAULT_CLASSES)
FAULT_TYPE_MAP  = {'Normal': 0, '3LG': 1, 'LG': 2, 'LLG': 3, 'LL': 4}
RANDOM_SEED     = 42
DATA_PATH       = 'final_data/ieee39_final_dataset.csv'
PINN_CKPT       = 'outputs/checkpoints/best_model.pth'
OUTPUT_DIR      = Path('outputs/eval_results')
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────
# MLP 基线（与 models/pinn.py 里保持一致）
# ──────────────────────────────────────────────
class MLPBaseline(nn.Module):
    def __init__(self, in_dim=156, hidden=128, depth=5, n_classes=5, dropout=0.1):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * depth
        for i in range(depth):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x), None, None


# ──────────────────────────────────────────────
# 数据加载：与 train_pinn.py 完全相同的切分逻辑
# ──────────────────────────────────────────────
def load_test_set(data_path=DATA_PATH):
    """
    读取原始 CSV，用相同 seed=42、8:1:1 切分，
    返回测试集 (X_test: FloatTensor, y_test: LongTensor)
    """
    print(f"[数据] 加载 {data_path} ...")
    df = pd.read_csv(data_path)

    vm_cols = [f'vm_{i}' for i in range(39)]
    va_cols = [f'va_{i}' for i in range(39)]
    p_cols  = [f'p_{i}'  for i in range(39)]
    q_cols  = [f'q_{i}'  for i in range(39)]
    fcols   = vm_cols + va_cols + p_cols + q_cols  # 156 列

    X_all = df[fcols].values.astype(np.float32)
    y_all = df['fault_type'].map(FAULT_TYPE_MAP).values.astype(np.int64)

    # ── 与 train_pinn.py 完全相同的切分 ──
    np.random.seed(RANDOM_SEED)
    idx    = np.random.permutation(len(X_all))
    n_val  = int(0.1 * len(X_all))
    n_test = int(0.1 * len(X_all))
    te_idx = idx[n_val : n_val + n_test]   # 测试集索引

    X_test = torch.tensor(X_all[te_idx], dtype=torch.float32)
    y_test = torch.tensor(y_all[te_idx], dtype=torch.long)
    print(f"  总样本: {len(X_all):,}  →  测试集: {len(te_idx):,}")
    print(f"  类别分布: {np.bincount(y_all[te_idx])}")
    return X_test, y_test


# ──────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────
def load_model(model_cls, ckpt_path, **kwargs):
    model = model_cls(**kwargs).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    print(f"  [加载] {model_cls.__name__} ← {ckpt_path}")
    return model


# ──────────────────────────────────────────────
# 推理
# ──────────────────────────────────────────────
@torch.no_grad()
def predict(model, X, noise_std=0.0, batch_size=4096):
    model.eval()
    all_pred, all_prob = [], []
    for start in range(0, len(X), batch_size):
        xb = X[start: start + batch_size].to(DEVICE)
        if noise_std > 0:
            xb = xb + torch.randn_like(xb) * noise_std
        logits, *_ = model(xb)
        prob = torch.softmax(logits, dim=-1)
        all_pred.append(prob.argmax(dim=-1).cpu().numpy())
        all_prob.append(prob.max(dim=-1).values.cpu().numpy())
    return np.concatenate(all_pred), np.concatenate(all_prob)


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────
def plot_confusion_matrix(cm, save_path, title="混淆矩阵"):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(N_CLASSES), yticks=np.arange(N_CLASSES),
           xticklabels=FAULT_CLASSES, yticklabels=FAULT_CLASSES,
           title=title, ylabel="真实标签", xlabel="预测标签")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {save_path}")


def plot_per_class_metrics(report_df, save_path):
    metrics = ["precision", "recall", "f1-score"]
    x, width = np.arange(N_CLASSES), 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(metrics):
        vals = report_df.loc[FAULT_CLASSES, m].values.astype(float)
        bars = ax.bar(x + i * width, vals, width, label=m.capitalize())
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(FAULT_CLASSES)
    ax.set_ylim(0.75, 1.05)
    ax.set_title("各类故障 Precision / Recall / F1")
    ax.set_ylabel("分值")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {save_path}")


def plot_ablation_noise(noise_levels, pinn_accs, mlp_accs, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(noise_levels, pinn_accs, "o-", label="PINN（本文方法）", linewidth=2)
    ax.plot(noise_levels, mlp_accs,  "s--", label="MLP 基线",       linewidth=2)
    for s, pa, ma in zip(noise_levels, pinn_accs, mlp_accs):
        ax.annotate(f"{pa:.1%}", (s, pa), xytext=(0,  8), textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{ma:.1%}", (s, ma), xytext=(0,-14), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("测量噪声标准差 σ (pu)")
    ax.set_ylabel("整体准确率")
    ax.set_title("消融实验：不同噪声水平下 PINN vs MLP 鲁棒性")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {save_path}")


def plot_ablation_per_class(pinn_per, mlp_per, save_path):
    x, width = np.arange(N_CLASSES), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, pinn_per, width, label="PINN（本文方法）")
    ax.bar(x + width/2, mlp_per,  width, label="MLP 基线")
    for i, (p, m) in enumerate(zip(pinn_per, mlp_per)):
        ax.text(i - width/2, p + 0.003, f"{p:.1%}", ha="center", fontsize=8)
        ax.text(i + width/2, m + 0.003, f"{m:.1%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(FAULT_CLASSES)
    ax.set_ylim(0.75, 1.05)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("消融实验：各类故障准确率（PINN vs MLP）")
    ax.set_ylabel("Accuracy"); ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {save_path}")


# ──────────────────────────────────────────────
# 评估模块
# ──────────────────────────────────────────────
def eval_classification(model, X, y_true, tag="PINN"):
    print(f"\n{'='*55}\n  [{tag}] 分类性能评估\n{'='*55}")
    y_pred, _ = predict(model, X)
    y_np = y_true.numpy()

    # 混淆矩阵
    cm = confusion_matrix(y_np, y_pred)
    pd.DataFrame(cm, index=FAULT_CLASSES, columns=FAULT_CLASSES)\
      .to_csv(OUTPUT_DIR / f"confusion_matrix_{tag}.csv")
    plot_confusion_matrix(cm, OUTPUT_DIR / f"confusion_matrix_{tag}.png",
                          title=f"混淆矩阵 [{tag}]")

    # 分类报告
    report_str = classification_report(y_np, y_pred,
                                        target_names=FAULT_CLASSES, digits=4)
    print(report_str)
    rp = OUTPUT_DIR / f"classification_report_{tag}.txt"
    rp.write_text(report_str, encoding="utf-8")
    print(f"  [保存] {rp}")

    # 逐类指标
    p, r, f1, sup = precision_recall_fscore_support(y_np, y_pred,
                                                     labels=list(range(N_CLASSES)))
    pd.DataFrame({"class": FAULT_CLASSES,
                  "precision": p, "recall": r, "f1-score": f1, "support": sup})\
      .set_index("class")\
      .to_csv(OUTPUT_DIR / f"per_class_metrics_{tag}.csv")

    report_df = pd.DataFrame(
        classification_report(y_np, y_pred, target_names=FAULT_CLASSES,
                               output_dict=True)).T
    plot_per_class_metrics(report_df, OUTPUT_DIR / f"per_class_metrics_{tag}.png")

    overall_acc    = (y_pred == y_np).mean()
    per_class_acc  = cm.diagonal() / cm.sum(axis=1)
    print(f"  整体准确率: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

    return {
        "overall_accuracy":    float(overall_acc),
        "per_class_accuracy":  {c: float(a) for c, a in zip(FAULT_CLASSES, per_class_acc)},
        "per_class_precision": {c: float(v) for c, v in zip(FAULT_CLASSES, p)},
        "per_class_recall":    {c: float(v) for c, v in zip(FAULT_CLASSES, r)},
        "per_class_f1":        {c: float(v) for c, v in zip(FAULT_CLASSES, f1)},
    }


def eval_ablation(pinn_model, mlp_model, X, y_true,
                  noise_levels=(0.0, 0.01, 0.02, 0.05)):
    print(f"\n{'='*55}\n  消融实验\n{'='*55}")
    y_np = y_true.numpy()

    # (a) 各类准确率（无噪声）
    pinn_pred, _ = predict(pinn_model, X)
    mlp_pred,  _ = predict(mlp_model,  X)
    pinn_cm = confusion_matrix(y_np, pinn_pred)
    mlp_cm  = confusion_matrix(y_np, mlp_pred)
    pinn_per = pinn_cm.diagonal() / pinn_cm.sum(axis=1)
    mlp_per  = mlp_cm.diagonal()  / mlp_cm.sum(axis=1)

    plot_ablation_per_class(pinn_per, mlp_per,
                             OUTPUT_DIR / "ablation_per_class_accuracy.png")
    pd.DataFrame({"class": FAULT_CLASSES,
                  "PINN_acc": pinn_per, "MLP_acc": mlp_per,
                  "diff": pinn_per - mlp_per})\
      .to_csv(OUTPUT_DIR / "ablation_per_class_accuracy.csv", index=False)

    print("\n  各类准确率对比（无噪声）:")
    for c, pa, ma in zip(FAULT_CLASSES, pinn_per, mlp_per):
        print(f"    {c:8s}  PINN={pa:.4f}  MLP={ma:.4f}  Δ={pa-ma:+.4f}")

    # (b) 噪声鲁棒性
    print("\n  噪声鲁棒性:")
    pinn_accs, mlp_accs = [], []
    for sigma in noise_levels:
        pp, _ = predict(pinn_model, X, noise_std=sigma)
        mp, _ = predict(mlp_model,  X, noise_std=sigma)
        pa, ma = (pp == y_np).mean(), (mp == y_np).mean()
        pinn_accs.append(pa); mlp_accs.append(ma)
        print(f"    σ={sigma:.2f}  PINN={pa:.4f}  MLP={ma:.4f}  Δ={pa-ma:+.4f}")

    plot_ablation_noise(list(noise_levels), pinn_accs, mlp_accs,
                        OUTPUT_DIR / "ablation_noise_robustness.png")
    pd.DataFrame({"noise_std": list(noise_levels),
                  "PINN_acc": pinn_accs, "MLP_acc": mlp_accs,
                  "diff": [p-m for p,m in zip(pinn_accs, mlp_accs)]})\
      .to_csv(OUTPUT_DIR / "ablation_noise_robustness.csv", index=False)

    return {
        "per_class_accuracy": {
            "PINN": {c: float(a) for c,a in zip(FAULT_CLASSES, pinn_per)},
            "MLP":  {c: float(a) for c,a in zip(FAULT_CLASSES, mlp_per)},
        },
        "noise_robustness": {
            str(s): {"PINN": float(p), "MLP": float(m)}
            for s,p,m in zip(noise_levels, pinn_accs, mlp_accs)
        },
    }


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn_model",   default=PINN_CKPT)
    parser.add_argument("--mlp_model",    default="outputs/checkpoints/mlp_baseline.pth")
    parser.add_argument("--data",         default=DATA_PATH)
    parser.add_argument("--noise_levels", nargs="+", type=float,
                        default=[0.0, 0.01, 0.02, 0.05])
    parser.add_argument("--skip_ablation", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"设备: {DEVICE}  |  输出目录: {OUTPUT_DIR.resolve()}")

    # 加载测试集（与 train_pinn.py 完全相同的切分）
    X_test, y_test = load_test_set(args.data)

    # 加载 PINN 模型
    pinn_model = load_model(PINNMultiClass, args.pinn_model,
                             in_dim=156, n_bus=39, n_classes=5, hidden=128, depth=5)

    # PINN 评估
    pinn_results = eval_classification(pinn_model, X_test, y_test, tag="PINN")

    # 消融实验
    ablation_results = {}
    if not args.skip_ablation:
        if not Path(args.mlp_model).exists():
            print(f"\n[提示] 找不到 MLP 权重: {args.mlp_model}")
            print("  → 跳过消融实验。训练 MLP 后用 --mlp_model 指定路径再跑一次即可")
        else:
            mlp_model = load_model(MLPBaseline, args.mlp_model,
                                    in_dim=156, hidden=128, depth=5, n_classes=5)
            eval_classification(mlp_model, X_test, y_test, tag="MLP")
            ablation_results = eval_ablation(
                pinn_model, mlp_model, X_test, y_test,
                noise_levels=tuple(args.noise_levels)
            )

    # 保存汇总 JSON
    summary = {"data": args.data, "n_test": len(X_test),
               "PINN": pinn_results, "ablation": ablation_results}
    sp = OUTPUT_DIR / "eval_summary.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*55}")
    print(f"全部完成！结果在 {OUTPUT_DIR.resolve()}/")
    print(f"  confusion_matrix_PINN.png/csv     混淆矩阵")
    print(f"  per_class_metrics_PINN.png/csv    逐类 P/R/F1")
    print(f"  classification_report_PINN.txt    完整报告")
    if ablation_results:
        print(f"  ablation_per_class_accuracy.*     消融(a) 各类准确率")
        print(f"  ablation_noise_robustness.*       消融(b) 噪声鲁棒性")
    print(f"  eval_summary.json                 所有指标汇总")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
