"""
Probing MLP 학습 스크립트.

collect_probing_data.py로 수집한 데이터를 사용하여,
사전 학습된 World Model 인코더의 latent state z로부터
잠수부 개수(0~6)를 classification하는 MLP를 학습합니다.

Usage:
    conda activate storm
    python train_probing_mlp.py \
        --data_path probing_data/Seaquest-life_done-wm_2L512D8H-100k-seed1_X_probing.npz \
        --run_name Seaquest-life_done-wm_2L512D8H-100k-seed1_X \
        --task diver_count
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from tqdm import tqdm
from collections import Counter
import colorama
import json

from utils import seed_np_torch, load_config
from sub_models.world_models import WorldModel


# ──────────────────────────────────────────────────────────────
# 1. Probing Tasks (RAM에서 Ground Truth 라벨 추출)
# ──────────────────────────────────────────────────────────────

PROBING_TASKS = {
    "diver_count": {
        "description": "하단 잠수부 개수 분류 (0~6)",
        "type": "classification",
        "num_classes": 7,
        "extract_label": lambda ram: ram[:, 62].astype(np.int64),  # RAM[62] = diver count
    },
    # 확장 가능: 잠수부 위치 regression, 잠수부 존재 detection 등
}


# ──────────────────────────────────────────────────────────────
# 2. Probing MLP 모델
# ──────────────────────────────────────────────────────────────

class ProbingMLP(nn.Module):
    """경량 MLP probe. 인코더의 표현력을 테스트하므로 일부러 간단하게 구성."""
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, z):
        return self.mlp(z)


# ──────────────────────────────────────────────────────────────
# 3. Dataset
# ──────────────────────────────────────────────────────────────

class ProbingDataset(Dataset):
    """사전에 추출된 (latent_z, label) 쌍을 저장하는 Dataset."""
    def __init__(self, latents, labels):
        self.latents = latents  # (N, latent_dim) float32
        self.labels = labels    # (N,) int64

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]


# ──────────────────────────────────────────────────────────────
# 4. Latent 추출 (Frozen Encoder)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents(world_model, obs_array, batch_size=128, sample_mode="mode"):
    """
    Frozen encoder로 obs → latent z 추출.
    
    Args:
        world_model: 사전 학습된 WorldModel
        obs_array: (N, H, W, C) uint8 numpy array
        batch_size: 한 번에 처리할 프레임 수
        sample_mode: "mode" (결정적), "probs" (확률 분포), "random_sample" (샘플링)
    
    Returns:
        latents: (N, 1024) float32 tensor
    """
    world_model.eval()
    all_latents = []

    N = len(obs_array)
    for start in tqdm(range(0, N, batch_size), desc="Extracting latents"):
        end = min(start + batch_size, N)
        batch_obs = obs_array[start:end]

        # (B, H, W, C) → (B, 1, C, H, W), normalize to [0, 1]
        obs_tensor = torch.from_numpy(batch_obs).float().cuda() / 255.0
        obs_tensor = rearrange(obs_tensor, "B H W C -> B 1 C H W")

        # Encoder forward (frozen, no gradient)
        z = world_model.encode_obs(obs_tensor, sample_mode=sample_mode)
        # z shape: (B, 1, 1024)
        z = z.squeeze(1).float().cpu()  # (B, 1024)
        all_latents.append(z)

    return torch.cat(all_latents, dim=0)


# ──────────────────────────────────────────────────────────────
# 5. 학습 및 평가
# ──────────────────────────────────────────────────────────────

def train_probe(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device="cuda"):
    """Probing MLP 학습."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for z, labels in train_loader:
            z, labels = z.to(device), labels.to(device)
            logits = model(z)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += len(labels)

        train_acc = 100.0 * train_correct / train_total

        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for z, labels in val_loader:
                z, labels = z.to(device), labels.to(device)
                logits = model(z)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_total += len(labels)

        val_acc = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss/train_total:.4f}, Train Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_acc:.1f}% "
                  f"{'★ best' if val_acc >= best_val_acc else ''}")

    return best_state, best_val_acc


def evaluate_per_class(model, dataset, num_classes, device="cuda"):
    """클래스별 정확도 보고."""
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for z, labels in loader:
            z, labels = z.to(device), labels.to(device)
            preds = model(z).argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            for c in range(num_classes):
                mask = labels == c
                class_correct[c] += (preds[mask] == c).sum().item()
                class_total[c] += mask.sum().item()

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    overall_acc = 100.0 * (all_preds == all_labels).mean()

    return class_correct, class_total, overall_acc


# ──────────────────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────────────────

def main():
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser(description="Probing MLP 학습")
    parser.add_argument("--data_path", type=str, required=True,
                        help="collect_probing_data.py로 생성된 .npz 파일 경로")
    parser.add_argument("--demo_path", type=str, default="",
                        help="선택적: 함께 학습할 demonstration .npz 파일 경로 (Zero-shot 일반화 테스트용)")
    parser.add_argument("--config_path", type=str, default="config_files/STORM.yaml",
                        help="STORM config 파일 경로")
    parser.add_argument("--env_name", type=str, default="ALE/Seaquest-v5",
                        help="환경 이름 (action_dim 결정용)")
    parser.add_argument("--run_name", type=str, required=True,
                        help="World Model 체크포인트 디렉토리 이름")
    parser.add_argument("--step", type=int, default=100000,
                        help="로드할 체크포인트 스텝")
    parser.add_argument("--task", type=str, default="diver_count",
                        choices=list(PROBING_TASKS.keys()),
                        help="프로빙 태스크")
    parser.add_argument("--sample_mode", type=str, default="mode",
                        choices=["mode", "probs", "random_sample"],
                        help="Latent 추출 시 샘플링 모드")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation 비율")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="MLP hidden dim")
    parser.add_argument("--epochs", type=int, default=50,
                        help="학습 에폭 수")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="학습 batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드")
    parser.add_argument("--output_dir", type=str, default="probing_results",
                        help="결과 저장 디렉토리")
    args = parser.parse_args()

    seed_np_torch(seed=args.seed)
    conf = load_config(args.config_path)
    task_info = PROBING_TASKS[args.task]

    print(f"\n{'='*60}")
    print(f"  Probing Task: {task_info['description']}")
    print(f"  Data: {args.data_path}")
    print(f"  World Model: ckpt/{args.run_name}/world_model_{args.step}.pth")
    print(f"  Sample mode: {args.sample_mode}")
    print(f"{'='*60}\n")

    # ── 1. 데이터 로드 ──
    print("[1/4] 데이터 로드 중...")
    data = np.load(args.data_path)
    obs_array = data["obs"]       # (N, H, W, C)
    ram_array = data["ram"]       # (N, 128)
    
    if args.demo_path and os.path.exists(args.demo_path):
        print(f"  Demonstration 데이터 병합 중: {args.demo_path}")
        demo_data = np.load(args.demo_path)
        demo_obs = demo_data["obs"]
        demo_ram = demo_data["ram"]
        
        # 합치기
        obs_array = np.concatenate([obs_array, demo_obs], axis=0)
        ram_array = np.concatenate([ram_array, demo_ram], axis=0)
        print(f"  병합 완료: 총 {len(obs_array)} 프레임")

    labels = task_info["extract_label"](ram_array)  # (N,)

    print(f"  총 프레임: {len(obs_array)}")
    print(f"  라벨 분포:")
    label_counts = Counter(labels.tolist())
    for c in sorted(label_counts.keys()):
        n = label_counts[c]
        print(f"    class {c}: {n} ({100*n/len(labels):.1f}%)")

    # ── 2. World Model 로드 & Latent 추출 ──
    print(f"\n[2/4] World Model 인코더로 latent 추출 중 (sample_mode={args.sample_mode})...")

    # action_dim 결정을 위해 임시 환경 생성
    import gymnasium
    import ale_py
    gymnasium.register_envs(ale_py)
    dummy_env = gymnasium.make(args.env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    world_model = WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads,
    ).cuda()

    wm_path = f"ckpt/{args.run_name}/world_model_{args.step}.pth"
    world_model.load_state_dict(torch.load(wm_path, map_location="cuda"))
    print(f"  World Model 로드 완료: {wm_path}")

    # Encoder freeze (gradient 차단)
    for param in world_model.parameters():
        param.requires_grad = False

    latents = extract_latents(world_model, obs_array, batch_size=128, sample_mode=args.sample_mode)
    print(f"  Latent shape: {latents.shape}")  # (N, 1024)

    # ── 3. Train/Val 분할 & MLP 학습 ──
    print(f"\n[3/4] Probing MLP 학습 중...")
    labels_tensor = torch.from_numpy(labels).long()

    N = len(latents)
    indices = np.random.permutation(N)
    val_size = int(N * args.val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = ProbingDataset(latents[train_indices], labels_tensor[train_indices])
    val_dataset = ProbingDataset(latents[val_indices], labels_tensor[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    probe = ProbingMLP(
        input_dim=latents.shape[1],
        num_classes=task_info["num_classes"],
        hidden_dim=args.hidden_dim,
    ).cuda()

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  MLP params: {sum(p.numel() for p in probe.parameters()):,}\n")

    best_state, best_val_acc = train_probe(
        probe, train_loader, val_loader,
        num_epochs=args.epochs, lr=args.lr, device="cuda"
    )

    # ── 4. 최종 평가 & 보고 ──
    print(f"\n[4/4] 최종 결과 (best val acc 기준 모델)")
    probe.load_state_dict(best_state)

    # 전체 데이터셋에 대한 클래스별 정확도
    full_dataset = ProbingDataset(latents, labels_tensor)
    class_correct, class_total, overall_acc = evaluate_per_class(
        probe, full_dataset, task_info["num_classes"]
    )

    print(f"\n{'='*60}")
    print(f"  Overall Accuracy: {colorama.Fore.GREEN}{overall_acc:.1f}%{colorama.Style.RESET_ALL}")
    print(f"  Best Val Accuracy: {colorama.Fore.GREEN}{best_val_acc:.1f}%{colorama.Style.RESET_ALL}")
    print(f"\n  Per-class accuracy:")
    for c in range(task_info["num_classes"]):
        if class_total[c] > 0:
            acc = 100.0 * class_correct[c] / class_total[c]
            bar = "█" * int(acc / 2)
            print(f"    class {c} (n={int(class_total[c]):5d}): {acc:5.1f}%  {bar}")
        else:
            print(f"    class {c} (n=    0):   N/A  (데이터 없음)")
    print(f"{'='*60}")

    # 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    result_name = f"{args.run_name}_{args.task}_{args.sample_mode}"

    # MLP 가중치 저장
    torch.save(best_state, os.path.join(args.output_dir, f"{result_name}_probe.pth"))

    # 결과 JSON 저장
    results = {
        "task": args.task,
        "description": task_info["description"],
        "run_name": args.run_name,
        "step": args.step,
        "sample_mode": args.sample_mode,
        "overall_accuracy": overall_acc,
        "best_val_accuracy": best_val_acc,
        "total_frames": int(N),
        "per_class": {
            str(c): {
                "total": int(class_total[c]),
                "correct": int(class_correct[c]),
                "accuracy": float(100 * class_correct[c] / class_total[c]) if class_total[c] > 0 else None,
            }
            for c in range(task_info["num_classes"])
        },
        "hyperparams": {
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "val_ratio": args.val_ratio,
        },
    }
    result_path = os.path.join(args.output_dir, f"{result_name}_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {result_path}")
    print(f"모델 저장: {os.path.join(args.output_dir, f'{result_name}_probe.pth')}")


if __name__ == "__main__":
    main()
