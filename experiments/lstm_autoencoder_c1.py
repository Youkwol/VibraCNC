from __future__ import annotations

import argparse
import os
import platform
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset


SENSOR_COLUMNS = [
    "force_x",
    "force_y",
    "force_z",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "ae_rms",
]


def load_condition_series(dataset_root: Path, condition: str) -> pd.DataFrame:
    data_dir = dataset_root / condition
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(
            path,
            header=None,
            names=SENSOR_COLUMNS,
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_conditions_series(dataset_root: Path, conditions: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for condition in conditions:
        frames.append(load_condition_series(dataset_root, condition))
    return pd.concat(frames, ignore_index=True)


def build_sequences(array: np.ndarray, seq_len: int, stride: int = 1) -> np.ndarray:
    """Build sequences from array with optional stride for memory efficiency."""
    n_samples = (len(array) - seq_len) // stride + 1
    if n_samples <= 0:
        raise ValueError("Not enough samples to build sequences.")
    
    # Pre-allocate array for memory efficiency
    sequences = np.zeros((n_samples, seq_len, array.shape[1]), dtype=array.dtype)
    
    for i, idx in enumerate(range(0, len(array) - seq_len + 1, stride)):
        sequences[i] = array[idx : idx + seq_len]
    
    return sequences


class SequenceDataset(Dataset):
    def __init__(self, data: np.ndarray) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_size: int = 64, latent_dim: int = 16) -> None:
        super().__init__()
        second_hidden = max(hidden_size // 2, latent_dim * 2)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.encoder_2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=second_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second_hidden * seq_len, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, second_hidden * seq_len),
            nn.ReLU(),
        )
        self.decoder_1 = nn.LSTM(
            input_size=second_hidden,
            hidden_size=second_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.decoder_2 = nn.LSTM(
            input_size=second_hidden,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        encoded, _ = self.encoder_2(encoded)

        latent = self.latent(encoded)
        latent = latent.view(encoded.size(0), encoded.size(1), -1)

        decoded, _ = self.decoder_1(latent)
        decoded, _ = self.decoder_2(decoded)
        output = self.output_layer(decoded)
        return output


class Seq2SeqAutoencoder(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros(
            batch_size,
            self.seq_len,
            self.input_dim,
            device=x.device,
            dtype=x.dtype,
        )
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        return self.output_layer(decoded)


class EarlyStopping:
    def __init__(self, patience: int = 10) -> None:
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.early_stop = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    checkpoint_path: Path | None = None,
    load_model_only: bool = False,
) -> float:
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(patience=patience)
    start_epoch = 0
    amp_enabled = use_amp and device.type == "cuda"
    scaler = GradScaler(device.type if amp_enabled else "cpu", enabled=amp_enabled)

    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        
        if load_model_only:
            # Only load model weights, reset optimizer and epoch
            print(f"Loaded model weights from checkpoint (optimizer and epoch reset)")
        else:
            # Load everything (for resuming same dataset)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = int(checkpoint.get("epoch", 0))
            stopper.best_loss = checkpoint.get("best_loss", float("inf"))
            if "best_state" in checkpoint:
                stopper.best_state = checkpoint["best_state"]
            print(f"Resuming training from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type if amp_enabled else "cpu", enabled=amp_enabled, dtype=amp_dtype):
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss += loss.detach().item() * len(batch)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                with autocast(device.type if amp_enabled else "cpu", enabled=amp_enabled, dtype=amp_dtype):
                    reconstructed = model(batch)
                    loss = criterion(reconstructed, batch)
                val_loss += loss.detach().item() * len(batch)
        val_loss /= len(val_loader.dataset)

        print(
            f"[train] epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        stopper.step(val_loss, model)

        if checkpoint_path and stopper.best_state is not None:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": stopper.best_state,
                    "optimizer_state": optimizer.state_dict(),
                    "best_loss": stopper.best_loss,
                    "best_state": stopper.best_state,
                },
                checkpoint_path,
            )

        if stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    stopper.restore_best(model)
    return stopper.best_loss


def compute_reconstruction_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    model.eval()
    errors: list[np.ndarray] = []
    criterion = nn.L1Loss(reduction="none")
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with autocast(device.type if amp_enabled else "cpu", enabled=amp_enabled, dtype=amp_dtype):
                recon = model(batch)
            batch_errors = criterion(recon, batch).mean(dim=(1, 2)).cpu().numpy()
            errors.append(batch_errors)
    return np.concatenate(errors)


def plot_anomalies(errors: np.ndarray, threshold: float) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(errors, label="Reconstruction Error")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.4f}")
    anomaly_indices = np.where(errors > threshold)[0]
    plt.scatter(
        anomaly_indices,
        errors[anomaly_indices],
        color="orange",
        label="Anomaly",
    )
    plt.xlabel("Sequence Index")
    plt.ylabel("MAE Error")
    plt.legend()
    plt.title("Reconstruction Error & Anomaly Threshold")
    plt.tight_layout()
    plt.show()


def prepare_loaders(
    sequences: np.ndarray,
    batch_size: int,
    val_ratio: float = 0.2,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[SequenceDataset, DataLoader, DataLoader]:
    dataset = SequenceDataset(sequences)
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        shuffle=True,
        random_state=seed,
    )
    # Windows doesn't support fork, so use num_workers=0
    num_workers = 0 if platform.system() == "Windows" else (os.cpu_count() or 4)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, train_loader, val_loader


def build_model(
    model_type: str,
    input_dim: int,
    seq_len: int,
    hidden_size: int,
    latent_dim: int,
) -> nn.Module:
    if model_type == "seq2seq":
        return Seq2SeqAutoencoder(input_dim, seq_len, hidden_size=hidden_size)
    if model_type == "flatten":
        return LSTMAutoencoder(
            input_dim,
            seq_len,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")


def run_grid_search(
    args: argparse.Namespace,
    normalized: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float | int]:
    batch_sizes = args.search_batch_sizes or [args.batch_size]
    seq_lens = args.search_seq_lens or [args.seq_len]
    hidden_sizes = args.search_hidden_sizes or [args.hidden_size]
    lrs = args.search_lrs or [args.learning_rate]
    search_epochs = args.search_epochs or args.epochs

    best: dict[str, float | int] | None = None
    best_loss = float("inf")
    amp_enabled = args.use_amp and device.type == "cuda"
    sequence_cache: dict[int, np.ndarray] = {}

    for seq_len, batch_size, hidden_size, lr in product(seq_lens, batch_sizes, hidden_sizes, lrs):
        if seq_len not in sequence_cache:
            try:
                sequence_cache[seq_len] = build_sequences(normalized, seq_len)
            except ValueError:
                print(f"[grid] skipped seq_len={seq_len} (insufficient samples)")
                continue
        sequences = sequence_cache[seq_len]
        _, train_loader, val_loader = prepare_loaders(
            sequences,
            batch_size,
            pin_memory=device.type == "cuda",
        )
        model = build_model(
            args.model_type,
            input_dim=len(SENSOR_COLUMNS),
            seq_len=seq_len,
            hidden_size=hidden_size,
            latent_dim=args.latent_dim,
        ).to(device)
        val_loss = train_autoencoder(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=search_epochs,
            patience=args.patience,
            lr=lr,
            use_amp=amp_enabled,
            amp_dtype=amp_dtype,
            checkpoint_path=None,
        )
        print(
            f"[grid] seq_len={seq_len} batch={batch_size} hidden={hidden_size} lr={lr} "
            f"best_val_loss={val_loss:.6f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "lr": lr,
            }

    if best is None:
        raise RuntimeError("Grid search failed to evaluate any configuration.")
    print(f"[grid] best config={best} val_loss={best_loss:.6f}")
    return best


def main(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    use_amp = args.use_amp and device.type == "cuda"

    raw_df = load_conditions_series(dataset_root, args.conditions)
    print(f"[data] Loaded {len(raw_df):,} rows from conditions: {args.conditions}")
    
    # Downsample if requested
    if args.downsample_factor > 1:
        raw_df = raw_df.iloc[::args.downsample_factor].reset_index(drop=True)
        print(f"[data] Downsampled to {len(raw_df):,} rows (factor={args.downsample_factor})")
    
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(raw_df.values)

    best_config = {
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "lr": args.learning_rate,
    }

    if args.grid_search:
        best_config = run_grid_search(args, normalized, device, amp_dtype)

    seq_len = int(best_config["seq_len"])
    batch_size = int(best_config["batch_size"])
    hidden_size = int(best_config["hidden_size"])
    lr = float(best_config["lr"])

    print(f"[sequences] Building sequences with seq_len={seq_len}, stride={args.stride}...")
    sequences = build_sequences(normalized, seq_len=seq_len, stride=args.stride)
    print(f"[sequences] Created {len(sequences):,} sequences (shape: {sequences.shape})")
    dataset, train_loader, val_loader = prepare_loaders(
        sequences,
        batch_size=batch_size,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model_type,
        input_dim=len(SENSOR_COLUMNS),
        seq_len=seq_len,
        hidden_size=hidden_size,
        latent_dim=args.latent_dim,
    ).to(device)
    checkpoint = Path(args.checkpoint_path) if args.checkpoint_path else None
    train_autoencoder(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        lr=lr,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        checkpoint_path=checkpoint,
        load_model_only=args.load_model_only,
    )

    num_workers = 0 if platform.system() == "Windows" else (os.cpu_count() or 4)
    full_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    errors = compute_reconstruction_errors(
        model,
        full_loader,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    threshold = np.percentile(errors, 95)
    print(f"95th percentile threshold: {threshold:.6f}")
    plot_anomalies(errors, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder on PHM conditions")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/phm2010",
        help="Root directory containing condition folders (c1, c4, ...)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=["c1", "c4", "c6"],
        help="Condition folders to load (e.g., c1 c4 c6)",
    )
    parser.add_argument("--model-type", choices=("flatten", "seq2seq"), default="flatten")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1, help="Stride for sequence generation (1=all sequences, 2=every other, etc.)")
    parser.add_argument("--downsample-factor", type=int, default=1, help="Downsample data by this factor (1=no downsampling, 2=every other row, etc.)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--use-amp", action="store_true", help="Enable torch.cuda.amp for mixed precision")
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="AMP compute dtype (effective only when --use-amp and CUDA available)",
    )
    parser.add_argument("--grid-search", action="store_true", help="Enable simple grid search before final training")
    parser.add_argument(
        "--search-batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Candidate batch sizes for grid search",
    )
    parser.add_argument(
        "--search-seq-lens",
        type=int,
        nargs="+",
        default=None,
        help="Candidate sequence lengths for grid search",
    )
    parser.add_argument(
        "--search-hidden-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Candidate hidden sizes for grid search",
    )
    parser.add_argument(
        "--search-lrs",
        type=float,
        nargs="+",
        default=None,
        help="Candidate learning rates for grid search",
    )
    parser.add_argument(
        "--search-epochs",
        type=int,
        default=None,
        help="Epochs per trial during grid search (defaults to --epochs)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="best_anomaly_model.pth",
        help="Path to save/load checkpoint",
    )
    parser.add_argument(
        "--load-model-only",
        action="store_true",
        help="When loading checkpoint, only load model weights (reset optimizer and epoch). Useful for transfer learning across different conditions.",
    )
    main(parser.parse_args())

