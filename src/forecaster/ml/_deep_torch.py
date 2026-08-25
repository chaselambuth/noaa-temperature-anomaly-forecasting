from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden1: int,
        hidden2: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        return self.net(x).squeeze(-1)

# Sequence models (LSTM, GRU, CNN1D, PatchTST, iTransformer, N-HiTS)
def _make_exog_encoder(exog_dim: int, hidden_dim: int, dropout: float) -> Optional[nn.Module]:
    if int(exog_dim) <= 0:
        return None
    return nn.Sequential(
        nn.Linear(int(exog_dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
    )


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        exog_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.exog_encoder = _make_exog_encoder(exog_dim, hidden_size, dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head_dim = hidden_size + (hidden_size if self.exog_encoder is not None else 0)
        self.head = nn.Linear(head_dim, 1)

    def forward(self, x: torch.Tensor, exog: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        last_h = h_n[-1]
        if self.exog_encoder is not None and exog is not None:
            last_h = torch.cat([last_h, self.exog_encoder(exog)], dim=1)
        return self.head(last_h).squeeze(-1)

class GRURegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        exog_dim: int = 0,
    ) -> None:
        super().__init__()
        self.exog_encoder = _make_exog_encoder(exog_dim, hidden_size, dropout)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head_dim = hidden_size + (hidden_size if self.exog_encoder is not None else 0)
        self.head = nn.Linear(head_dim, 1)

    def forward(self, x: torch.Tensor, exog: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, h_n = self.gru(x)
        last_h = h_n[-1]
        if self.exog_encoder is not None and exog is not None:
            last_h = torch.cat([last_h, self.exog_encoder(exog)], dim=1)
        return self.head(last_h).squeeze(-1)

class CNN1DRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        channels: int,
        kernel_size: int,
        exog_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.exog_encoder = _make_exog_encoder(exog_dim, channels, dropout)
        self.conv1 = nn.Conv1d(1, channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        head_dim = channels + (channels if self.exog_encoder is not None else 0)
        self.head = nn.Linear(head_dim, 1)

    def forward(self, x: torch.Tensor, exog: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.squeeze(-1)
        if self.exog_encoder is not None and exog is not None:
            x = torch.cat([x, self.exog_encoder(exog)], dim=1)
        return self.head(x).squeeze(-1)

class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        exog_dim: int = 0,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_len = int(max(1, min(patch_len, seq_len)))
        self.stride = int(max(1, stride))
        self.num_patches = 1 + max(0, (self.seq_len - self.patch_len) // self.stride)
        self.exog_encoder = _make_exog_encoder(exog_dim, d_model, dropout)
        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        head_dim = d_model + (d_model if self.exog_encoder is not None else 0)
        self.head = nn.Linear(head_dim, 1)

    def forward(self, x: torch.Tensor, exog: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, seq_len, 1). Patches turn short monthly histories into tokens.
        x = x.squeeze(-1)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        tokens = self.patch_embed(patches) + self.pos_embed[:, : patches.shape[1], :]
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded).mean(dim=1)
        if self.exog_encoder is not None and exog is not None:
            pooled = torch.cat([pooled, self.exog_encoder(exog)], dim=1)
        return self.head(pooled).squeeze(-1)

class ITransformerRegressor(nn.Module):
    def __init__(
        self,
        num_variates: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_variates = int(num_variates)
        self.value_embed = nn.Linear(1, d_model)
        self.variate_embed = nn.Parameter(torch.zeros(1, self.num_variates, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_variates * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, exog: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, num_features, 1). Features become tokens, following the inverted-transformer idea.
        tokens = self.value_embed(x) + self.variate_embed[:, : x.shape[1], :]
        encoded = self.norm(self.encoder(tokens))
        return self.head(encoded).squeeze(-1)

class NHiTSRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        num_blocks: int,
        pool_sizes: Tuple[int, ...],
        dropout: float,
        exog_dim: int = 0,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.exog_head = (
            nn.Sequential(
                nn.Linear(int(exog_dim), hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
            if int(exog_dim) > 0
            else None
        )
        pool_sizes = tuple(int(max(1, p)) for p in pool_sizes) or (1, 2, 4)
        blocks = []
        for i in range(int(max(1, num_blocks))):
            pool = pool_sizes[i % len(pool_sizes)]
            pooled_len = int(np.ceil(self.seq_len / pool))
            blocks.append(
                nn.Sequential(
                    nn.AvgPool1d(kernel_size=pool, stride=pool, ceil_mode=True),
                    nn.Flatten(),
                    nn.Linear(pooled_len, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, exog: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        preds = [block(x).squeeze(-1) for block in self.blocks]
        out = torch.stack(preds, dim=0).sum(dim=0)
        if self.exog_head is not None and exog is not None:
            out = out + self.exog_head(exog).squeeze(-1)
        return out

def _require_torch(model_name: str, model_cls: Any) -> None:
    if model_cls is not None:
        return

    raise RuntimeError(
        f"PyTorch is required to run {model_name}. "
        "Install a working PyTorch build or choose a non-Torch model such as Ridge, Huber, or XGBoost."
    )

def _seed_torch(cfg: Any) -> int:
    seed = int(getattr(cfg, "random_seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed

def _torch_generator(seed: int) -> Any:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator

def _is_lag_feature(column: str) -> bool:
    return (column.startswith("y_") and "lag" in column) or column.startswith("lag_")

def prepare_sequence_inputs(split: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:

    X_train = split.X_train
    cols = list(X_train.columns)
    lag_cols: List[Tuple[str, int]] = []
    best_by_lag: dict[int, str] = {}
    for c in cols:
        if not _is_lag_feature(c):
            continue
        m = re.search(r"(\d+)$", c)
        if m:
            lag = int(m.group(1))
            if lag not in best_by_lag or c.startswith("lag_"):
                best_by_lag[lag] = c
    if not best_by_lag:
        raise ValueError("No lag columns found for sequence model input.")
    lag_cols = sorted(((c, lag) for lag, c in best_by_lag.items()), key=lambda x: x[1])
    lag_col_names = [c for c, _ in lag_cols]

    Xtr = np.asarray(X_train[lag_col_names].values, dtype=np.float32)
    Xva = np.asarray(split.X_val[lag_col_names].values, dtype=np.float32)
    Xte = np.asarray(split.X_test[lag_col_names].values, dtype=np.float32)

    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True)
    std = np.where(std <= 0.0, 1.0, std)
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std
    Xte = (Xte - mean) / std

    # Reshape to (batch, seq_len, 1)
    Xtr_seq = Xtr[:, :, np.newaxis]
    Xva_seq = Xva[:, :, np.newaxis]
    Xte_seq = Xte[:, :, np.newaxis]
    return Xtr_seq, Xva_seq, Xte_seq, lag_col_names

def prepare_exogenous_inputs(split: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    exog_cols = [c for c in split.X_train.columns if not _is_lag_feature(str(c))]

    if not exog_cols:
        return (
            np.zeros((len(split.X_train), 0), dtype=np.float32),
            np.zeros((len(split.X_val), 0), dtype=np.float32),
            np.zeros((len(split.X_test), 0), dtype=np.float32),
            [],
        )

    Xtr, Xva, Xte = _standardize_splits(
        split.X_train[exog_cols],
        split.X_val[exog_cols],
        split.X_test[exog_cols],
    )
    return Xtr, Xva, Xte, exog_cols

def _train_seq_model(
    cfg: Any,
    split: Any,
    model: Any,
    Xtr_seq: np.ndarray,
    Xva_seq: np.ndarray,
    Xte_seq: np.ndarray,
    Xtr_exog: Optional[np.ndarray] = None,
    Xva_exog: Optional[np.ndarray] = None,
    Xte_exog: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    seed = _seed_torch(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    Xtr_t = torch.from_numpy(Xtr_seq)
    Xva_t = torch.from_numpy(Xva_seq)
    Xte_t = torch.from_numpy(Xte_seq)
    if Xtr_exog is None:
        Xtr_exog = np.zeros((Xtr_seq.shape[0], 0), dtype=np.float32)
    if Xva_exog is None:
        Xva_exog = np.zeros((Xva_seq.shape[0], 0), dtype=np.float32)
    if Xte_exog is None:
        Xte_exog = np.zeros((Xte_seq.shape[0], 0), dtype=np.float32)
    Xtr_exog_t = torch.from_numpy(np.asarray(Xtr_exog, dtype=np.float32))
    Xva_exog_t = torch.from_numpy(np.asarray(Xva_exog, dtype=np.float32))
    Xte_exog_t = torch.from_numpy(np.asarray(Xte_exog, dtype=np.float32))
    ytr_t = torch.from_numpy(np.asarray(split.y_train.values, dtype=np.float32))
    yva_t = torch.from_numpy(np.asarray(split.y_val.values, dtype=np.float32))

    w_train: Optional[np.ndarray] = getattr(split, "w_train", None)
    if w_train is not None:
        w_train = np.asarray(w_train, dtype=np.float32)
        if w_train.shape[0] != Xtr_seq.shape[0]:
            raise ValueError("split.w_train length does not match X_train.")
        wtr_t = torch.from_numpy(w_train)
    else:
        wtr_t = torch.ones_like(ytr_t)

    train_ds = TensorDataset(Xtr_t, Xtr_exog_t, ytr_t, wtr_t)
    val_ds = TensorDataset(Xva_t, Xva_exog_t, yva_t)
    batch_size = int(getattr(cfg, "torch_batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=_torch_generator(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    lr = float(getattr(cfg, "torch_lr", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="none")
    epochs = int(getattr(cfg, "torch_epochs", 200))
    patience = int(getattr(cfg, "torch_patience", 20))

    best_val_loss = float("inf")
    best_state: Optional[dict[str, Any]] = None
    epochs_without_improve = 0

    for _ in range(epochs):
        model.train()
        for xb, xex, yb, wb in train_loader:
            xb, xex, yb, wb = xb.to(device), xex.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            preds = model(xb, xex)
            loss = (criterion(preds, yb) * wb).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_losses = [
                criterion(model(xb.to(device), xex.to(device)), yb.to(device)).mean().item()
                for xb, xex, yb in val_loader
            ]
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_val = model(Xva_t.to(device), Xva_exog_t.to(device)).cpu().numpy().ravel()
        pred_test = model(Xte_t.to(device), Xte_exog_t.to(device)).cpu().numpy().ravel()
    return pred_val.astype(float), pred_test.astype(float)

def fit_predict_lstm(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    _require_torch("LSTM", LSTMRegressor)
    _seed_torch(cfg)
    Xtr_seq, Xva_seq, Xte_seq, _ = prepare_sequence_inputs(split)
    Xtr_exog, Xva_exog, Xte_exog, exog_cols = prepare_exogenous_inputs(split)
    seq_len = Xtr_seq.shape[1]
    hidden_size = int(getattr(cfg, "torch_seq_hidden_size", 32))
    num_layers = int(getattr(cfg, "torch_seq_num_layers", 1))
    dropout = float(getattr(cfg, "torch_seq_dropout", 0.0))
    model = LSTMRegressor(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        exog_dim=len(exog_cols),
    )
    return _train_seq_model(cfg, split, model, Xtr_seq, Xva_seq, Xte_seq, Xtr_exog, Xva_exog, Xte_exog)

def fit_predict_gru(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    _require_torch("GRU", GRURegressor)
    _seed_torch(cfg)
    Xtr_seq, Xva_seq, Xte_seq, _ = prepare_sequence_inputs(split)
    Xtr_exog, Xva_exog, Xte_exog, exog_cols = prepare_exogenous_inputs(split)
    hidden_size = int(getattr(cfg, "torch_seq_hidden_size", 32))
    num_layers = int(getattr(cfg, "torch_seq_num_layers", 1))
    dropout = float(getattr(cfg, "torch_seq_dropout", 0.0))
    model = GRURegressor(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        exog_dim=len(exog_cols),
    )
    return _train_seq_model(cfg, split, model, Xtr_seq, Xva_seq, Xte_seq, Xtr_exog, Xva_exog, Xte_exog)

def fit_predict_cnn1d(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    _require_torch("CNN1D", CNN1DRegressor)
    _seed_torch(cfg)
    Xtr_seq, Xva_seq, Xte_seq, _ = prepare_sequence_inputs(split)
    Xtr_exog, Xva_exog, Xte_exog, exog_cols = prepare_exogenous_inputs(split)
    seq_len = Xtr_seq.shape[1]
    channels = int(getattr(cfg, "torch_cnn_channels", 32))
    kernel_size = int(getattr(cfg, "torch_cnn_kernel_size", 3))
    model = CNN1DRegressor(
        seq_len=seq_len,
        channels=channels,
        kernel_size=kernel_size,
        exog_dim=len(exog_cols),
        dropout=float(getattr(cfg, "torch_dropout", 0.1)),
    )
    return _train_seq_model(cfg, split, model, Xtr_seq, Xva_seq, Xte_seq, Xtr_exog, Xva_exog, Xte_exog)

def fit_predict_patchtst(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    _require_torch("PatchTST", PatchTSTRegressor)
    _seed_torch(cfg)
    Xtr_seq, Xva_seq, Xte_seq, _ = prepare_sequence_inputs(split)
    Xtr_exog, Xva_exog, Xte_exog, exog_cols = prepare_exogenous_inputs(split)
    seq_len = Xtr_seq.shape[1]
    d_model = int(getattr(cfg, "torch_transformer_d_model", 32))
    nhead = int(getattr(cfg, "torch_transformer_nhead", 4))
    if d_model % nhead != 0:
        nhead = 1
    model = PatchTSTRegressor(
        seq_len=seq_len,
        patch_len=int(getattr(cfg, "torch_patch_len", 4)),
        stride=int(getattr(cfg, "torch_patch_stride", 2)),
        d_model=d_model,
        nhead=nhead,
        num_layers=int(getattr(cfg, "torch_transformer_layers", 2)),
        dropout=float(getattr(cfg, "torch_seq_dropout", getattr(cfg, "torch_dropout", 0.1))),
        exog_dim=len(exog_cols),
    )
    return _train_seq_model(cfg, split, model, Xtr_seq, Xva_seq, Xte_seq, Xtr_exog, Xva_exog, Xte_exog)

def prepare_feature_token_inputs(split: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feature_names = list(split.X_train.columns)
    if not feature_names:
        raise ValueError("No feature columns found for iTransformer input.")

    Xtr_std, Xva_std, Xte_std = _standardize_splits(split.X_train, split.X_val, split.X_test)
    return (
        Xtr_std[:, :, np.newaxis],
        Xva_std[:, :, np.newaxis],
        Xte_std[:, :, np.newaxis],
        feature_names,
    )

def fit_predict_itransformer(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    _require_torch("iTransformer", ITransformerRegressor)
    _seed_torch(cfg)
    Xtr_seq, Xva_seq, Xte_seq, _ = prepare_feature_token_inputs(split)
    d_model = int(getattr(cfg, "torch_itransformer_d_model", getattr(cfg, "torch_transformer_d_model", 32)))
    nhead = int(getattr(cfg, "torch_itransformer_nhead", getattr(cfg, "torch_transformer_nhead", 4)))
    if d_model % nhead != 0:
        nhead = 1
    model = ITransformerRegressor(
        num_variates=Xtr_seq.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=int(getattr(cfg, "torch_itransformer_layers", getattr(cfg, "torch_transformer_layers", 2))),
        dropout=float(getattr(cfg, "torch_seq_dropout", getattr(cfg, "torch_dropout", 0.1))),
    )
    return _train_seq_model(cfg, split, model, Xtr_seq, Xva_seq, Xte_seq)

def fit_predict_nhits(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    _require_torch("N-HiTS", NHiTSRegressor)
    _seed_torch(cfg)
    Xtr_seq, Xva_seq, Xte_seq, _ = prepare_sequence_inputs(split)
    Xtr_exog, Xva_exog, Xte_exog, exog_cols = prepare_exogenous_inputs(split)
    seq_len = Xtr_seq.shape[1]
    pool_sizes = tuple(getattr(cfg, "torch_nhits_pool_sizes", (1, 2, 4)))
    model = NHiTSRegressor(
        seq_len=seq_len,
        hidden_size=int(getattr(cfg, "torch_nhits_hidden_size", 64)),
        num_blocks=int(getattr(cfg, "torch_nhits_num_blocks", 3)),
        pool_sizes=pool_sizes,
        dropout=float(getattr(cfg, "torch_dropout", 0.1)),
        exog_dim=len(exog_cols),
    )
    return _train_seq_model(cfg, split, model, Xtr_seq, Xva_seq, Xte_seq, Xtr_exog, Xva_exog, Xte_exog)

def _standardize_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xtr = np.asarray(X_train.values, dtype=np.float32)
    Xva = np.asarray(X_val.values, dtype=np.float32)
    Xte = np.asarray(X_test.values, dtype=np.float32)

    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True)
    std = np.where(std <= 0.0, 1.0, std)

    Xtr_std = (Xtr - mean) / std
    Xva_std = (Xva - mean) / std
    Xte_std = (Xte - mean) / std

    return Xtr_std, Xva_std, Xte_std

def fit_predict_mlp(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:

    _require_torch("MLP", MLPRegressor)

    seed = _seed_torch(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    Xtr_std, Xva_std, Xte_std = _standardize_splits(split.X_train, split.X_val, split.X_test)

    ytr = np.asarray(split.y_train.values, dtype=np.float32)
    yva = np.asarray(split.y_val.values, dtype=np.float32)

    Xtr_t = torch.from_numpy(Xtr_std)  
    Xva_t = torch.from_numpy(Xva_std)
    Xte_t = torch.from_numpy(Xte_std)
    ytr_t = torch.from_numpy(ytr)
    yva_t = torch.from_numpy(yva)

    # Recency Weights
    w_train: Optional[np.ndarray] = getattr(split, "w_train", None)
    if w_train is not None:
        w_train = np.asarray(w_train, dtype=np.float32)
        if w_train.shape[0] != Xtr_std.shape[0]:
            raise ValueError("split.w_train length does not match X_train.")
        wtr_t = torch.from_numpy(w_train)
    else:
        wtr_t = torch.ones_like(ytr_t)

    train_ds = TensorDataset(Xtr_t, ytr_t, wtr_t)
    val_ds = TensorDataset(Xva_t, yva_t)

    batch_size = int(getattr(cfg, "torch_batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=_torch_generator(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = Xtr_std.shape[1]
    hidden1 = int(getattr(cfg, "torch_hidden1", 64))
    hidden2 = int(getattr(cfg, "torch_hidden2", 32))
    dropout = float(getattr(cfg, "torch_dropout", 0.1))

    model = MLPRegressor(input_dim, hidden1, hidden2, dropout).to(device)  

    lr = float(getattr(cfg, "torch_lr", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    criterion = nn.MSELoss(reduction="none") 

    epochs = int(getattr(cfg, "torch_epochs", 200))
    patience = int(getattr(cfg, "torch_patience", 20))

    best_val_loss = float("inf")
    best_state: Optional[dict[str, Any]] = None
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss_vec = criterion(preds, yb)
            loss = (loss_vec * wb).mean()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss_vec = criterion(preds, yb)
                val_losses.append(loss_vec.mean().item())
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    # Restore best weights if available
    if best_state is not None:
        model.load_state_dict(best_state)  

    # Final predictions
    model.eval()
    with torch.no_grad():
        pred_val = model(Xva_t.to(device)).cpu().numpy()
        pred_test = model(Xte_t.to(device)).cpu().numpy()

    return pred_val.astype(float).ravel(), pred_test.astype(float).ravel()
