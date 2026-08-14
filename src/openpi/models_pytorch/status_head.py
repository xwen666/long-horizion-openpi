"""Subtask completion head operating on the frozen VLM prefix states."""

import torch
from torch import nn


class StatusEncoder(nn.Module):
    """Aggregates VLM prefix tokens into one subtask-status representation."""

    def __init__(
        self,
        vla_dim: int,
        status_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(vla_dim, status_dim)
        self.status_token = nn.Parameter(torch.randn(1, 1, status_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=status_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(status_dim)

    def forward(self, prefix_out: torch.Tensor, prefix_valid_mask: torch.Tensor) -> torch.Tensor:
        """Return ``z_status`` with shape ``[B, status_dim]``."""
        if prefix_out.ndim != 3:
            raise ValueError(f"prefix_out must have shape [B, M, D], got {tuple(prefix_out.shape)}")
        if prefix_valid_mask.shape != prefix_out.shape[:2]:
            raise ValueError(
                "prefix_valid_mask must have shape [B, M], "
                f"got {tuple(prefix_valid_mask.shape)} for prefix {tuple(prefix_out.shape)}"
            )

        x = self.input_proj(prefix_out.float())
        batch_size = x.shape[0]
        status_token = self.status_token.expand(batch_size, -1, -1)
        x = torch.cat([x, status_token], dim=1)

        # OpenPI uses True for valid tokens; TransformerEncoder uses True for padding.
        prefix_padding_mask = ~prefix_valid_mask.bool()
        status_padding_mask = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=prefix_padding_mask.device
        )
        padding_mask = torch.cat([prefix_padding_mask, status_padding_mask], dim=1)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.output_norm(x[:, -1, :])


class DoneHead(nn.Module):
    """Maps the status representation to an unnormalized completion logit."""

    def __init__(self, status_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(status_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_status: torch.Tensor) -> torch.Tensor:
        return self.net(z_status)


class DoneScheduler:
    """Requires consecutive high-probability predictions before replanning."""

    def __init__(self, threshold: float = 0.8, consecutive_steps: int = 3) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if consecutive_steps < 1:
            raise ValueError("consecutive_steps must be positive")
        self.threshold = threshold
        self.consecutive_steps = consecutive_steps
        self._count = 0

    def reset(self) -> None:
        self._count = 0

    def update(self, done_prob: float | torch.Tensor) -> bool:
        probability = float(done_prob.detach().mean().cpu()) if isinstance(done_prob, torch.Tensor) else done_prob
        self._count = self._count + 1 if probability >= self.threshold else 0
        return self._count >= self.consecutive_steps
