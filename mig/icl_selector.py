from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .data import DataPoint
from .label import LabelGraph

Tensor = torch.Tensor


def _as_cpu_float_tensor(x: Tensor) -> Tensor:
    """Convert tensor-like matrix to CPU float tensor."""
    if hasattr(x, "detach"):
        return x.detach().cpu().float()
    return torch.tensor(x, dtype=torch.float32)


def build_propagation_matrix(label_graph: LabelGraph, prop_weight: float = 1.0, eps: float = 1e-12) -> Tensor:
    """Build propagation matrix A from label graph matrix G."""
    if prop_weight < 0.0:
        raise ValueError("prop_weight must be non-negative.")

    graph = _as_cpu_float_tensor(label_graph.wam)
    n_labels = len(label_graph.labels)
    if graph.ndim != 2 or graph.shape != (n_labels, n_labels):
        raise ValueError("Label graph WAM shape does not match label count.")

    eye = torch.eye(n_labels, dtype=graph.dtype)
    mat_diag = graph * eye
    mat_offdiag = graph * (1.0 - eye) * prop_weight
    mat_prop = mat_diag + mat_offdiag

    col_sum = mat_prop.sum(dim=0, keepdim=True).clamp(min=eps)
    return mat_prop / col_sum


def build_label_document_frequency(pool: Sequence[DataPoint], label_graph: LabelGraph) -> Tensor:
    """Count document frequency for each label in label graph order."""
    labels = label_graph.labels
    label_to_index = {label: i for i, label in enumerate(labels)}

    df = torch.zeros(len(labels), dtype=torch.float32)
    for dp in pool:
        seen: set[int] = set()
        for label in dp.labels:
            idx = label_to_index.get(label)
            if idx is None or idx in seen:
                continue
            df[idx] += 1.0
            seen.add(idx)
    return df


def build_query_relevance_vector(
    query_labels: Sequence[str],
    label_graph: LabelGraph,
    *,
    sim_threshold: float = 0.8,
    direct_weight: float = 1.0,
    neighbor_weight: float = 0.5,
) -> Tensor:
    """Build relevance vector by direct query labels + one-hop graph neighbors."""
    if direct_weight < 0.0 or neighbor_weight < 0.0:
        raise ValueError("direct_weight and neighbor_weight must be non-negative.")

    labels = label_graph.labels
    n_labels = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    relevance = torch.zeros(n_labels, dtype=torch.float32)

    query_indices = sorted({label_to_index[label] for label in query_labels if label in label_to_index})
    if not query_indices:
        return relevance

    relevance[query_indices] = direct_weight

    if neighbor_weight > 0.0:
        graph = _as_cpu_float_tensor(label_graph.wam)
        if graph.ndim != 2 or graph.shape[0] != n_labels or graph.shape[1] != n_labels:
            raise ValueError("Label graph WAM shape does not match label count.")

        q_idx = torch.tensor(query_indices, dtype=torch.long)
        neighbor_mask = (graph[:, q_idx] >= sim_threshold).any(dim=1)
        relevance[neighbor_mask] = torch.maximum(
            relevance[neighbor_mask],
            torch.tensor(neighbor_weight, dtype=torch.float32),
        )
        # Keep direct labels at direct_weight even if neighbor_weight is larger.
        relevance[query_indices] = direct_weight

    return relevance


def build_idf_vector(label_df: Tensor, num_docs: int) -> Tensor:
    """Compute IDF with smoothing: log((N + 1) / (df + 1))."""
    if num_docs <= 0:
        raise ValueError("num_docs must be positive.")

    df = _as_cpu_float_tensor(label_df).clamp(min=0.0)
    return torch.log((float(num_docs) + 1.0) / (df + 1.0))


def build_query_label_weight_vector(
    query_labels: Sequence[str],
    label_graph: LabelGraph,
    *,
    label_df: Tensor,
    num_docs: int,
    sim_threshold: float = 0.8,
    direct_weight: float = 1.0,
    neighbor_weight: float = 0.5,
    normalize: bool = True,
) -> Tensor:
    """Build query-aware label weight vector w_q = normalize(relevance * idf)."""
    relevance = build_query_relevance_vector(
        query_labels=query_labels,
        label_graph=label_graph,
        sim_threshold=sim_threshold,
        direct_weight=direct_weight,
        neighbor_weight=neighbor_weight,
    )
    idf = build_idf_vector(label_df=label_df, num_docs=num_docs)
    if idf.numel() != relevance.numel():
        raise ValueError("label_df size must equal label_graph label count.")

    weights = relevance * idf
    if normalize:
        denom = weights.sum()
        if denom > 0:
            weights = weights / denom
    return weights


def build_query_label_weight_matrix(
    query_labels: Sequence[str],
    label_graph: LabelGraph,
    *,
    label_df: Tensor,
    num_docs: int,
    sim_threshold: float = 0.8,
    direct_weight: float = 1.0,
    neighbor_weight: float = 0.5,
    normalize: bool = True,
) -> tuple[Tensor, Tensor]:
    """Build query-aware vector/matrix pair: (w_q, W_q=diag(w_q))."""
    w_q = build_query_label_weight_vector(
        query_labels=query_labels,
        label_graph=label_graph,
        label_df=label_df,
        num_docs=num_docs,
        sim_threshold=sim_threshold,
        direct_weight=direct_weight,
        neighbor_weight=neighbor_weight,
        normalize=normalize,
    )
    W_q = torch.diag(w_q)
    return w_q, W_q


def _normalize_minmax(vec: Tensor, eps: float = 1e-12) -> Tensor:
    vec = _as_cpu_float_tensor(vec)
    vmin = vec.min()
    vmax = vec.max()
    denom = (vmax - vmin).clamp(min=eps)
    if float(vmax - vmin) <= eps:
        return torch.zeros_like(vec)
    return (vec - vmin) / denom


def _estimate_length_from_raw(raw: dict[str, Any]) -> int:
    dialogs = raw.get("dialogs")
    if isinstance(dialogs, list) and dialogs:
        text_parts: list[str] = []
        for turn in dialogs:
            if not isinstance(turn, dict):
                continue
            content = turn.get("content")
            if isinstance(content, str) and content:
                text_parts.append(content)
        text = "\n".join(text_parts)
    else:
        prompt = raw.get("prompt") if isinstance(raw.get("prompt"), str) else ""
        solution = raw.get("canonical_solution") if isinstance(raw.get("canonical_solution"), str) else ""
        text = f"{prompt}\n{solution}".strip()

    if not text:
        return 1
    return max(1, len(text.split()))


def _vectorize_difficulty(pool: Sequence[DataPoint]) -> Tensor:
    """Build one-hot difficulty vectors in order: easy, medium, hard."""
    label_to_idx = {"easy": 0, "medium": 1, "hard": 2}
    vec = torch.zeros((len(pool), 3), dtype=torch.float32)
    for i, dp in enumerate(pool):
        raw = getattr(dp, "raw", None) or {}
        ann = raw.get("annotation") or {}
        diff = ann.get("difficulty") or {}
        label = str(diff.get("label", "")).lower().strip()
        idx = label_to_idx.get(label)
        if idx is not None:
            vec[i, idx] = 1.0
    return vec


@dataclass
class GreedyStep:
    pool_index: int
    score: float
    gain: float
    quality: float
    len_penalty: float
    red_penalty: float


class QueryAwareICLSelector:
    """Query-aware greedy selector implementing TODO5 section 7 flow."""

    def __init__(
        self,
        pool: Sequence[DataPoint],
        label_graph: LabelGraph,
        *,
        prop_weight: float = 1.0,
        phi_type: str = "pow",
        phi_alpha: float = 1.0,
        phi_a: float = 1e-6,
        phi_b: float = 0.8,
        use_difficulty_penalty: bool = False,
        difficulty_penalty_weight: float = 0.0,
    ):
        self.pool = list(pool)
        self.label_graph = label_graph
        self.prop_weight = prop_weight
        self.phi_type = phi_type
        self.phi_alpha = phi_alpha
        self.phi_a = phi_a
        self.phi_b = phi_b
        self.use_difficulty_penalty = use_difficulty_penalty
        self.difficulty_penalty_weight = difficulty_penalty_weight

        self.prop_matrix = build_propagation_matrix(label_graph, prop_weight=prop_weight)
        self.label_df = build_label_document_frequency(self.pool, label_graph)

        # z_i with normalized label contribution, then semantic propagation z_i A
        self.vec_pool = _as_cpu_float_tensor(label_graph.vectorize(self.pool, norm=True))
        self.vec_pool_prop = self.vec_pool @ self.prop_matrix

        scores = torch.tensor([float(dp.score) for dp in self.pool], dtype=torch.float32)
        self.quality_norm = _normalize_minmax(scores)

        lengths = torch.tensor([float(_estimate_length_from_raw(dp.raw)) for dp in self.pool], dtype=torch.float32)
        self.len_penalty = _normalize_minmax(torch.log1p(lengths))

        self.diff_vec = _vectorize_difficulty(self.pool) if use_difficulty_penalty else None

    def _phi(self, x: Tensor) -> Tensor:
        if self.phi_type == "pow":
            return (x + self.phi_a) ** self.phi_b
        if self.phi_type == "exp":
            return 1.0 - torch.exp(-self.phi_alpha * x)
        raise ValueError(f"Invalid phi_type: {self.phi_type}")

    def _marginal_gain(self, candidates: Tensor, x_sel: Tensor) -> Tensor:
        base = self._phi(x_sel).unsqueeze(0)
        cand = self._phi(candidates + x_sel.unsqueeze(0))
        return (cand - base).sum(dim=1)

    def select_k_for_query(
        self,
        query_labels: Sequence[str],
        k: int,
        *,
        tau_q: float = 0.8,
        lambda_quality: float = 0.1,
        lambda_len: float = 0.05,
        lambda_red: float = 0.1,
        lambda_diff: float = 0.0,
    ) -> tuple[list[int], list[GreedyStep]]:
        if k <= 0:
            return [], []

        n = len(self.pool)
        if n == 0:
            return [], []

        w_q = build_query_label_weight_vector(
            query_labels=query_labels,
            label_graph=self.label_graph,
            label_df=self.label_df,
            num_docs=n,
            sim_threshold=tau_q,
            normalize=True,
        )

        # v_i = z_i A W_q ; W_q is diagonal so this is element-wise multiply by w_q
        # Remove zero-weight dimensions to speed up full-pool greedy scoring.
        active_dims = torch.nonzero(w_q > 0, as_tuple=True)[0]
        if active_dims.numel() > 0:
            sem_vec = self.vec_pool_prop[:, active_dims] * w_q[active_dims].unsqueeze(0)
        else:
            sem_vec = torch.zeros((n, 1), dtype=torch.float32)

        if self.use_difficulty_penalty and self.diff_vec is not None:
            all_vec = torch.cat([sem_vec, self.diff_vec * float(self.difficulty_penalty_weight)], dim=1)
        else:
            all_vec = sem_vec

        candidate_indices = torch.arange(n, dtype=torch.long)

        selected: list[int] = []
        steps: list[GreedyStep] = []

        mask = torch.ones(candidate_indices.shape[0], dtype=torch.bool)
        x_sel = torch.zeros(all_vec.shape[1], dtype=torch.float32)

        sem_norm = sem_vec / sem_vec.norm(dim=1, keepdim=True).clamp(min=1e-12)

        rounds = min(k, int(candidate_indices.shape[0]))
        for _ in range(rounds):
            active_indices = candidate_indices[mask]
            cand_vec = all_vec[active_indices]

            gain = self._marginal_gain(cand_vec, x_sel)
            score = gain.clone()

            q_term = self.quality_norm[active_indices]
            l_term = self.len_penalty[active_indices]
            # Product-mode length penalty (design doc "修正后的 DEITA"):
            # quality_eff = quality * exp(-beta * length)
            # where beta is mapped from lambda_len, and length is normalized.
            if float(lambda_len) < 0.0:
                raise ValueError("lambda_len must be non-negative in product-mode length penalty.")
            q_eff = q_term * torch.exp(-float(lambda_len) * l_term)
            score = score + float(lambda_quality) * q_eff

            red_term = torch.zeros_like(score)
            if selected and lambda_red > 0.0:
                selected_tensor = torch.tensor(selected, dtype=torch.long)
                selected_sem = sem_norm[selected_tensor]  # (s, d)
                cand_sem = sem_norm[active_indices]       # (m, d)
                cos_mat = cand_sem @ selected_sem.T
                red_term = cos_mat.max(dim=1).values
                score = score - float(lambda_red) * red_term

            if lambda_diff != 0.0:
                # TODO6 placeholder: keep interface but no difficulty-distribution penalty yet.
                score = score - 0.0 * float(lambda_diff)

            best_pos = int(torch.argmax(score).item())
            chosen_pool_idx = int(active_indices[best_pos].item())

            selected.append(chosen_pool_idx)
            steps.append(
                GreedyStep(
                    pool_index=chosen_pool_idx,
                    score=float(score[best_pos].item()),
                    gain=float(gain[best_pos].item()),
                    quality=float(q_term[best_pos].item()),
                    len_penalty=float(l_term[best_pos].item()),
                    red_penalty=float(red_term[best_pos].item()) if red_term.numel() else 0.0,
                )
            )

            x_sel = x_sel + all_vec[chosen_pool_idx]

            active_positions = torch.nonzero(mask, as_tuple=True)[0]
            chosen_mask_pos = int(active_positions[best_pos].item())
            mask[chosen_mask_pos] = False

        return selected, steps


def select_k_for_query(
    query: dict[str, Any],
    pool: Sequence[DataPoint],
    label_graph: LabelGraph,
    k: int,
    *,
    query_labels_key: str = "query_labels",
    lambda_quality: float = 0.1,
    lambda_len: float = 0.05,
    lambda_red: float = 0.1,
    lambda_diff: float = 0.0,
    prop_weight: float = 1.0,
    tau_q: float = 0.8,
    phi_type: str = "pow",
    phi_alpha: float = 1.0,
    phi_a: float = 1e-6,
    phi_b: float = 0.8,
    use_difficulty_penalty: bool = False,
    difficulty_penalty_weight: float = 0.0,
) -> list[int]:
    """Select K demonstrations for a query. Returns pool indices."""
    labels = query.get(query_labels_key) or []
    if not isinstance(labels, list):
        labels = []

    selector = QueryAwareICLSelector(
        pool=pool,
        label_graph=label_graph,
        prop_weight=prop_weight,
        phi_type=phi_type,
        phi_alpha=phi_alpha,
        phi_a=phi_a,
        phi_b=phi_b,
        use_difficulty_penalty=use_difficulty_penalty,
        difficulty_penalty_weight=difficulty_penalty_weight,
    )
    indices, _ = selector.select_k_for_query(
        query_labels=labels,
        k=k,
        tau_q=tau_q,
        lambda_quality=lambda_quality,
        lambda_len=lambda_len,
        lambda_red=lambda_red,
        lambda_diff=lambda_diff,
    )
    return indices
