import torch

from enum import Enum
from abc import ABCMeta, abstractmethod
from typing import Iterator, Optional

from .data import DataPoint, Dataset
from .label import LabelGraph

Tensor = torch.Tensor


class SamplerType(Enum):
    MIG = "mig"
    RANDOM = "random"
    IFD = "ifd"
    

class Sampler(metaclass=ABCMeta):
    """Abstract base class for samplers."""
    
    @abstractmethod
    def sample(self, pool: Dataset, num_sample: int, batch_size: int = 0) -> Iterator[DataPoint]:
        """Sample a subset of the pool with 'num_sample' samples."""


class RandomSampler(Sampler):
    
    def sample(self, pool: Dataset, num_sample: int, batch_size: int = 0) -> Iterator[DataPoint]:
        if num_sample >= len(pool):
            raise ValueError("num_sample must be less than the size of the pool.")
        import random
        rng = random.Random(42)
        indices = rng.sample(range(len(pool)), num_sample)
        for i in indices:
            yield pool[i]


class IFDSampler(Sampler):
    """
    Sample data by IFD
    """
    
    def __init__(
        self,
        upper_bound: float = 1.0,
        lower_bound: float = 0.0,
    ):
        super().__init__()
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
    
    def filter(self, pool: Dataset):
        """Filter outilers"""
        
        filtered_pool = []
        for dp in pool:
            if dp.score <= self.upper_bound and dp.score >= self.lower_bound:
                filtered_pool.append(dp)
        
        return filtered_pool
        
    def sample(self, pool: Dataset, num_sample: int, batch_size: int = 0) -> Iterator[DataPoint]:
        
        # filter
        filtered_pool = self.filter(pool)
        
        # sort
        sorted_pool = sorted(filtered_pool, key=lambda x: x.score, reverse=True)
        
        # sample
        for i in range(num_sample):
            yield sorted_pool[i]
    
    
class MIGSampler(Sampler):
    """Maximum Information Gain Sampler.

    Args:
        label_graph
    """

    def __init__(
        self,
        label_graph: LabelGraph,
        phi_type: str = "pow",
        phi_alpha: float = 1.0,
        phi_a: float = 1e-6,
        phi_b: float = 0.8,
        prop_weight: float = 1.0,
        norm: bool = True,
        use_difficulty_penalty: bool = False,
        difficulty_penalty_weight: float = 0.0,
    ):
        super().__init__()
        self.label_graph = label_graph
        self.phi_type = phi_type
        self.phi_alpha = phi_alpha
        self.phi_a = phi_a
        self.phi_b = phi_b
        self.prop_weight = prop_weight
        self.norm = norm
        self.use_difficulty_penalty = use_difficulty_penalty
        self.difficulty_penalty_weight = difficulty_penalty_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def phi(self, x: Tensor) -> Tensor:
        """Element-wise concave utility phi(x)."""
        if self.phi_type == "exp":
            return self._phi_exp(x)
        elif self.phi_type == "pow":
            return self._phi_pow(x)
        else:
            raise ValueError(f"Invalid phi_type: {self.phi_type}")

    def _phi_exp(self, x: Tensor) -> Tensor:
        # Concave increasing utility in [0, 1): 1 - exp(-alpha * x)
        return 1.0 - torch.exp(-self.phi_alpha * x)

    def _phi_pow(self, x: Tensor) -> Tensor:
        # Concave increasing utility when 0 < phi_b < 1
        return (x + self.phi_a) ** self.phi_b

    def _calc_marginal_gain(self, vec_candidates: Tensor, vec_x_sel: Tensor, vec_w: Tensor) -> Tensor:
        """Compute exact marginal gain: sum_j w_j * (phi(x_j + dx_ij) - phi(x_j))."""
        base = self.phi(vec_x_sel).unsqueeze(0)  # (1, d)
        cand = self.phi(vec_candidates + vec_x_sel.unsqueeze(0))  # (m, d)
        delta = (cand - base) * vec_w.unsqueeze(0)
        return delta.sum(dim=1)

    def _vectorize_difficulty(self, pool: Dataset) -> Tensor:
        """Build one-hot difficulty vectors in order: easy, medium, hard."""
        label_to_idx = {
            "easy": 0,
            "medium": 1,
            "hard": 2,
        }
        vec = torch.zeros((len(pool), 3), dtype=torch.float32, device=self.device)
        for i, dp in enumerate(pool):
            raw = getattr(dp, "raw", None) or {}
            ann = raw.get("annotation") or {}
            diff = ann.get("difficulty") or {}
            label = str(diff.get("label", "")).lower().strip()
            idx = label_to_idx.get(label)
            if idx is not None:
                vec[i, idx] = 1.0
        return vec

    def calc_prop_matrix(self):
        # get the weighted adjacency matrix
        mat_w = self.label_graph.wam  # (n_labels, n_labels)
        if mat_w is None:
            raise ValueError("Weighted adjacency matrix is not set.")

        # get the propagation matrix
        n = mat_w.size(0)
        mask = torch.eye(n, device=mat_w.device)
        mat_p_diag = mat_w * mask
        mat_p_ndiag = mat_w * (1 - mask) * self.prop_weight
        mat_p = mat_p_diag + mat_p_ndiag

        # normalization
        mat_p = mat_p / mat_p.sum(axis=0)

        mat_p = mat_p.to(self.device).float()

        return mat_p

    def sample(self, pool: Dataset, num_sample: int, batch_size: int = 0) -> Iterator[DataPoint]:
        """Sample a subset of the pool with 'num_sample' samples."""
        if num_sample > len(pool):
            raise ValueError("num_sample must be less than or equal to the size of the pool.")

        # make a copy of the pool
        pool = pool.copy()

        # get the propagation matrix
        mat_p = self.calc_prop_matrix()

        # 3.A semantic vector: e_sem -> A * e_sem
        vec_pool = self.label_graph.vectorize(pool, self.norm).to(self.device).float()  # (n_samples, n_labels)
        vec_pool_prop = vec_pool @ mat_p

        # Optional: difficulty vector and concatenation [A*e_sem ; lambda_d*e_diff]
        if self.use_difficulty_penalty:
            vec_pool_diff = self._vectorize_difficulty(pool) * float(self.difficulty_penalty_weight)
            vec_pool_all = torch.cat([vec_pool_prop, vec_pool_diff], dim=1)
        else:
            vec_pool_all = vec_pool_prop

        # initialize the selected samples and the information gain (score) distribution
        vec_w = torch.ones(vec_pool_all.shape[1], device=self.device, dtype=torch.float32)
        n_sel = 0
        mask = torch.ones(len(pool), dtype=torch.bool, device=self.device)
        vec_x_sel = torch.zeros(vec_pool_all.shape[1], device=self.device, dtype=torch.float32)

        while n_sel < num_sample:
            # calculate exact marginal gain of each data point (no gradient approximation)
            if not batch_size:
                vec_candidate = self._calc_marginal_gain(vec_pool_all[mask], vec_x_sel, vec_w)  # (n_candidates,)
            else:
                vec_candidate_list = []
                total_size = vec_pool_all.shape[0]
                for i in range(0, total_size, batch_size):
                    # get indices for current batch
                    end_index = min(i + batch_size, total_size)
                    # process batch data
                    batch_vec_pool_all = vec_pool_all[i:end_index]
                    batch_mask = mask[i:end_index]
                    batch_masked_vec_pool_all = batch_vec_pool_all[batch_mask]
                    # check len
                    if batch_masked_vec_pool_all.shape[0] == 0:
                        continue
                    # calculate marginal gain of each data point
                    vec_candidate_batch = self._calc_marginal_gain(batch_masked_vec_pool_all, vec_x_sel, vec_w)
                    vec_candidate_list.append(vec_candidate_batch)
                # merge results
                vec_candidate = torch.cat(vec_candidate_list, dim=0)

            # select the data point with the maximum information gain
            idx = torch.argmax(vec_candidate)
            # Keep indices 1D even when only one candidate remains.
            indices = torch.nonzero(mask, as_tuple=True)[0]
            selected_idx = indices[idx]

            vec_x_sel += vec_pool_all[selected_idx]
            mask[selected_idx] = False
            n_sel += 1
            # Keep pool indexing aligned with vec_pool_all/mask by not mutating pool.
            dp = pool[selected_idx.item()]

            yield dp


def create_sampler(
    sampler_type: SamplerType,
    label_graph: Optional[LabelGraph],
    phi_type: str,
    phi_alpha: float,
    phi_a: float,
    phi_b: float,
    prop_weight: float,
    norm: bool,
    use_difficulty_penalty: bool = False,
    difficulty_penalty_weight: float = 0.0,
) -> "Sampler":
    if sampler_type == SamplerType.MIG:
        if label_graph is None:
            raise ValueError("label_graph must be provided for MIG sampler.")
        return MIGSampler(
            label_graph,
            phi_type=phi_type,
            phi_alpha=phi_alpha,
            phi_a=phi_a,
            phi_b=phi_b,
            prop_weight=prop_weight,
            norm=norm,
            use_difficulty_penalty=use_difficulty_penalty,
            difficulty_penalty_weight=difficulty_penalty_weight,
        )
    elif sampler_type == SamplerType.RANDOM:
        return RandomSampler()
