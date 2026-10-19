# Copyright 2026 OPRD
from collections.abc import Iterator, Sized

import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler

CODE_DOMAIN_SOURCES = frozenset({"codecontests", "apps", "codeforces", "taco"})


def _is_code_source(source) -> bool:
    return str(source) in CODE_DOMAIN_SOURCES


def math_sample_probability(t_math: float, t_code: float, *, eps: float, p_min: float, p_max: float) -> float:
    if t_math <= 0.0 and t_code <= 0.0:
        return 0.5
    p = t_code / (t_math + t_code + eps)
    return float(min(p_max, max(p_min, p)))


class TokenCountBalancedPoolSampler(AbstractCurriculumSampler):
    """Draw the next prompt from math vs code pools using trained token counts.

    After each consumed training batch, ``T_math`` / ``T_code`` accumulate
    padding-free prompt+response tokens. The next draw uses
    ``p_math = clip(T_code / (T_math + T_code + eps), p_min, p_max)``.
    Loss aggregation is unchanged (Control token-mean).
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        super().__init__(data_source=data_source, data_config=data_config)
        sampler_cfg = data_config.get("sampler") or {}
        self.p_min = float(sampler_cfg.get("p_min", 0.05))
        self.p_max = float(sampler_cfg.get("p_max", 0.95))
        self.eps = float(sampler_cfg.get("eps", 1.0))
        seed = data_config.get("seed")
        self._seed = 0 if seed is None else int(seed)

        self._math_idx: list[int] = []
        self._code_idx: list[int] = []
        n = len(data_source)
        dataframe = getattr(data_source, "dataframe", None)
        if dataframe is None:
            raise ValueError("TokenCountBalancedPoolSampler requires RLHFDataset.dataframe")
        sources = dataframe["data_source"]
        for i in range(n):
            if _is_code_source(sources[i]):
                self._code_idx.append(i)
            else:
                self._math_idx.append(i)
        if not self._math_idx or not self._code_idx:
            raise ValueError(
                f"TokenCountBalancedPoolSampler needs both pools; math={len(self._math_idx)} code={len(self._code_idx)}"
            )

        self.t_math = 0.0
        self.t_code = 0.0
        self._choice_gen = torch.Generator()
        self._choice_gen.manual_seed(self._seed)
        self._math_gen = torch.Generator()
        self._math_gen.manual_seed(self._seed + 1)
        self._code_gen = torch.Generator()
        self._code_gen.manual_seed(self._seed + 2)
        self._math_perm: list[int] = []
        self._code_perm: list[int] = []
        self._math_ptr = 0
        self._code_ptr = 0
        self._reshuffle_pool("math")
        self._reshuffle_pool("code")
        self._num_samples = n
        self._last_metrics: dict[str, float] = {}

    def __len__(self) -> int:
        return self._num_samples

    def _p_math(self) -> float:
        return math_sample_probability(
            self.t_math, self.t_code, eps=self.eps, p_min=self.p_min, p_max=self.p_max
        )

    def _reshuffle_pool(self, which: str) -> None:
        pool = self._math_idx if which == "math" else self._code_idx
        gen = self._math_gen if which == "math" else self._code_gen
        perm = torch.randperm(len(pool), generator=gen).tolist()
        if which == "math":
            self._math_perm = perm
            self._math_ptr = 0
        else:
            self._code_perm = perm
            self._code_ptr = 0

    def _next_from_pool(self, which: str) -> int:
        if which == "math":
            if self._math_ptr >= len(self._math_perm):
                self._reshuffle_pool("math")
            idx = self._math_idx[self._math_perm[self._math_ptr]]
            self._math_ptr += 1
            return idx
        if self._code_ptr >= len(self._code_perm):
            self._reshuffle_pool("code")
        idx = self._code_idx[self._code_perm[self._code_ptr]]
        self._code_ptr += 1
        return idx

    def __iter__(self) -> Iterator[int]:
        for _ in range(self._num_samples):
            p_math = self._p_math()
            if torch.rand((), generator=self._choice_gen).item() < p_math:
                yield self._next_from_pool("math")
            else:
                yield self._next_from_pool("code")

    def update(self, batch: DataProto) -> None:
        responses = batch.batch["responses"]
        max_response = responses.shape[-1]
        attention_mask = batch.batch["attention_mask"]
        prompt_tokens = attention_mask[:, :-max_response].sum(dim=-1)
        response_tokens = attention_mask[:, -max_response:].sum(dim=-1)
        consumed = (prompt_tokens + response_tokens).to(dtype=torch.float64)
        sources = batch.non_tensor_batch.get("data_source")
        if sources is None:
            raise ValueError("token-count pool sampler requires non_tensor_batch data_source")
        step_math = 0.0
        step_code = 0.0
        for i, source in enumerate(sources):
            n_tok = float(consumed[i].item())
            if _is_code_source(source):
                self.t_code += n_tok
                step_code += n_tok
            else:
                self.t_math += n_tok
                step_math += n_tok
        p_math = self._p_math()
        self._last_metrics = {
            "sampler/t_math": self.t_math,
            "sampler/t_code": self.t_code,
            "sampler/p_math": p_math,
            "sampler/p_code": 1.0 - p_math,
            "sampler/step_tokens_math": step_math,
            "sampler/step_tokens_code": step_code,
        }

    def metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    def state_dict(self) -> dict:
        return {
            "t_math": self.t_math,
            "t_code": self.t_code,
            "choice_gen": self._choice_gen.get_state(),
            "math_gen": self._math_gen.get_state(),
            "code_gen": self._code_gen.get_state(),
            "math_perm": list(self._math_perm),
            "code_perm": list(self._code_perm),
            "math_ptr": self._math_ptr,
            "code_ptr": self._code_ptr,
        }

    def load_state_dict(self, state: dict) -> None:
        self.t_math = float(state["t_math"])
        self.t_code = float(state["t_code"])
        self._choice_gen.set_state(state["choice_gen"])
        self._math_gen.set_state(state["math_gen"])
        self._code_gen.set_state(state["code_gen"])
        self._math_perm = list(state["math_perm"])
        self._code_perm = list(state["code_perm"])
        self._math_ptr = int(state["math_ptr"])
        self._code_ptr = int(state["code_ptr"])
