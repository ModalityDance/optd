"""Mixed math/code batches, cycling each complete source pool without replacement."""

from collections import defaultdict
from collections.abc import Iterator, Sized

import torch
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler


class BalancedCycleSampler(AbstractSampler):
    def __init__(self, data_source: Sized, data_config: DictConfig):
        super().__init__(data_source=data_source, data_config=data_config)
        self.batch_size = int(data_config.train_batch_size)
        if self.batch_size % 2:
            raise ValueError("balanced_cycle requires an even train_batch_size")
        frame = getattr(data_source, "dataframe", None)
        columns = getattr(frame, "column_names", frame)
        if frame is None or "data_source" not in columns:
            raise ValueError("balanced_cycle requires indexed data_source rows")
        pools = defaultdict(list)
        for index, source in enumerate(frame["data_source"]):
            pools[str(source)].append(index)
        if set(pools) != {"math_dapo", "codecontests"}:
            raise ValueError(f"balanced_cycle expects math_dapo and codecontests: {sorted(pools)}")
        self.pools = dict(pools)
        self.seed = int(data_config.get("seed") or 0)
        self.num_batches = int(data_config.get("sampler", {}).get("balanced_cycle_num_batches", 0))
        if self.num_batches <= 0:
            raise ValueError("balanced_cycle_num_batches must be set")

    def __len__(self) -> int:
        return self.num_batches * self.batch_size

    def __iter__(self) -> Iterator[int]:
        generators = {source: torch.Generator().manual_seed(self.seed + offset)
                      for offset, source in enumerate(sorted(self.pools))}
        permutations = {source: [] for source in self.pools}
        positions = {source: 0 for source in self.pools}

        def next_index(source):
            if positions[source] == len(permutations[source]):
                permutations[source] = torch.randperm(len(self.pools[source]), generator=generators[source]).tolist()
                positions[source] = 0
            index = self.pools[source][permutations[source][positions[source]]]
            positions[source] += 1
            return index

        # Alternation also keeps every prefix and every even-size batch balanced.
        for _ in range(self.num_batches):
            for _ in range(self.batch_size // 2):
                yield next_index("math_dapo")
                yield next_index("codecontests")
