"""Sampler that keeps every generated training batch within one data source.

The fixed-batch mode samples with replacement for a configured number of
batches. This is useful for step-budgeted comparisons whose source pools are
not divisible by a larger rollout batch size.
"""

from collections import defaultdict
from collections.abc import Iterator, Sized

import torch
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler


class SourceHomogeneousBatchSampler(AbstractSampler):
    """Shuffle complete source-specific batches, then interleave the sources.

    The DataLoader still supplies ``batch_size``; this sampler only controls
    the index order. It therefore preserves the existing batch/update settings
    while ensuring no batch contains multiple ``data_source`` values.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        super().__init__(data_source=data_source, data_config=data_config)
        self.data_config = data_config
        self.batch_size = int(data_config.get("train_batch_size", 0))
        if self.batch_size <= 0:
            raise ValueError("SourceHomogeneousBatchSampler requires data.train_batch_size")
        dataframe = getattr(data_source, "dataframe", None)
        if dataframe is not None and "data_source" in dataframe:
            source_values = list(dataframe["data_source"])
        else:
            # Some trainer paths pass a Dataset wrapper rather than RLHFDataset.
            # Use its indexed rows as the portable fallback after filtering.
            try:
                source_values = [data_source[index]["data_source"] for index in range(len(data_source))]
            except Exception as exc:
                raise ValueError("SourceHomogeneousBatchSampler requires indexed data_source rows") from exc
        self.seed = int(data_config.get("seed") or 0)
        pools = defaultdict(list)
        for index, source in enumerate(source_values):
            pools[str(source)].append(index)
        if len(pools) < 2:
            raise ValueError("SourceHomogeneousBatchSampler requires at least two data sources")
        self.pools = dict(pools)
        sampler_config = data_config.get("sampler", {})
        self.sampler_config = sampler_config
        self.num_batches = int(sampler_config.get("source_homogeneous_num_batches", 0))
        if self.num_batches <= 0:
            incomplete = {source: len(indices) % self.batch_size for source, indices in self.pools.items()}
            incomplete = {source: remainder for source, remainder in incomplete.items() if remainder}
            if incomplete:
                raise ValueError(
                    "source counts must be divisible by batch size unless "
                    "source_homogeneous_num_batches is set: "
                    f"batch_size={self.batch_size}, remainders={incomplete}"
                )
            self.num_batches = sum(len(indices) for indices in self.pools.values()) // self.batch_size
        self._num_samples = self.num_batches * self.batch_size

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed)
        sources = sorted(self.pools)
        weights = torch.tensor(
            [float(self.sampler_config.get("source_homogeneous_weights", {}).get(source, 1.0)) for source in sources],
            dtype=torch.float,
        )
        weights = weights / weights.sum()
        for _ in range(self.num_batches):
            source_index = int(torch.multinomial(weights, 1, generator=generator).item())
            indices = self.pools[sources[source_index]]
            draws = torch.randint(len(indices), (self.batch_size,), generator=generator).tolist()
            yield from (indices[i] for i in draws)
