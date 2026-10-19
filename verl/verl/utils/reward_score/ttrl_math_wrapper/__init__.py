"""Compatibility entrypoint for the historical math scorer."""
from __future__ import annotations

import inspect
from verl.utils.reward_score.ttrl_math import reward_func as _historical_reward_func

_ACCEPTED = set(inspect.signature(_historical_reward_func).parameters)


def reward_func(*args, **kwargs):
    """Preserve historical math grading while ignoring newer code-only kwargs."""
    return _historical_reward_func(*args, **{k: v for k, v in kwargs.items() if k in _ACCEPTED})
