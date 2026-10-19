# Copyright 2026 OPRD contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Three independent OPD signals on common student top-k support."""

import torch


@torch.no_grad()
def mean_teacher_log_probs(teacher_outputs):
    """Stream three log-prob tensors into one FP32 sum; never stack teachers.

    This is a mean of log probabilities, not a probability-mixture teacher.
    With shared student_p weights and the single-epoch linear OPD objective,
    it gives the same gradient as the mean of three independent losses.
    """
    total = None
    count = 0
    for log_probs in teacher_outputs:
        if total is None:
            total = log_probs.to(dtype=torch.float32, copy=True)
        else:
            if total.shape != log_probs.shape:
                raise ValueError("Teachers must use identical student top-k support shapes")
            total.add_(log_probs)
        count += 1
        del log_probs
    if count != 3:
        raise ValueError("Expected exactly three teachers")
    return total.div_(count)
