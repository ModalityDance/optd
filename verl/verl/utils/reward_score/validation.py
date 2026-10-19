"""Validation-only routing; training reward dispatch is deliberately unchanged."""


def compute_validation_score(data_source, solution_str, ground_truth, extra_info=None, *, fallback, **kwargs):
    source = data_source.lower()
    if source in {"amc23", "aimo", "math", "math_dapo", "math_dapo_reasoning"} or source.startswith("aime"):
        from .math_evaluation import extract_answer, grade_answer_verl

        correct = bool(grade_answer_verl(solution_str, str(ground_truth)))
        # Preserve the existing validation score scale and metric keys.
        return {"score": 1.0 if correct else -1.0, "acc": correct,
                "pred": extract_answer(solution_str) or ""}
    return fallback(data_source=data_source, solution_str=solution_str,
                    ground_truth=ground_truth, extra_info=extra_info, **kwargs)
