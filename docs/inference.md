# Use a released OPTD model

Choose one of the [six released models](../README.md#checkpoints). Each contains complete student weights and a tokenizer: inference does not require a teacher or the training framework.

## Run with Transformers

Install PyTorch for your hardware, Transformers with Qwen3 support, and Accelerate. The training environment described in the README can also run this example.

```bash
python -m pip install torch transformers accelerate
```

The following example uses Qwen3-4B-Late on hardware that supports BF16:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ModalityDance/OPTD-Qwen3-4B-Late"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
messages = [{"role": "user", "content": "Solve 2x + 5 = 17."}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.95
    )
response = output[0, inputs.input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

For either **Qwen3-4B** model, keep `enable_thinking=False`, matching the paper's non-thinking setting. For an **R1-Distill** model, change `model_id` and omit that argument to use its native reasoning template. The model's tokenizer supplies the chat format. Since the example first formats the chat as text, `add_special_tokens=False` avoids adding special tokens a second time.

The 1,024-token limit is for this short demonstration; longer reasoning tasks may need more tokens. Use the [evaluation guide](evaluation.md) for the paper's sampling settings and benchmark token limits.

## Download a local copy

```bash
python -m pip install huggingface_hub
hf download ModalityDance/OPTD-Qwen3-4B-Late --local-dir models/optd-qwen3-4b-late
```

Then use `models/optd-qwen3-4b-late` as `model_id` in the example. Teacher checkpoints are only needed to train a new student.

## Reproduce a release version

[released_checkpoints.json](../configs/released_checkpoints.json) records the original release commit for each model. For the 4B Late release:

```bash
hf download ModalityDance/OPTD-Qwen3-4B-Late \
  --revision c628ec8524a1564d62b83c163c0f2c479c7216f8 \
  --local-dir models/optd-qwen3-4b-late
```

You can also pass that commit as `revision=` to both `from_pretrained` calls. Each model repository includes `training_metadata.json` and `checksums.json`, which records SHA256 hashes for the checkpoint and tokenizer files. Releases contain inference weights rather than optimizer state; they do not restore the complete state of an interrupted training run.
