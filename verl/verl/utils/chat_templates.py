"""Explicit chat-template profiles for reproducible external harnesses."""

# Equivalent to agentica-project/rllm's DeepseekQwenChatTemplateParser at
# revision 7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4. Whitespace control is
# intentional: a one-user prompt must end in EOS + Assistant + ``<think>\n``.
RLLM_DEEPSEEK_QWEN_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}{{ message['content'] }}"
    "{%- elif message['role'] == 'user' %}{{ '<｜User｜>' + message['content'] }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{ '<｜Assistant｜>' + message['content'] + eos_token }}"
    "{%- else %}{{ raise_exception('Unsupported role: ' + message['role']) }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{ eos_token + '<｜Assistant｜><think>\n' }}"
    "{%- endif %}"
)

# Same shared DeepSeek/Qwen format, but without the extra real EOS inserted
# between the user turn and the generation assistant marker.  IF-specialist
# examples use this only when explicitly routed by data source; math/code keep
# the historical profile above.
RLLM_DEEPSEEK_QWEN_NO_PREASSISTANT_EOS_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}{{ message['content'] }}"
    "{%- elif message['role'] == 'user' %}{{ '<｜User｜>' + message['content'] }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{ '<｜Assistant｜>' + message['content'] + eos_token }}"
    "{%- else %}{{ raise_exception('Unsupported role: ' + message['role']) }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{ '<｜Assistant｜><think>\n' }}"
    "{%- endif %}"
)
