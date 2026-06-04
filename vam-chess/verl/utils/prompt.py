"""Prompt formatting helpers shared by training and evaluation code."""

from __future__ import annotations

from typing import Any


FALSE_STRINGS = {"0", "false", "no", "off"}
TRUE_STRINGS = {"1", "true", "yes", "on"}


def as_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in FALSE_STRINGS:
        return False
    if text in TRUE_STRINGS:
        return True
    return bool(default)


def is_qwen3_base_model(model: str | None) -> bool:
    model_lc = str(model or "").strip().lower()
    return "qwen3-" in model_lc and "-base" in model_lc


def infer_use_chat_template_from_model_name(model: str | None, *, default: bool = True) -> bool:
    """Return the default prompt wrapping mode for known model variants."""

    if is_qwen3_base_model(model):
        return False
    return bool(default)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item.get("text") or ""))
                elif "content" in item:
                    parts.append(_content_to_text(item.get("content")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def messages_to_plain_text(messages: Any) -> str:
    """Render chat-message-like records as plain text without role/control tokens."""

    if isinstance(messages, dict):
        return _content_to_text(messages.get("content")).strip()
    if not isinstance(messages, (list, tuple)):
        return str(messages or "").strip()

    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            text = _content_to_text(message.get("content"))
        else:
            text = str(message)
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def render_prompt_from_messages(
    tokenizer: Any,
    messages: Any,
    *,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Render a prompt either through the tokenizer chat template or as plain text."""

    if use_chat_template:
        kwargs = dict(apply_chat_template_kwargs or {})
        return str(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                **kwargs,
            )
        )
    return messages_to_plain_text(messages)


def encode_prompt_from_messages(
    tokenizer: Any,
    messages: Any,
    *,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[str, list[int]]:
    raw_prompt = render_prompt_from_messages(
        tokenizer,
        messages,
        use_chat_template=use_chat_template,
        add_generation_prompt=add_generation_prompt,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )
    return raw_prompt, [int(x) for x in tokenizer.encode(raw_prompt, add_special_tokens=False)]
