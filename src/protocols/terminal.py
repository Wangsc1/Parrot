"""Completed JSON response validation, independent of output being empty."""
from __future__ import annotations


def non_stream_terminal_error(obj, protocol: str, *, allow_async: bool = False) -> str | None:
    if not isinstance(obj, dict):
        return 'upstream response must be a JSON object'
    if isinstance(obj.get('error'), dict) or obj.get('type') == 'error':
        return None  # Preserve the existing typed upstream error classifier.
    if protocol == 'openai-responses':
        status = obj.get('status')
        if status in ('completed', 'incomplete'):
            return None
        if allow_async and status in ('queued', 'in_progress'):
            return None
        return f'upstream Responses status is not a consumable terminal: {status!r}'
    if protocol == 'openai-chat':
        choices = obj.get('choices')
        if not isinstance(choices, list) or not choices:
            return 'upstream Chat response has no terminal choice'
        for choice in choices:
            if not isinstance(choice, dict) or choice.get('finish_reason') not in ('stop', 'length', 'tool_calls', 'function_call', 'content_filter'):
                return 'upstream Chat choice is missing a valid finish_reason'
    elif protocol == 'anthropic':
        if obj.get('stop_reason') not in ('end_turn', 'stop_sequence', 'tool_use', 'max_tokens', 'model_context_window_exceeded', 'pause_turn', 'refusal'):
            return 'upstream Anthropic message is missing a valid stop_reason'
    return None
