"""Lossless legacy Chat function normalization at cross-protocol boundaries."""
from __future__ import annotations

import copy
from collections import defaultdict, deque


def normalize_request(body: dict) -> dict:
    out = dict(body)
    if isinstance(body.get('functions'), list):
        tools = list(body.get('tools') or [])
        names = {(tool.get('function') or {}).get('name') for tool in tools if isinstance(tool, dict)}
        for fn in body['functions']:
            if isinstance(fn, dict) and fn.get('name') not in names:
                tools.append({'type': 'function', 'function': copy.deepcopy(fn)})
                names.add(fn.get('name'))
        out['tools'] = tools
    if 'tool_choice' not in body and 'function_call' in body:
        choice = body['function_call']
        out['tool_choice'] = ({'type': 'function', 'function': {'name': choice['name']}}
                              if isinstance(choice, dict) and choice.get('name') else choice)
    # Modern controls take precedence; do not emit the legacy declarations twice.
    out.pop('functions', None)
    out.pop('function_call', None)
    messages = body.get('messages') or []
    reserved = {str(tc.get('id')) for msg in messages if isinstance(msg, dict)
                for tc in msg.get('tool_calls') or [] if isinstance(tc, dict) and tc.get('id')}
    pending = defaultdict(deque)
    converted = []
    for index, source in enumerate(messages):
        if not isinstance(source, dict):
            converted.append(source)
            continue
        msg = dict(source)
        legacy = msg.get('function_call')
        if msg.get('role') == 'assistant' and isinstance(legacy, dict) and not msg.get('tool_calls'):
            call_id = f'call_legacy_{index}'
            while call_id in reserved:
                call_id += '_'
            reserved.add(call_id)
            name = str(legacy.get('name') or '')
            pending[name].append(call_id)
            msg['tool_calls'] = [{'type': 'function', 'id': call_id, 'function': copy.deepcopy(legacy)}]
            msg.pop('function_call', None)
        elif msg.get('role') == 'function':
            name = str(msg.get('name') or '')
            msg['role'] = 'tool'
            msg['tool_call_id'] = pending[name].popleft() if pending[name] else name
            msg.pop('name', None)
        converted.append(msg)
    out['messages'] = converted
    return out


def normalize_response(obj: dict) -> dict:
    out = dict(obj)
    choices = []
    for index, source in enumerate(obj.get('choices') or []):
        choice = dict(source)
        msg = dict(choice.get('message') or {})
        legacy = msg.get('function_call')
        if isinstance(legacy, dict) and not msg.get('tool_calls'):
            msg['tool_calls'] = [{'type': 'function', 'id': f'call_legacy_{index}', 'function': copy.deepcopy(legacy)}]
            msg.pop('function_call', None)
            if choice.get('finish_reason') == 'function_call':
                choice['finish_reason'] = 'tool_calls'
        choice['message'] = msg
        choices.append(choice)
    out['choices'] = choices
    return out
