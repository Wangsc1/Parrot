"""Regression contracts for the 2026-09-29 wire/tool/terminal audit."""
import json
import importlib
import pytest

from src import upstream
from src.protocols import errors, registry
from src.protocols.commit_gate import SseCommitGate
from src.protocols.sse import split_sse_events
from src.openai.transform import common


def event(name, **kw):
    return name, dict(type=name, **kw)


def frame(name, data, nl='\n', multiline=False):
    text = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    if multiline and isinstance(data, dict):
        text = text.replace(',', ',' + nl + 'data: ', 1)
    return ((f'event: {name}{nl}' if name else '') + 'data: ' + text + nl + nl).encode()


def decode(chunks):
    tail, blocks = split_sse_events(b''.join(chunks))
    assert not tail
    return [data for block in blocks for _, data in [upstream.parse_sse_event_bytes(block)] if data is not None]


M = {'id': 'msg1', 'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': '中文\u0085\u2028ok', 'annotations': []}]}
F = {'id': 'fc1', 'type': 'function_call', 'call_id': 'call1', 'name': 'Read', 'arguments': '{}', 'status': 'completed'}
R = [event('response.created', response={'id': 'r1', 'status': 'in_progress', 'output': []}), event('response.output_item.added', output_index=0, item={**M, 'content': [], 'status': 'in_progress'}), event('response.output_text.delta', output_index=0, item_id='msg1', content_index=0, delta='中文\u0085\u2028ok'), event('response.output_item.done', output_index=0, item=M), event('response.completed', response={'status': 'completed', 'output': [M], 'usage': {'input_tokens': 5, 'output_tokens': 2}})]
C = [('', {'id': 'chat1', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}), ('', {'choices': [{'index': 0, 'delta': {'content': '中文\u0085\u2028ok'}, 'finish_reason': None}]}), ('', {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 5, 'completion_tokens': 2}}), ('', '[DONE]')]
A = [event('message_start', message={'id': 'a1', 'role': 'assistant', 'type': 'message', 'content': [], 'usage': {'input_tokens': 5, 'output_tokens': 0}}), event('content_block_start', index=0, content_block={'type': 'text', 'text': ''}), event('content_block_delta', index=0, delta={'type': 'text_delta', 'text': '中文\u0085\u2028ok'}), event('content_block_stop', index=0), event('message_delta', delta={'stop_reason': 'end_turn'}, usage={'output_tokens': 2}), event('message_stop')]
CONVERTERS = [('stream_r2c', R, 'openai-chat'), ('stream_responses_to_anthropic', R, 'anthropic'), ('stream_c2r', C, 'openai-responses'), ('stream_chat_to_anthropic', C, 'anthropic'), ('stream_anthropic_to_responses', A, 'openai-responses'), ('stream_anthropic_to_chat', A, 'openai-chat')]


@pytest.mark.parametrize('module,events,protocol', CONVERTERS)
@pytest.mark.parametrize('nl,multiline', [('\n', False), ('\r\n', False), ('\r', False), ('\n', True)])
@pytest.mark.parametrize('bytewise', [False, True])
def test_six_directions_preserve_text_usage_across_wire_boundaries(module, events, protocol, nl, multiline, bytewise):
    tr = importlib.import_module('src.openai.transform.' + module).StreamTranslator(model='fixture')
    gate = SseCommitGate(protocol=protocol, stream_translator=tr)
    wire = b''.join(frame(n, d, nl, multiline) for n, d in events)
    chunks, committed = [], False
    for raw in ([wire[i:i+1] for i in range(len(wire))] if bytewise else [wire]):
        if not committed:
            step = gate.feed(raw)
            assert step.error_event is None
            chunks += step.downstream_chunks
            committed = bool(chunks)
        else:
            chunks += list(tr.feed(raw))
    chunks += list(tr.close())
    data = decode(chunks)
    text = ''.join(str(x['delta'].get('text', '') if isinstance(x.get('delta'), dict) else '') + ''.join(str((c.get('delta') or {}).get('content') or '') for c in x.get('choices', [])) + str(x.get('delta', '') if x.get('type') == 'response.output_text.delta' else '') for x in data)
    assert text == '中文\u0085\u2028ok'
    assert not any('error' in x or x.get('type') == 'error' for x in data)
    if protocol == 'openai-responses':
        terminal = next(x['response'] for x in data if x.get('type') == 'response.completed')
        assert terminal['usage']['input_tokens'] == 5
        assert terminal['usage']['output_tokens'] == 2


@pytest.mark.parametrize('terminal_output', [[M, F], [M], []])
def test_sparse_stream_identity_and_partial_terminal_do_not_duplicate_or_lose(terminal_output):
    builder = upstream.ResponsesSSEAssistantBuilder()
    for n, d in [event('response.output_item.done', output_index=2, item=M), event('response.output_item.done', output_index=4, item=F), event('response.output_item.done', output_index=9, item=F), event('response.completed', response={'status': 'completed', 'output': terminal_output})]:
        builder.feed(frame(n, d))
    assert builder.get_output_items() == [M, F]


@pytest.mark.parametrize('cleaning', [False, True])
def test_tool_done_and_terminal_emit_one_argument_value(cleaning):
    from src.openai.transform.stream_responses_to_anthropic import StreamTranslator
    schema = {'type': 'object', 'properties': {'file_path': {'type': 'string'}, 'pages': {'type': 'string'}}, 'required': ['file_path']}
    tr = StreamTranslator(model='fixture', request_body={'tools': [{'name': 'Read' if cleaning else 'Custom', 'input_schema': schema}]})
    item = {**F, 'name': 'Read' if cleaning else 'Custom', 'arguments': '{"file_path":"a.pdf","pages":"1-2"}'}
    chunks = []
    for n, d in [event('response.output_item.added', output_index=3, item={**item, 'arguments': '', 'status': 'in_progress'}), event('response.function_call_arguments.delta', output_index=3, item_id='fc1', delta=item['arguments']), event('response.output_item.done', output_index=3, item=item), event('response.completed', response={'status': 'completed', 'output': [item]})]:
        chunks += list(tr.feed(frame(n, d)))
    chunks += list(tr.close())
    data = decode(chunks)
    args = ''.join(x['delta']['partial_json'] for x in data if (x.get('delta') or {}).get('type') == 'input_json_delta')
    assert json.loads(args) == json.loads(item['arguments'])
    assert sum(x.get('type') == 'content_block_start' for x in data) == 1
    assert sum(x.get('type') == 'content_block_stop' for x in data) == 1


def test_valid_empty_string_retained_while_schema_invalid_empty_omitted():
    schema = {'properties': {'suffix': {'type': 'string', 'default': 'KEEP'}, 'selector': {'type': 'string', 'minLength': 1}}}
    fields = common.optional_empty_string_fields_from_tool_schema(schema)
    assert fields == {'selector'}
    assert common.normalize_tool_input_optional_empty_strings('custom', {'suffix': '', 'selector': ''}, {'custom': fields}) == {'suffix': ''}


@pytest.mark.parametrize('reason,finish', [('max_output_tokens', 'length'), ('content_filter', 'content_filter')])
def test_incomplete_is_forwarded_not_context_error(reason, finish):
    from src.openai.transform.stream_r2c import StreamTranslator
    obj = {'status': 'incomplete', 'output': [], 'incomplete_details': {'reason': reason}}
    name, payload = event('response.incomplete', response=obj)
    assert not upstream.is_stream_error_event(name, payload)
    assert not registry.is_openai_error_json(obj)
    tracker = upstream.ResponsesSSEUsageTracker()
    tracker.feed(frame(name, payload))
    assert tracker.saw_stream_end and not tracker.saw_stream_error
    tr = StreamTranslator(model='fixture')
    gate = SseCommitGate(protocol='openai-chat', stream_translator=tr)
    step = gate.feed(frame(name, payload))
    assert step.error_event is None
    data = decode([*step.downstream_chunks, *tr.close()])
    assert not any('error' in x for x in data)
    assert [c['finish_reason'] for x in data for c in x.get('choices', []) if c.get('finish_reason')] == [finish]
    if reason == 'max_output_tokens':
        code, msg = errors.extract_error_info(payload)
        assert not errors.is_context_length_code_or_message(code, msg)


@pytest.mark.parametrize('events,tracker_cls,builder_cls', [
    (A, upstream.SSEUsageTracker, upstream.SSEAssistantBuilder),
    (C, upstream.ChatSSEUsageTracker, upstream.ChatSSEAssistantBuilder),
    (R, upstream.ResponsesSSEUsageTracker, upstream.ResponsesSSEAssistantBuilder),
])
@pytest.mark.parametrize('nl', ['\n', '\r\n', '\r'])
def test_usage_and_history_builders_accept_same_wire_rules(events, tracker_cls, builder_cls, nl):
    tracker, builder = tracker_cls(), builder_cls()
    wire = b''.join(frame(n, d, nl, True) for n, d in events)
    for value in wire:
        tracker.feed(bytes([value]))
        builder.feed(bytes([value]))
    assert tracker.saw_stream_end and not tracker.saw_stream_error
    assert tracker.usage['input_tokens'] == 5
    assert tracker.usage['output_tokens'] == 2
    assert '中文\u0085\u2028ok' in json.dumps(builder.get_assistant(), ensure_ascii=False)


def test_anthropic_tool_truncation_stays_length():
    from src.openai.transform.stream_anthropic_to_chat import StreamTranslator
    tr = StreamTranslator(model='fixture')
    chunks = []
    for n, d in [A[0], event('content_block_start', index=0, content_block={'type': 'tool_use', 'id': 'call1', 'name': 'Read', 'input': {}}), event('content_block_delta', index=0, delta={'type': 'input_json_delta', 'partial_json': '{}'}), event('content_block_stop', index=0), event('message_delta', delta={'stop_reason': 'max_tokens'}, usage={'output_tokens': 2}), event('message_stop')]:
        chunks += list(tr.feed(frame(n, d)))
    data = decode([*chunks, *tr.close()])
    assert [c['finish_reason'] for x in data for c in x.get('choices', []) if c.get('finish_reason')] == ['length']
