"""Ordered output and terminal-snapshot regression tests (offline converter functions)."""
from __future__ import annotations

import json

from src import config
from src.openai import store
from src.openai.transform import anthropic_to_responses, responses_to_anthropic, responses_to_chat
from src.openai.transform.stream_chat_to_anthropic import StreamTranslator as ChatToAnthropic
from src.openai.transform.stream_c2r import StreamTranslator as ChatToResponses
from src.openai.transform.stream_r2c import StreamTranslator as ResponsesToChat
from src.openai.transform.stream_responses_to_anthropic import StreamTranslator as ResponsesToAnthropic
from src.search_hosted_codec import responses_to_anthropic as web_blocks


def msg(label, text):
    return {"id": label, "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text}]}


def call(label):
    return {"id": "fc_" + label, "type": "function_call", "call_id": label,
            "name": "lookup", "arguments": "{}", "status": "completed"}


def event_stream(translator, events, *, chat=False):
    raw = b"".join((('data: ' if chat else 'event: ' + event['type'] + '\ndata: ')
                    + json.dumps(event) + '\n\n').encode() for event in events)
    chunks = list(translator.feed(raw)) + list(translator.close())
    return [json.loads(line[6:]) for line in b''.join(chunks).decode().splitlines()
            if line.startswith('data: ') and line != 'data: [DONE]']


def types(items):
    return [item['type'] for item in items]


def test_nonstream_both_directions_and_hosted_history_order(monkeypatch):
    monkeypatch.setitem(config.get().setdefault('openai', {}), 'reasoningBridge', 'passthrough')
    source = {'id': 'm', 'model': 'x', 'stop_reason': 'tool_use', 'content': [
        {'type': 'text', 'text': 'A'}, {'type': 'tool_use', 'id': 'a', 'name': 'lookup', 'input': {}},
        {'type': 'text', 'text': 'B'}, {'type': 'tool_use', 'id': 'b', 'name': 'lookup', 'input': {}},
        {'type': 'text', 'text': 'C'},
    ]}
    saved = []
    monkeypatch.setattr(store, 'is_enabled', lambda: True)
    monkeypatch.setattr(store, 'save', lambda **kwargs: saved.append(kwargs))
    response = responses_to_anthropic.translate_response(source, model='x', api_key_name='offline',
                                                         current_input_items=[])
    output = response['output']
    assert saved[0]['output_items'] == output
    assert types(output) == ['message', 'function_call', 'message', 'function_call', 'message']
    assert [item['content'][0]['text'] for item in output if item['type'] == 'message'] == ['A', 'B', 'C']
    replay = responses_to_anthropic.translate_request({'model': 'x', 'input': output}, store_enabled=False)
    assert types(replay['messages'][0]['content']) == ['text', 'tool_use', 'text', 'tool_use', 'text']
    upstream = {'status': 'completed', 'output_text': 'ABC', 'output': output}
    for bridge in (False, True):
        converted = anthropic_to_responses.translate_response(upstream, model='x', allow_reasoning_bridge=bridge)
        assert types(converted['content']) == ['text', 'tool_use', 'text', 'tool_use', 'text']
    web = {'type': 'web_search_call', 'id': 'ws', 'status': 'completed', 'action': {'type': 'search', 'query': 'q'}}
    hosted_source = {**source, 'content': [{'type': 'text', 'text': 'A'}, *web_blocks(web), {'type': 'text', 'text': 'B'}]}
    hosted_output = responses_to_anthropic.translate_response(hosted_source, model='x')['output']
    assert types(hosted_output) == ['message', 'web_search_call', 'message']
    assert types(anthropic_to_responses.translate_response({'output': hosted_output})['content']) == [
        'text', 'server_tool_use', 'web_search_tool_result', 'text']


def test_chat_to_anthropic_reopened_text_block_matches_snapshot_and_replay():
    translator = ChatToAnthropic(model='x')
    events = event_stream(translator, [
        {'choices': [{'delta': {'content': 'A'}}]},
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'a', 'type': 'function',
                                             'function': {'name': 'lookup', 'arguments': '{}'}}]}}]},
        {'choices': [{'delta': {'content': 'B'}, 'finish_reason': 'tool_calls'}]},
    ], chat=True)
    starts = [e for e in events if e['type'] == 'content_block_start']
    assert [(e['index'], e['content_block']['type']) for e in starts] == [(0, 'text'), (1, 'tool_use'), (2, 'text')]
    assistant = translator.get_downstream_anthropic_assistant()
    assert types(assistant['content']) == ['text', 'tool_use', 'text']
    replay = anthropic_to_responses.translate_request({'model': 'x', 'max_tokens': 20,
                                                        'messages': [assistant]})
    assert types(replay['input']) == ['message', 'function_call', 'message']


def test_responses_to_anthropic_terminal_and_done_text_are_not_duplicated():
    output = [msg('m0', 'AB'), call('a'), msg('m2', 'C')]
    events = [{'type': 'response.output_item.added', 'output_index': 0, 'item': {'id': 'm0', 'type': 'message'}},
              {'type': 'response.output_text.delta', 'output_index': 0, 'item_id': 'm0', 'content_index': 0, 'delta': 'A'},
              {'type': 'response.output_item.done', 'output_index': 0, 'item': output[0]},
              {'type': 'response.output_item.done', 'output_index': 1, 'item': output[1]},
              {'type': 'response.completed', 'response': {'status': 'completed', 'output': output}}]
    translator = ResponsesToAnthropic(model='x')
    stream = event_stream(translator, events)
    assert [(e['index'], e['delta']['text']) for e in stream if e['type'] == 'content_block_delta'
            and e['delta']['type'] == 'text_delta'] == [(0, 'A'), (0, 'B'), (2, 'C')]
    content = translator.get_downstream_anthropic_assistant()['content']
    assert types(content) == ['text', 'tool_use', 'text']
    assert [part['text'] for part in content if part['type'] == 'text'] == ['AB', 'C']
    replay = anthropic_to_responses.translate_request({'model': 'x', 'max_tokens': 20,
                                                        'messages': [{'role': 'assistant', 'content': content}]})
    assert types(replay['input']) == ['message', 'function_call', 'message']
    terminal_only = ResponsesToAnthropic(model='x')
    terminal_events = event_stream(terminal_only, [events[-1]])
    assert [e['content_block']['type'] for e in terminal_events if e['type'] == 'content_block_start'] == [
        'text', 'tool_use', 'text']
    assert terminal_only.get_downstream_anthropic_assistant()['content'] == content


def test_responses_to_anthropic_hosted_late_done_and_signature_only_shape():
    web = {'type': 'web_search_call', 'id': 'ws', 'status': 'completed',
           'action': {'type': 'search', 'query': 'q'}}
    translator = ResponsesToAnthropic(model='x')
    stream = event_stream(translator, [
        {'type': 'response.output_item.added', 'output_index': 0, 'item': {'type': 'message', 'id': 'm0'}},
        {'type': 'response.output_text.delta', 'output_index': 0, 'item_id': 'm0', 'delta': 'A'},
        {'type': 'response.output_item.added', 'output_index': 1, 'item': {'type': 'web_search_call', 'id': 'ws'}},
        {'type': 'response.output_item.added', 'output_index': 2, 'item': {'type': 'message', 'id': 'm2'}},
        {'type': 'response.output_text.delta', 'output_index': 2, 'item_id': 'm2', 'delta': 'B'},
        {'type': 'response.output_item.done', 'output_index': 1, 'item': web},
        {'type': 'response.completed', 'response': {'status': 'completed', 'output': [msg('m0', 'A'), web, msg('m2', 'B')]}},
    ])
    kinds = [e['content_block']['type'] for e in stream if e['type'] == 'content_block_start']
    assert kinds == types(translator.get_downstream_anthropic_assistant()['content']) == [
        'text', 'server_tool_use', 'web_search_tool_result', 'text']
    hosted_replay = anthropic_to_responses.translate_request({'model': 'x', 'max_tokens': 20,
        'messages': [translator.get_downstream_anthropic_assistant()]})
    assert types(hosted_replay['input']) == ['message', 'web_search_call', 'message']
    # A preceding item's final text may itself arrive after hosted.added.
    preceding = ResponsesToAnthropic(model='x')
    before = event_stream(preceding, [
        {'type': 'response.output_item.added', 'output_index': 0, 'item': {'type': 'message', 'id': 'm0'}},
        {'type': 'response.output_text.delta', 'output_index': 0, 'item_id': 'm0', 'delta': 'A'},
        {'type': 'response.output_item.added', 'output_index': 1, 'item': {'type': 'web_search_call', 'id': 'ws'}},
        {'type': 'response.output_item.done', 'output_index': 0, 'item': msg('m0', 'AB')},
        {'type': 'response.output_item.done', 'output_index': 1, 'item': web},
        {'type': 'response.completed', 'response': {'status': 'completed', 'output': [msg('m0', 'AB'), web]}},
    ])
    assert [e['content_block']['type'] for e in before if e['type'] == 'content_block_start'] == [
        'text', 'server_tool_use', 'web_search_tool_result']
    assert preceding.get_downstream_anthropic_assistant()['content'][0]['text'] == 'AB'
    translator = ResponsesToAnthropic(model='x', allow_reasoning_bridge=True)
    stream = event_stream(translator, [
        {'type': 'response.output_item.added', 'output_index': 0,
         'item': {'type': 'reasoning', 'id': 'r', 'encrypted_content': 'sig'}},
        {'type': 'response.output_item.added', 'output_index': 1, 'item': {'type': 'message', 'id': 'm'}},
        {'type': 'response.output_text.delta', 'output_index': 1, 'item_id': 'm', 'delta': 'A'},
        {'type': 'response.output_item.done', 'output_index': 0,
         'item': {'type': 'reasoning', 'id': 'r', 'encrypted_content': 'sig', 'summary': []}},
        {'type': 'response.completed', 'response': {'status': 'completed', 'output': [
            {'type': 'reasoning', 'id': 'r', 'encrypted_content': 'sig', 'summary': []}, msg('m', 'A')]}}])
    assert [e['content_block']['type'] for e in stream if e['type'] == 'content_block_start'] == [
        'redacted_thinking', 'text']
    assert types(translator.get_downstream_anthropic_assistant()['content']) == ['redacted_thinking', 'text']
    signature_replay = anthropic_to_responses.translate_request({'model': 'x', 'max_tokens': 20,
        'messages': [translator.get_downstream_anthropic_assistant()]}, allow_reasoning_effort=True)
    assert types(signature_replay['input']) == ['reasoning', 'message']
    # Readable summary only appears at reasoning.done, after the later text delta.
    late_readable = ResponsesToAnthropic(model='x', allow_reasoning_bridge=True)
    late_stream = event_stream(late_readable, [
        {'type': 'response.output_item.added', 'output_index': 0,
         'item': {'type': 'reasoning', 'id': 'r2', 'encrypted_content': 'sig2'}},
        {'type': 'response.output_item.added', 'output_index': 1, 'item': {'type': 'message', 'id': 'm2'}},
        {'type': 'response.output_text.delta', 'output_index': 1, 'item_id': 'm2', 'delta': 'A'},
        {'type': 'response.output_item.done', 'output_index': 0,
         'item': {'type': 'reasoning', 'id': 'r2', 'encrypted_content': 'sig2',
                  'summary': [{'type': 'summary_text', 'text': 'thought'}]}},
        {'type': 'response.completed', 'response': {'status': 'completed', 'output': [
            {'type': 'reasoning', 'id': 'r2', 'encrypted_content': 'sig2',
             'summary': [{'type': 'summary_text', 'text': 'thought'}]}, msg('m2', 'A')]}}])
    assert [e['content_block']['type'] for e in late_stream if e['type'] == 'content_block_start'] == ['thinking', 'text']
    assert types(late_readable.get_downstream_anthropic_assistant()['content']) == ['thinking', 'text']
    assert late_readable.get_downstream_anthropic_assistant()['content'][0]['thinking'] == 'thought'


def test_mixed_responses_done_and_later_delta_keep_visible_order():
    for first, early in [('A', 'done'), ('AA', 'delta_then_late_done'), ('A', 'complete_delta')]:
        output = [msg('m0', first), msg('m1', 'B')]
        stream_events = []
        if early == 'done':
            stream_events.append({'type': 'response.output_item.done', 'output_index': 0, 'item': output[0]})
        else:
            stream_events.extend([
                {'type': 'response.output_item.added', 'output_index': 0,
                 'item': {'type': 'message', 'id': 'm0'}},
                {'type': 'response.output_text.delta', 'output_index': 0,
                 'content_index': 0, 'item_id': 'm0', 'delta': 'A'},
            ])
        stream_events.extend([
            {'type': 'response.output_text.delta', 'output_index': 1,
             'content_index': 0, 'item_id': 'm1', 'delta': 'B'},
        ])
        if early != 'done':
            stream_events.append({'type': 'response.output_item.done', 'output_index': 0, 'item': output[0]})
        stream_events.append({'type': 'response.completed', 'response': {'status': 'completed', 'output': output}})
        expected = 'AAB' if early == 'delta_then_late_done' else 'AB'
        anthro = ResponsesToAnthropic(model='x')
        events = event_stream(anthro, stream_events)
        assert ''.join(e['delta']['text'] for e in events if e['type'] == 'content_block_delta'
                       and e['delta']['type'] == 'text_delta') == expected
        saved = anthro.get_downstream_anthropic_assistant()['content']
        assert ''.join(b['text'] for b in saved if b['type'] == 'text') == expected
        replay = anthropic_to_responses.translate_request({'model': 'x', 'max_tokens': 20,
            'messages': [{'role': 'assistant', 'content': saved}]})['input']
        assert ''.join(part.get('text', '') for item in replay if item['type'] == 'message'
                       for part in item['content']) == expected
        chat = ResponsesToChat(model='x')
        chunks = event_stream(chat, stream_events)
        assert ''.join(e['choices'][0]['delta'].get('content', '') for e in chunks if e.get('choices')) == expected
        assert chat.get_downstream_chat_assistant()['content'] == expected
        if early == 'complete_delta':
            for translator in (ResponsesToAnthropic(model='x'), ResponsesToChat(model='x')):
                raw = 'event: response.output_text.delta\\ndata: ' + json.dumps(stream_events[1]) + '\\n\\n'
                first_chunks = b''.join(translator.feed(raw.replace('\\n', '\n').encode()))
                assert b'A' in first_chunks  # Normal first delta is emitted immediately.


def test_two_text_parts_early_delta_late_complete_suffix_are_ordered():
    message = msg('m0', 'AA')
    message['content'].append({'type': 'output_text', 'text': 'B'})
    events = [{'type': 'response.output_item.added', 'output_index': 0,
               'item': {'type': 'message', 'id': 'm0'}},
              {'type': 'response.output_text.delta', 'output_index': 0, 'content_index': 0,
               'item_id': 'm0', 'delta': 'A'},
              {'type': 'response.output_text.delta', 'output_index': 0, 'content_index': 1,
               'item_id': 'm0', 'delta': 'B'},
              {'type': 'response.output_text.done', 'output_index': 0, 'content_index': 0,
               'item_id': 'm0', 'text': 'AA'},
              {'type': 'response.output_item.done', 'output_index': 0, 'item': message},
              {'type': 'response.completed', 'response': {'status': 'completed', 'output': [message]}}]
    anthro = ResponsesToAnthropic(model='x')
    frames = event_stream(anthro, events)
    assert ''.join(e['delta']['text'] for e in frames if e['type'] == 'content_block_delta'
                   and e['delta']['type'] == 'text_delta') == 'AAB'
    assert ''.join(b['text'] for b in anthro.get_downstream_anthropic_assistant()['content']) == 'AAB'
    chat = ResponsesToChat(model='x')
    chunks = event_stream(chat, events)
    assert ''.join(e['choices'][0]['delta'].get('content', '') for e in chunks if e.get('choices')) == 'AAB'
    assert chat.get_downstream_chat_assistant()['content'] == 'AAB'
    # No delta/done for the first part: the enclosing item.done is the first
    # complete evidence and must precede the already-observed second-part delta.
    done_only = [events[0], events[2], events[4], events[5]]
    for translator in (ResponsesToAnthropic(model='x'), ResponsesToChat(model='x')):
        frames = event_stream(translator, done_only)
        if isinstance(translator, ResponsesToAnthropic):
            visible = ''.join(e['delta']['text'] for e in frames if e['type'] == 'content_block_delta'
                              and e['delta']['type'] == 'text_delta')
            saved = ''.join(b['text'] for b in translator.get_downstream_anthropic_assistant()['content'])
        else:
            visible = ''.join(e['choices'][0]['delta'].get('content', '') for e in frames if e.get('choices'))
            saved = translator.get_downstream_chat_assistant()['content']
        assert visible == saved == 'AAB'


def test_responses_to_chat_terminal_text_and_reasoning_bridge(monkeypatch):
    monkeypatch.setitem(config.get().setdefault('openai', {}), 'reasoningBridge', 'passthrough')
    output = [msg('m0', 'A'), call('a'), msg('m2', 'B')]
    translator = ResponsesToChat(model='x')
    stream = event_stream(translator, [{'type': 'response.completed', 'response': {
        'status': 'completed', 'output': output}}])
    assert ''.join(e['choices'][0]['delta'].get('content', '') for e in stream if e.get('choices')) == 'AB'
    assert [t['id'] for t in translator.get_downstream_chat_assistant()['tool_calls']] == ['a']
    hist = [*output[:2], {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 'for B'}]}, output[2]]
    messages = responses_to_chat.translate_request({'model': 'x', 'input': hist})['messages']
    assert [(m.get('content'), m.get('reasoning_content')) for m in messages] == [('A', ''), ('B', 'for B')]
    before_tool = [output[0], {'type': 'reasoning', 'summary': [
        {'type': 'summary_text', 'text': 'for tool'}]}, output[1]]
    tool_messages = responses_to_chat.translate_request({'model': 'x', 'input': before_tool})['messages']
    assert tool_messages[0]['reasoning_content'] == 'for tool'
    assert [c['id'] for c in tool_messages[0]['tool_calls']] == ['a']


def test_refusal_first_content_index_matches_terminal_content_and_output_text():
    translator = ChatToResponses(model='x')
    stream = event_stream(translator, [
        {'choices': [{'delta': {'refusal': 'N'}}]},
        {'choices': [{'delta': {'content': 'Y'}, 'finish_reason': 'stop'}]},
    ], chat=True)
    part_indices = {e['part']['type']: e['content_index'] for e in stream if e['type'] == 'response.content_part.added'}
    output = stream[-1]['response']['output'][0]
    assert [part['type'] for part in output['content']] == ['refusal', 'output_text']
    assert all(output['content'][index]['type'] == kind for kind, index in part_indices.items())
    assert stream[-1]['response']['output_text'] == 'Y'
