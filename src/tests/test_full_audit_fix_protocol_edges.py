"""Protocol closure edges beyond the original audit probes."""
import copy
import json
import httpx
import pytest
from src.tests import test_protocol_fake_upstreams as h
from src.tests.test_full_audit_fix_protocol_stream import frame, chat, terminal, fn, run, decoded, chat_assistant
from src.openai.transform import chat_to_anthropic, chat_to_responses, responses_to_chat
from src.openai.transform import stream_r2c, stream_responses_to_anthropic as r2a
from src.openai.transform.guard import GuardError


def _import_modules():
    return h._import_modules()


def setup(m, protocol='openai-responses', stream_only=False):
    h._setup(m)
    h._install_keys(m, h._default_key())
    ch = h._make_openai_channel('edge', 'https://edge.invalid', protocol=protocol, alias='edge', real='target')
    ch.upstream_stream_only = stream_only
    h._install_channels(m, [ch])
    return h.MockRouter()


@pytest.mark.parametrize('stream', [True, False])
@pytest.mark.parametrize('batch', [True, False])
async def test_chat_stop_http_stream_and_stream_only_json(m, stream, batch):
    router = setup(m, stream_only=not stream)
    tool = {**fn(), 'arguments': '{}'}
    frames = [frame('response.output_text.delta', output_index=0, content_index=0, delta=s)
              for s in ['before<ST', 'OP>after']]
    frames += [frame('response.output_item.added', output_index=1, item=tool), terminal([])]
    router.register('https://edge.invalid', lambda req: httpx.Response(200, stream=h.ChunkedByteStream([b''.join(frames)] if batch else frames), headers={'content-type': 'text/event-stream'}))
    response, client = await h._call_openai_handler(m, router, 'chat', {'model':'edge', 'messages':[{'role':'user','content':'hi'}], 'stream':stream, 'stop':['<STOP>'], 'stream_options':{'include_usage':True}})
    try:
        assert response.status_code == 200
        if stream:
            wire = (await h._consume_streaming_to_string(response)).encode()
            obj = chat_assistant(wire)
            events = decoded(wire)
            assert any((e.get('usage') or {}).get('total_tokens') == 5 for e in events)
            assert [c['finish_reason'] for e in events for c in e.get('choices', []) if c.get('finish_reason')] == ['stop']
        else:
            full = json.loads(response.body)
            obj = full['choices'][0]['message']
            assert full['choices'][0]['finish_reason'] == 'stop'
            assert full['usage']['total_tokens'] == 5
        assert obj['content'] == 'before'
        assert not obj.get('tool_calls')
        assert m['log_db']._get_conn().execute('select status from request_log order by id desc limit 1').fetchone()[0] == 'success'
    finally:
        await client.aclose()


@pytest.mark.parametrize('matched', [True, False])
def test_stop_wrapper_getter_and_pending_prefix(matched):
    from src.protocols.runtime import make_stream_translator
    tr = make_stream_translator({'response_translator':'chat_to_responses', 'request_body':{'stop':'<STOP>'}})
    text = 'hello<ST' + ('OP>tail' if matched else '')
    out = run(tr, [frame('response.output_text.delta', output_index=0, content_index=0, delta=text), terminal([])])
    expected = 'hello' if matched else text
    assert chat_assistant(out)['content'] == expected
    assert tr.get_downstream_chat_assistant()['content'] == expected


@pytest.mark.parametrize('status', ['completed', 'incomplete'])
@pytest.mark.parametrize('ingress', ['chat', 'anthropic'])
async def test_empty_legal_nonstream_terminal(m, status, ingress):
    router = setup(m)
    source = {'id':'r_empty','object':'response','status':status,'output':[], 'error':None,
              'incomplete_details': {'reason':'max_output_tokens'} if status=='incomplete' else None}
    router.register('https://edge.invalid', lambda req: httpx.Response(200,json=source))
    body = {'model':'edge','messages':[{'role':'user','content':'hi'}],'max_tokens':32,'stream':False}
    if ingress == 'chat':
        response, client = await h._call_openai_handler(m,router,'chat',body)
    else:
        response, client, _ = await h._call_anthropic_core(m,router,body)
    try:
        assert response.status_code == 200, response.body
        obj = json.loads(response.body)
        if ingress == 'chat':
            assert obj['choices'][0]['finish_reason'] == ('length' if status=='incomplete' else 'stop')
        else:
            assert obj['stop_reason'] == ('max_tokens' if status=='incomplete' else 'end_turn')
    finally:
        await client.aclose()


@pytest.mark.parametrize('status', ['queued','in_progress'])
async def test_native_async_not_coerced(m, status):
    router = setup(m)
    source = {'id':'r_async','object':'response','status':status,'output':[],'error':None}
    router.register('https://edge.invalid', lambda req: httpx.Response(200,json=source))
    response, client = await h._call_openai_handler(m,router,'responses',{'model':'edge','input':'hi','background':True,'stream':False})
    try:
        assert response.status_code == 200, response.body
        assert json.loads(response.body)['status'] == status
    finally:
        await client.aclose()


@pytest.mark.parametrize('target', ['responses','anthropic'])
def test_legacy_repeated_same_name_call_ids(target):
    body = {'model':'x','functions':[{'name':'f','parameters':{'type':'object'}}], 'function_call':{'name':'f'}, 'messages':[]}
    for i in range(2):
        body['messages'] += [{'role':'assistant','function_call':{'name':'f','arguments':'{}'}}, {'role':'function','name':'f','content':str(i)}]
    before = copy.deepcopy(body)
    if target == 'responses':
        sent = chat_to_responses.translate_request(body)
        calls = [i['call_id'] for i in sent['input'] if i.get('type') == 'function_call']
        results = [i['call_id'] for i in sent['input'] if i.get('type') == 'function_call_output']
    else:
        sent = chat_to_anthropic.translate_request(body)
        calls = [b['id'] for m in sent['messages'] for b in m['content'] if b['type']=='tool_use']
        results = [b['tool_use_id'] for m in sent['messages'] for b in m['content'] if b['type']=='tool_result']
        assert len(sent['tools']) == 1
    assert len(set(calls)) == 2 and calls == results and body == before


@pytest.mark.parametrize('target', ['chat','anthropic'])
def test_expanded_native_only_history_is_candidate(monkeypatch, target):
    from src.openai import store
    from src.openai.transform import responses_to_anthropic
    monkeypatch.setattr(store, 'is_enabled', lambda:True)
    calls = []
    def expand(*args, **kwargs):
        calls.append(1)
        return [{'type':'computer_call','id':'cc_1','status':'completed','action':{'type':'click','x':10,'y':20}}]
    monkeypatch.setattr(store,'expand_history',expand)
    translate = responses_to_chat.translate_request if target=='chat' else responses_to_anthropic.translate_request
    with pytest.raises(GuardError) as caught:
        translate({'model':'x','previous_response_id':'r1','input':'continue'},api_key_name='tenant')
    assert caught.value.scope == 'candidate' and calls == [1]


@pytest.mark.parametrize('translator', [stream_r2c,r2a])
def test_text_snapshot_conflict_fails_without_success(translator):
    item = {'type':'message','id':'m1','role':'assistant','content':[{'type':'output_text','text':'wrong'}]}
    out = run(translator.StreamTranslator(model='x'), [frame('response.output_text.delta',output_index=0,item_id='m1',content_index=0,delta='right'),terminal([item])])
    events = decoded(out)
    assert any('error' in e for e in events)
    assert not any(e.get('type')=='message_stop' or any(c.get('finish_reason') for c in e.get('choices',[])) for e in events)


@pytest.mark.parametrize('protocol', ['openai-chat','openai-responses'])
async def test_truncated_arguments_keep_budget_terminal(m, protocol):
    router = setup(m, protocol)
    if protocol == 'openai-chat':
        frames = [chat({'tool_calls':[{'index':0,'id':'call1','type':'function','function':{'name':'Read','arguments':'{"path":'}}]}), chat({},'length'), b'data: [DONE]\n\n']
    else:
        item = {**fn(), 'arguments':'{"path":', 'status':'incomplete'}
        frames = [terminal([item],kind='incomplete')]
    router.register('https://edge.invalid',lambda req:httpx.Response(200,stream=h.ChunkedByteStream(frames),headers={'content-type':'text/event-stream'}))
    response, client, _ = await h._call_anthropic_core(m,router,{'model':'edge','messages':[{'role':'user','content':'hi'}],'stream':True,'max_tokens':32})
    try:
        assert response.status_code == 200
        events = decoded((await h._consume_streaming_to_string(response)).encode())
        assert not any(e.get('type')=='error' for e in events)
        assert [e['delta']['stop_reason'] for e in events if e.get('type')=='message_delta'] == ['max_tokens']
        assert ''.join(e.get('delta',{}).get('partial_json','') for e in events) == '{"path":'
    finally:
        await client.aclose()


@pytest.mark.parametrize('mapping,expected', [
    (None, 'sessions_list'),
    ({}, 'cc_sess_list'),
    ({'cc_sess_list':'cc_sess_list'}, 'cc_sess_list'),
    ({'Read':'fakeRead01'}, 'cc_sess_list'),
    ({'sessions_list':'cc_sess_list'}, 'sessions_list'),
])
def test_cc_restore_default_vs_explicit_map_across_byte_boundaries(mapping, expected):
    from src.transform import cc_mimicry as cc
    event = {'type':'content_block_start','content_block':{
        'type':'tool_use','name':'cc_sess_list',
        'input':{'type':'tool_use','name':'cc_sess_list'}}}
    text = {'type':'content_block_delta','delta':{'type':'text_delta','text':'cc_sess_list 中文'}}
    raw = frame(**event) + frame(**text)
    wanted = copy.deepcopy(event)
    wanted['content_block']['name'] = expected
    for split in range(len(raw) + 1):
        state = cc.ToolNameRestoreMap(mapping)
        output = cc._restore_tool_names_in_chunk(raw[:split], state)
        output += cc._restore_tool_names_in_chunk(raw[split:], state)
        assert decoded(output) == [wanted, text], split


async def test_cc_mapping_history_choice_and_input_data():
    from src.channel.api_channel import ApiChannel
    names = ['cc_ses_lookup','session_lookup']
    body = {'model':'claude-sonnet-4-6','tools':[{'name':n,'input_schema':{'type':'object'}} for n in names],
            'tool_choice':{'type':'tool','name':'session_lookup'}, 'messages':[
                {'role':'assistant','content':[{'type':'tool_use','id':'t1','name':'session_lookup','input':{'name':'session_lookup'}}]},
                {'role':'user','content':[{'type':'tool_result','tool_use_id':'t1','content':[{'type':'tool_reference','tool_name':'session_lookup'}]}]}]}
    before = copy.deepcopy(body)
    channel = ApiChannel({'name':'edge','baseUrl':'https://edge.invalid','cc_mimicry':True})
    request = await channel.build_upstream_request(body,body['model'])
    sent = json.loads(request.body)
    alias = sent['tools'][1]['name']
    assert alias != sent['tools'][0]['name']
    assert sent['tool_choice']['name'] == alias
    tool = next(b for m in sent['messages'] for b in m['content'] if b['type']=='tool_use')
    assert tool['name']==alias and tool['input']=={'name':'session_lookup'}
    ref = next(b for m in sent['messages'] for b in m['content'] if b['type']=='tool_result')['content'][0]
    assert ref['tool_name']==alias and body==before
    wire = {'type':'message','content':[{'type':'tool_use','id':'t2','name':alias,'input':{'name':alias}}]}
    restored = json.loads(await channel.restore_response(json.dumps(wire).encode(),request.dynamic_tool_map))
    assert restored['content'][0]['name']=='session_lookup'
    assert restored['content'][0]['input']['name']==alias


@pytest.mark.parametrize('ingress', ['chat','anthropic'])
async def test_published_tool_snapshot_conflict_logs_error(m, ingress):
    router = setup(m)
    item = fn()
    frames = [frame('response.output_item.added',output_index=0,item=item),
              frame('response.function_call_arguments.delta',output_index=0,item_id='fc1',delta='{"x":1}'),
              frame('response.output_item.done',output_index=0,item={**item,'arguments':'{"x":1}'}),
              terminal([{**item,'arguments':'{"x":2}'}])]
    router.register('https://edge.invalid',lambda req:httpx.Response(200,stream=h.ChunkedByteStream(frames),headers={'content-type':'text/event-stream'}))
    body={'model':'edge','messages':[{'role':'user','content':'hi'}],'stream':True,'max_tokens':32}
    if ingress=='chat':
        response, client=await h._call_openai_handler(m,router,'chat',body)
    else:
        response, client,_=await h._call_anthropic_core(m,router,body)
    try:
        wire=(await h._consume_streaming_to_string(response)).encode()
        events=decoded(wire)
        assert any('error' in e for e in events)
        assert not any(e.get('type')=='message_stop' or any(c.get('finish_reason') for c in e.get('choices',[])) for e in events)
        assert m['log_db']._get_conn().execute('select status from request_log order by id desc limit 1').fetchone()[0]=='error'
    finally:
        await client.aclose()


@pytest.mark.parametrize('ingress', ['responses','anthropic'])
async def test_chat_empty_stop_done_is_legal_stream(m, ingress):
    router=setup(m,'openai-chat')
    frames=[chat({'role':'assistant'}),chat({},'stop'),b'data: [DONE]\n\n']
    router.register('https://edge.invalid',lambda req:httpx.Response(200,stream=h.ChunkedByteStream(frames),headers={'content-type':'text/event-stream'}))
    if ingress=='responses':
        response,client=await h._call_openai_handler(m,router,'responses',{'model':'edge','input':'hi','stream':True})
    else:
        response,client,_=await h._call_anthropic_core(m,router,{'model':'edge','messages':[{'role':'user','content':'hi'}],'max_tokens':32,'stream':True})
    try:
        assert response.status_code==200, response.body
        events=decoded((await h._consume_streaming_to_string(response)).encode())
        assert not any('error' in e for e in events)
        assert events[-1]['type']==('response.completed' if ingress=='responses' else 'message_stop')
        assert m['log_db']._get_conn().execute('select status from request_log order by id desc limit 1').fetchone()[0]=='success'
    finally:
        await client.aclose()
