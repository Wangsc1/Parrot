"""Full audit protocol fixes: assert client-visible content and terminal truth."""
import json
import pytest
import httpx
from src import upstream
from src.openai.transform import stream_c2r, stream_r2c, stream_chat_to_anthropic as c2a, stream_anthropic_to_chat as a2c, stream_responses_to_anthropic as r2a, stream_anthropic_to_responses as a2r
from src.transports.chat_aggregate import ChatAggregateBuilder
from src.tests import test_protocol_fake_upstreams as h

def _import_modules(): return h._import_modules()
def frame(event_name='', **data):
    if event_name: data.setdefault('type', event_name)
    return ((('event: '+event_name+'\n') if event_name else '')+'data: '+json.dumps(data, ensure_ascii=False)+'\n\n').encode()
def chat(delta, finish=None): return frame(choices=[{'index':0,'delta':delta,'finish_reason':finish}])
def run(tr, frames):
    out=[]
    for f in frames: out.extend(tr.feed(f))
    out.extend(tr.close())
    return b''.join(out)
def decoded(wire):
    _, blocks=upstream.split_sse_events(wire)
    return [d for b in blocks for _,d in [upstream.parse_sse_event_bytes(b)] if isinstance(d,dict)]
def anth_assistant(wire):
    b=upstream.SSEAssistantBuilder(); b.feed(wire); return b.get_assistant()
def chat_assistant(wire):
    b=upstream.ChatSSEAssistantBuilder(); b.feed(wire); return b.get_assistant()
def fn(**kw): return dict(type='function_call',id='fc1',call_id='call1',name='Read',arguments='',**kw)
def terminal(items, kind='completed'):
    resp=dict(id='r1',status=kind,output=items,usage={'input_tokens':3,'output_tokens':2})
    if kind=='incomplete': resp['incomplete_details']={'reason':'max_output_tokens'}
    return frame('response.'+kind,response=resp)

@pytest.mark.parametrize('batch',[False,True])
async def test_http_c2r_error_is_sent(m,batch):
    h._setup(m)
    m['config'].update(lambda cfg:cfg.update({'apiKeys':h._default_key(),'network':{'routing':{'default':'direct'}},'timeouts':{'connect':3,'firstByte':3,'idle':3,'total':10}}))
    base='https://audit-streams.invalid'
    router=h.MockRouter()
    frames=[chat({'content':'partial'}),frame(error={'type':'server_error','code':'server_error','message':'probe upstream failure'})]
    router.register(base,lambda req:httpx.Response(200,stream=h.ChunkedByteStream([b''.join(frames)] if batch else frames),headers={'content-type':'text/event-stream'}))
    h._install_channels(m,[h._make_openai_channel('audit',base,protocol='openai-chat',alias='test-model',real='gpt-real')])
    resp,client=await h._call_openai_handler(m,router,'responses',{'model':'test-model','stream':True,'input':'hi'})
    try:
        assert hasattr(resp,'body_iterator'), (resp.status_code,resp.body)
        wire=await h._consume_streaming_to_string(resp)
        rows=[dict(r) for r in m['log_db']._get_conn().execute('select status from request_log order by id').fetchall()]
        print('C2R_ERROR',batch,'http',resp.status_code,'events',[d.get('type') for d in decoded(wire.encode())],'db_status',rows[-1]['status'])
        assert resp.status_code==200 and 'partial' in wire
        assert 'probe upstream failure' in wire and 'response.failed' in wire
        assert rows[-1]['status']=='error'
    finally: await client.aclose()

@pytest.mark.parametrize('kind',['function','custom'])
def test_fragmented_chat_tool_name(kind):
    def delta(name,args): return {'tool_calls':[{'index':0,'id':'call1','type':kind,kind:{'name':name,('arguments' if kind=='function' else 'input'):args}}]}
    frames=[chat(delta('get_','')),chat(delta('weather','{}')),chat({},'tool_calls'),b'data: [DONE]\n\n']
    cr=stream_c2r.StreamTranslator(model='x'); wire=run(cr,frames)
    output=decoded(wire)[-1]['response']['output'][0]
    b=ChatAggregateBuilder()
    for f in frames:b.feed(f)
    print('FRAGMENTED_NAME',kind,'c2r',output,'aggregate',b.get_assistant())
    assert output['name']=='get_weather'
    if kind=='function':
        ca=c2a.StreamTranslator(model='x'); aw=run(ca,frames)
        assert anth_assistant(aw)['content'][0]['name']=='get_weather'
        assert b.get_assistant()['tool_calls'][0]['function']['name']=='get_weather'

@pytest.mark.parametrize('legacy',[False,True])
def test_chat_aggregate_tool_loss(legacy):
    delta=({'function_call':{'name':'lookup','arguments':'{"q":"x"}'}} if legacy else {'tool_calls':[{'index':0,'id':'call1','type':'custom','custom':{'name':'dsl','input':'RUN x'}}]})
    b=ChatAggregateBuilder()
    b.feed(chat(delta));b.feed(chat({},'function_call' if legacy else 'tool_calls'));b.feed(b'data: [DONE]\n\n')
    result=b.to_full_json(fallback_model='x')
    print('AGGREGATE_TOOL_LOSS',legacy,result['choices'])
    if legacy: assert b.get_assistant()['function_call']=={'name':'lookup','arguments':'{"q":"x"}'} and result['choices'][0]['finish_reason']=='function_call'
    else: assert b.get_assistant()['tool_calls'][0]=={'id':'call1','type':'custom','custom':{'name':'dsl','input':'RUN x'}}

def test_anthropic_initial_content_and_input_lost_in_chat():
    frames=[frame('message_start',message={'id':'a1','role':'assistant','usage':{'input_tokens':3}}),frame('content_block_start',index=0,content_block={'type':'text','text':'前缀'}),frame('content_block_delta',index=0,delta={'type':'text_delta','text':'后缀'}),frame('content_block_stop',index=0),frame('content_block_start',index=1,content_block={'type':'tool_use','id':'call1','name':'lookup','input':{'q':'x'}}),frame('content_block_stop',index=1),frame('message_delta',delta={'stop_reason':'tool_use'},usage={'output_tokens':2}),frame('message_stop')]
    ac=a2c.StreamTranslator(model='x'); got=chat_assistant(run(ac,frames))
    ar=a2r.StreamTranslator(model='x'); other=decoded(run(ar,frames))[-1]['response']['output']
    print('A_INITIAL',got,'a2r',other)
    assert got['content']=='前缀后缀' and json.loads(got['tool_calls'][0]['function']['arguments'])=={'q':'x'}
    assert other[0]['content'][0]['text']=='前缀后缀' and json.loads(other[1]['arguments'])=={'q':'x'}

@pytest.mark.parametrize('partial',[False,True])
def test_r2c_incomplete_snapshot_lost(partial):
    item={'type':'message','id':'msg1','role':'assistant','content':[{'type':'output_text','text':'hello world'}]}
    tr=stream_r2c.StreamTranslator(model='x')
    frames=([frame('response.output_text.delta',output_index=0,item_id='msg1',content_index=0,delta='hello')] if partial else [])+[terminal([item],kind='incomplete')]
    wire=run(tr,frames); got=chat_assistant(wire)
    print('R2C_INCOMPLETE',partial,got)
    assert got['content']=='hello world'
    assert decoded(wire)[-1]['choices'][0]['finish_reason']=='length'

def test_r2a_sparse_buffered_arguments_differ_from_getter():
    tr=r2a.StreamTranslator(model='x',optional_empty_string_fields_by_tool={'Read':{'pages'}})
    args='{"file_path":"a.txt","pages":""}'
    frames=[frame('response.output_item.added',output_index=0,item=fn()),frame('response.function_call_arguments.delta',output_index=0,item_id='fc1',delta=args),terminal([])]
    wire=run(tr,frames); client=anth_assistant(wire); stored=tr.get_downstream_anthropic_assistant()
    print('R2A_BUFFERED',client,stored)
    assert client==stored and client['content'][0]['input']=={'file_path':'a.txt'}

def test_r2a_function_done_only_arguments_dropped():
    tr=r2a.StreamTranslator(model='x')
    frames=[frame('response.output_item.added',output_index=0,item=fn()),frame('response.function_call_arguments.done',output_index=0,item_id='fc1',name='Read',arguments='{"file_path":"a.txt"}'),terminal([])]
    got=anth_assistant(run(tr,frames))
    print('R2A_ARGS_DONE',got)
    assert got['content'][0]['input']=={'file_path':'a.txt'}

def test_r2a_late_tool_metadata_client_differs():
    tr=r2a.StreamTranslator(model='x')
    item=fn();initial={**item,'name':'','call_id':''}
    frames=[frame('response.output_item.added',output_index=0,item=initial),frame('response.output_item.done',output_index=0,item={**item,'arguments':'{}'}),terminal([{**item,'arguments':'{}'}])]
    client=anth_assistant(run(tr,frames)); stored=tr.get_downstream_anthropic_assistant()
    print('R2A_METADATA',client,stored)
    assert client==stored and client['content'][0]['name']=='Read'
    assert client['content'][0]['id']=='call1'

@pytest.mark.parametrize('translator',[stream_r2c,r2a])
def test_conflicting_snapshot_silently_success(translator):
    item=fn();frames=[frame('response.output_item.added',output_index=0,item=item),frame('response.function_call_arguments.delta',output_index=0,item_id='fc1',delta='{"x":1}'),frame('response.output_item.done',output_index=0,item={**item,'arguments':'{"x":2}'}),terminal([{**item,'arguments':'{"x":2}'}])]
    tr=translator.StreamTranslator(model='x');wire=run(tr,frames)
    b=upstream.ResponsesSSEAssistantBuilder()
    for f in frames:b.feed(f)
    print('CONFLICT',translator.__name__,wire.decode(),b.get_output_items())
    assert b.get_output_items()[0]['arguments']=='{"x":2}'
    if translator is stream_r2c:
        assert any('error' in d for d in decoded(wire))
        assert not any(c.get('finish_reason') for d in decoded(wire) for c in d.get('choices',[]))
    else:
        assert anth_assistant(wire)['content'][0]['input']=={'x':2}
        assert not any(d.get('type')=='error' for d in decoded(wire))

def test_anthropic_context_window_stop_claimed_completed():
    tr=a2r.StreamTranslator(model='x')
    wire=run(tr,[frame('message_start',message={'id':'a1'}),frame('message_delta',delta={'stop_reason':'model_context_window_exceeded'},usage={'output_tokens':2}),frame('message_stop')])
    result=decoded(wire)[-1]
    print('A2R_CONTEXT_STOP',result)
    assert result['type']=='response.incomplete' and result['response']['incomplete_details']=={'reason':'max_output_tokens'}

"""Additional bounded probes of stream/aggregation state fidelity."""
import time
import json
import pytest
import httpx
from src import upstream
from src.openai.transform import common
from src.openai.transform import stream_r2c, stream_responses_to_anthropic as r2a
from src.transports.chat_aggregate import ChatAggregateBuilder
from src.tests import test_protocol_fake_upstreams as h
from src.tests.test_stream_as_non_stream_errors import _Ctx, _Channel

def _import_modules(): return h._import_modules()

async def test_http_c2a_conversion_error_logged_error(m):
    h._setup(m)
    m['config'].update(lambda cfg:cfg.update({'apiKeys':h._default_key(),'network':{'routing':{'default':'direct'}},'timeouts':{'connect':3,'firstByte':3,'idle':3,'total':10}}))
    base='https://audit-c2a.invalid'; router=h.MockRouter()
    frames=[chat({'content':'partial'}),chat({'tool_calls':[{'index':0,'id':'call1','type':'function','function':{'arguments':'{}'}}]}),chat({},'tool_calls'),b'data: [DONE]\n\n']
    router.register(base,lambda req:httpx.Response(200,stream=h.ChunkedByteStream(frames),headers={'content-type':'text/event-stream'}))
    h._install_channels(m,[h._make_openai_channel('audit',base,protocol='openai-chat',alias='test-model',real='gpt-real')])
    body={'model':'test-model','stream':True,'max_tokens':32,'messages':[{'role':'user','content':'hi'}], 'tools':[{'name':n,'input_schema':{'type':'object','properties':{}}} for n in ['one','two']]}
    resp,client,_=await h._call_anthropic_core(m,router,body)
    try:
        assert hasattr(resp,'body_iterator'),(resp.status_code,resp.body)
        wire=await h._consume_streaming_to_string(resp)
        status=m['log_db']._get_conn().execute('select status from request_log order by id desc limit 1').fetchone()[0]
        print('C2A_CLOSE_ERROR',wire,'db_status',status)
        assert 'Upstream tool call ended without a tool name' in wire
        assert status=='error'
    finally: await client.aclose()

async def test_chat_aggregate_choices_cross_contaminated():
    from src.transports.http_runtime import aggregate_stream_as_non_stream_response
    class Channel(_Channel): protocol='openai-chat'
    frames=[frame(id='chat1',choices=[{'index':0,'delta':{'content':'A'},'logprobs':{'content':[{'token':'A'}]}}]),frame(choices=[{'index':1,'delta':{'content':'B'},'logprobs':{'content':[{'token':'B'}]}}]),frame(choices=[{'index':0,'delta':{},'finish_reason':'stop'},{'index':1,'delta':{},'finish_reason':'length'}]),b'data: [DONE]\n\n']
    resp=httpx.Response(200,stream=h.ChunkedByteStream(frames),headers={'content-type':'text/event-stream'})
    start=time.time()
    result=await aggregate_stream_as_non_stream_response(_Ctx(),resp,Channel(),'x',dynamic_map=None,connect_ms=1,start_time=start,deadline_ts=start+30,total_timeout=30,first_byte_timeout=5,idle_timeout=5,translator_ctx=None)
    print('MULTICHOICE',result.obj)
    assert result.error is None
    assert [(c['index'],c['message']['content'],c['finish_reason']) for c in result.obj['choices']]==[(0,'A','stop'),(1,'B','length')]
    assert [[p['token'] for p in c['logprobs']['content']] for c in result.obj['choices']]==[['A'],['B']]

@pytest.mark.parametrize('proto',['responses','anthropic'])
@pytest.mark.parametrize('batch',[False,True])
async def test_http_terminal_latch_depends_on_packet(m,proto,batch):
    h._setup(m)
    m['config'].update(lambda cfg:cfg.update({'apiKeys':h._default_key(),'network':{'routing':{'default':'direct'}},'timeouts':{'connect':3,'firstByte':3,'idle':3,'total':10}}))
    base='https://audit-terminal.invalid'; router=h.MockRouter()
    if proto=='responses':
        frames=[frame('response.output_text.delta',output_index=0,content_index=0,delta='OK'),terminal([]),frame('error',message='after-terminal',code='server_error')]
        ch=h._make_openai_channel('audit',base,protocol='openai-responses',alias='test-model',real='gpt-real')
    else:
        payload=h._anthropic_sse_response('OK').content
        frames=[payload,frame('error',error={'type':'api_error','message':'after-terminal'})]
        ch=h._make_anthropic_channel(m,'audit',base,alias='test-model',real='claude-real')
    router.register(base,lambda req:httpx.Response(200,stream=h.ChunkedByteStream([b''.join(frames)] if batch else frames),headers={'content-type':'text/event-stream'}))
    h._install_channels(m,[ch])
    if proto=='responses': resp,client=await h._call_openai_handler(m,router,'responses',{'model':'test-model','stream':True,'input':'hi'})
    else: resp,client,_=await h._call_anthropic_core(m,router,{'model':'test-model','stream':True,'max_tokens':32,'messages':[{'role':'user','content':'hi'}]})
    try:
        wire=await h._consume_streaming_to_string(resp)
        status=m['log_db']._get_conn().execute('select status from request_log order by id desc limit 1').fetchone()[0]
        print('TERMINAL_PACKET',proto,batch,'after-terminal' in wire,status)
        assert status=='success'
        assert 'after-terminal' not in wire
    finally: await client.aclose()

@pytest.mark.parametrize('kind',['function_call','custom_tool_call'])
def test_r2c_incomplete_tool_snapshot_lost(kind):
    item=(fn() if kind=='function_call' else dict(type=kind,id='ct1',call_id='call1',name='dsl',input='RUN x'))
    if kind=='function_call': item['arguments']='{"file_path":"a.txt"}'
    tr=stream_r2c.StreamTranslator(model='x'); wire=run(tr,[terminal([item],kind='incomplete')])
    print('R2C_INCOMPLETE_TOOL',kind,chat_assistant(wire))
    assert chat_assistant(wire)['tool_calls'][0]['id']=='call1'
    assert decoded(wire)[-1]['choices'][0]['finish_reason']=='length'

@pytest.mark.parametrize('snapshot',[True,False])
def test_r2c_reasoning_snapshot_or_getter_loss(monkeypatch,snapshot):
    monkeypatch.setattr(common,'reasoning_passthrough_enabled',lambda:True)
    item=dict(type='reasoning',id='rs1',summary=[{'type':'summary_text','text':'think'}])
    tr=stream_r2c.StreamTranslator(model='x')
    frames=([frame('response.reasoning_summary_text.delta',output_index=0,item_id='rs1',summary_index=0,delta='think')] if not snapshot else [])+[terminal([item])]
    client=chat_assistant(run(tr,frames));stored=tr.get_downstream_chat_assistant()
    print('REASONING_LOSS',snapshot,client,stored)
    assert client['reasoning_content']=='think' and stored['reasoning_content']=='think'

@pytest.mark.parametrize('kind',['custom','reasoning'])
def test_responses_builder_drops_sparse_nonfunction_delta(kind):
    b=upstream.ResponsesSSEAssistantBuilder()
    if kind=='custom':
        item=dict(type='custom_tool_call',id='ct1',call_id='call1',name='dsl',input='')
        frames=[frame('response.output_item.added',output_index=0,item=item),frame('response.custom_tool_call_input.delta',output_index=0,item_id='ct1',delta='RUN x'),frame('response.custom_tool_call_input.done',output_index=0,item_id='ct1',input='RUN x'),terminal([])]
    else:
        item=dict(type='reasoning',id='rs1',summary=[])
        frames=[frame('response.output_item.added',output_index=0,item=item),frame('response.reasoning_summary_text.delta',output_index=0,item_id='rs1',summary_index=0,delta='think'),frame('response.reasoning_summary_text.done',output_index=0,item_id='rs1',summary_index=0,text='think'),terminal([])]
    for f in frames:b.feed(f)
    print('RESP_BUILDER',kind,b.get_output_items())
    assert b.get_output_items()==[{**item, **({'input':'RUN x'} if kind=='custom' else {'summary':[{'type':'summary_text','text':'think'}]})}]

import json
import pytest
import httpx
from src.openai.transform import tool_arguments
from src.tests import test_protocol_fake_upstreams as h
from src.tests.test_protocol_audit_tool_argument_runtime import response_json, response_sse

def _import_modules(): return h._import_modules()

@pytest.mark.parametrize('protocol',['openai-chat','openai-responses'])
@pytest.mark.parametrize('raw',['```json\n{"path":"a",}\n```','{"path":'])
async def test_completed_stream_forwards_invalid_anthropic_tool_json(m,protocol,raw):
    h._setup(m)
    m['config'].update(lambda cfg:cfg.update({'apiKeys':h._default_key(),'network':{'routing':{'default':'direct'}},'timeouts':{'connect':3,'firstByte':3,'idle':3,'total':10}}))
    base='https://audit-json.invalid';router=h.MockRouter()
    wire=response_sse(protocol,response_json(protocol,raw))
    router.register(base,lambda req:httpx.Response(200,stream=h.ChunkedByteStream([wire]),headers={'content-type':'text/event-stream'}))
    h._install_channels(m,[h._make_openai_channel('audit',base,protocol=protocol,alias='test-model',real='gpt-real')])
    body={'model':'test-model','stream':True,'max_tokens':32,'messages':[{'role':'user','content':'hi'}],'tools':[{'name':'Read','input_schema':{'type':'object','properties':{'path':{'type':'string'}}}}]}
    resp,client,_=await h._call_anthropic_core(m,router,body)
    try:
        output=await h._consume_streaming_to_string(resp);events=decoded(output.encode())
        args=''.join(e.get('delta',{}).get('partial_json','') for e in events)
        status=m['log_db']._get_conn().execute('select status from request_log order by id desc limit 1').fetchone()[0]
        print('INVALID_JSON_STREAM',protocol,repr(raw),'args',repr(args),'stop',[e.get('delta',{}).get('stop_reason') for e in events if e.get('type')=='message_delta'],'status',status,'aggregate',anth_assistant(output.encode()))
        if raw.startswith('```'):
            assert json.loads(args)=={'path':'a'} and status=='success'
            assert events[-1]['type']=='message_stop' and events[-2]['delta']['stop_reason']=='tool_use'
        else:
            assert status=='error'
            assert any(e.get('type')=='error' for e in events)
            assert not any(e.get('type')=='message_stop' for e in events)
    finally:await client.aclose()
