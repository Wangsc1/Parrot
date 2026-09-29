"""S01-S16 regression tests: real owners/parsers with synthetic upstream IO."""
import asyncio
import json
import threading
import hashlib
from types import SimpleNamespace
import pytest
from src.tests import test_openai_responses_ws as legacy
from src.tests.test_openai_responses_ws import _isolate_ws_config
from src.tests.test_protocol_audit_ws_regressions import runtime, Client, create, terminal, install_native, next_dispatch, finish
from src.tests.test_protocol_audit_cursor_regressions import local_client, pending, request, HISTORY
from src.tests.test_cursor_proxy_lifecycle import ScriptedH2Stream
from src.openai import responses_ws as ws
from src.cursor_bridge.client import CursorClient
from src.cursor_bridge.runtime import CursorBridgeRuntime
from src.cursor_bridge.session import SessionEvent
from src.cursor_bridge.h2stream import CursorH2Stream, cursor_headers

@pytest.mark.asyncio
async def test_sessions_cancel_during_first_translation_releases_key_lease(monkeypatch, runtime):
    cfg = runtime['config']._cache
    cfg['apiKeyConcurrency'] = {'enabled': True, 'defaultMaxConcurrent': 1, 'defaultMaxQueue': 0, 'defaultQueueWaitSeconds': 0}
    legacy._make_channel(runtime)
    entered = asyncio.Event()
    leases = []
    original = runtime['apikey_limiter'].acquire
    async def acquire(*args, **kw):
        lease = await original(*args, **kw)
        leases.append(lease)
        return lease
    async def translate(*args, **kw):
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(runtime['apikey_limiter'], 'acquire', acquire)
    monkeypatch.setattr(ws.translation, 'translate_body', translate)
    client = Client()
    task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create('lane'))
        await asyncio.wait_for(entered.wait(), 2)
        client.send({'type': 'response.cancel', 'stream_id': 'lane'})
        await client.until(lambda e: e.get('error', {}).get('code') == 'response_cancelled')
        assert not client.close_calls  # new cancel semantics remain correct
        await finish(client, task)
        assert runtime['apikey_limiter'].key_snapshot('ws-key')['in_flight'] == 0
    finally:
        if not task.done():
            client.disconnect()
            await asyncio.wait_for(task, 3)
        for lease in leases:
            await lease.release()  # only isolated test teardown, never production state

@pytest.mark.asyncio
async def test_sessions_cancel_while_flushing_first_visible_joins_control_reader(monkeypatch, runtime):
    dispatched, _ = await install_native(monkeypatch, runtime)
    entered = asyncio.Event()
    class SlowClient(Client):
        async def send_text(self, text):
            if json.loads(text).get('type') == 'response.output_text.delta':
                entered.set()
                await asyncio.Event().wait()
            await super().send_text(text)
    client = SlowClient()
    task = asyncio.create_task(ws.handle_responses_ws(client))
    leaked = []
    try:
        client.send(create('lane'))
        up, _ = await next_dispatch(dispatched)
        up.feed({'type': 'response.output_text.delta', 'delta': 'hello'})
        await asyncio.wait_for(entered.wait(), 2)
        client.send({'type': 'response.cancel', 'stream_id': 'lane'})
        await client.until(lambda e: e.get('error', {}).get('code') == 'response_cancelled')
        await finish(client, task)
        leaked = [t for t in asyncio.all_tasks() if not t.done() and '_relay_ws_session.<locals>.downstream_to_upstream' in t.get_coro().__qualname__]
        assert not leaked
        assert up.closed
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for t in leaked:
            t.cancel()
        await asyncio.gather(*leaked, return_exceptions=True)

@pytest.mark.asyncio
async def test_sessions_reconnect_restores_full_history_without_native_parent(monkeypatch, runtime):
    from src import search_tool_policy
    search_tool_policy._REPLAY.clear()
    dispatched, _ = await install_native(monkeypatch, runtime)
    c1 = Client(); t1 = asyncio.create_task(ws.handle_responses_ws(c1))
    c2 = Client(); t2 = None
    try:
        c1.send(create(text='ORIGINAL'))
        up1, first = await next_dispatch(dispatched)
        up1.feed(terminal('origin'))
        await c1.until(lambda e: e.get('type') == 'response.completed')
        # give the session owner time to cache its successful response
        for _ in range(200):
            if search_tool_policy._REPLAY:
                break
            await asyncio.sleep(.005)
        assert search_tool_policy._REPLAY
        await finish(c1, t1)
        t2 = asyncio.create_task(ws.handle_responses_ws(c2))
        c2.send(create(text='CONTINUE', previous_response_id='resp_origin'))
        up2, wire = await next_dispatch(dispatched)
        assert 'previous_response_id' not in wire
        assert 'ORIGINAL' in json.dumps(wire['input'])
        assert 'CONTINUE' in json.dumps(wire['input'])
        assert any(i.get('role') == 'assistant' for i in wire['input'])
    finally:
        if not t1.done():
            await finish(c1, t1)
        if t2 and not t2.done():
            await finish(c2, t2)
        search_tool_policy._REPLAY.clear()

@pytest.mark.asyncio
async def test_sessions_generate_false_preserved_before_real_http_bridge(monkeypatch, runtime):
    ch = legacy._make_channel(runtime)
    ch.responses_ws_upstream_transport = 'sse'
    observed = []
    async def open_fake(**kwargs):
        observed.append(json.loads(kwargs['upstream_req'].body))
        class Response:
            status_code = 200
            headers = {'content-type': 'text/event-stream'}
            async def aiter_bytes(self):
                yield ('data: ' + json.dumps(terminal('GENERATED')) + '\n\n').encode()
        class Context:
            async def __aexit__(self, *args):
                pass
        return SimpleNamespace(error=None, response=Response(), connect_ms=1, timing=None,
            proxy_name=None, proxy_bytes={'up': 1, 'down': 1}, proxy_client=None,
            proxy_attempt_id=None, round_timeouts=None, ctx=Context())
    monkeypatch.setattr(ws, 'open_response_with_proxy_chain', open_fake)
    client = Client(); task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create('warmup', generate=False))
        error = await client.until(lambda e: e.get('type') == 'error')
        assert 'generate:false' in error['error']['message']
        assert error['error']['param'] == 'generate'
        assert not observed
    finally:
        await finish(client, task)


def test_sessions_cursor_tool_id_cannot_cross_principal():
    rt = CursorBridgeRuntime()
    principal_a = hashlib.sha256(b'key-a:session-a').hexdigest()
    principal_b = hashlib.sha256(b'key-b:session-b').hexdigest()
    body_a = {'messages': HISTORY[:4]}
    a = rt.session_for('shared-account', body_a, 'anchor', principal=principal_a)
    rt.register_tool_call('shared-account', a, 'c1', principal=principal_a)
    b = rt.session_for('shared-account', {'messages': [{'role':'tool','tool_call_id':'c1','content':'injected'}]}, 'anchor', principal=principal_b)
    assert b != a
    assert rt.session_for('shared-account', {'messages': HISTORY[:6]}, principal=principal_a) == a
    assert rt.session_for('shared-account', body_a, principal_b) != a


def test_sessions_cursor_resume_honors_changed_model_and_tools():
    events = [SessionEvent(type='toolCall', exec=pending()), SessionEvent(type='batchReady'),
              SessionEvent(type='text', text='old-model-result'), SessionEvent(type='done')]
    client, sessions = local_client(events, [SessionEvent(type='text', text='new-model-result'), SessionEvent(type='done')])
    try:
        client.chat_completions(model='model-A', messages=HISTORY[:4], session_id='stable', stream=False, tools=[{'type':'function','function':{'name':'Read','parameters':{'type':'object','properties':{'path':{'type':'string'}}}}}])
        result = client.chat_completions(model='model-B', messages=HISTORY[:6], session_id='stable', stream=False, tool_choice='none')
        assert len(sessions) == 2
        assert not sessions[0].alive
        assert sessions[1].kwargs['model'] == 'model-B'
        assert b'TOOL_RESULT_UNIQUE' in sessions[1].kwargs['request_bytes']
        assert result['model'] == 'model-B'
        assert result['choices'][0]['message']['content'] == 'new-model-result'
        assert not sessions[0].results
    finally:
        client.close()


def test_sessions_cursor_same_anchor_second_request_does_not_close_first():
    client, sessions = local_client()
    try:
        first = client.chat_completions(model='model-A', messages=[{'role':'user','content':'one'}], session_id='shared')
        assert sessions[0].alive
        second = client.chat_completions(model='model-A', messages=[{'role':'user','content':'two'}], session_id='shared')
        assert sessions[0].alive
        assert sessions[1].alive
        first.close()
        assert sessions[1].alive
        second.close()
    finally:
        client.close()


def test_sessions_cursor_close_during_open_closes_late_routed_socket(monkeypatch):
    from src import network
    entered = threading.Event(); resume = threading.Event()
    routed = ScriptedH2Stream(mode='stall')
    def open_route(*args, **kw):
        entered.set()
        assert resume.wait(2)
        return routed
    monkeypatch.setattr(network, 'open_sync_stream', open_route)
    stream = CursorH2Stream(timeout_s=.1)
    errors = []
    def run():
        try:
            stream.open(cursor_headers(path='/test.Run', access_token='fake', content_type='application/connect+proto'), b'hello')
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(2)
        stream.close()
        resume.set(); worker.join(2)
        assert not worker.is_alive()
        assert errors == []
        stream.close()
        assert stream._closed.is_set()
        assert routed.closed
        assert stream._sock is None
    finally:
        resume.set(); worker.join(2)
        routed.close()


def test_sessions_cursor_abandoned_tool_sessions_are_reaped():
    rt = CursorBridgeRuntime()
    client, sessions = local_client([SessionEvent(type='toolCall', exec=pending()), SessionEvent(type='batchReady')])
    rt._clients['account'] = client
    try:
        client.chat_completions(model='M', messages=HISTORY[:4], session_id='paused', stream=False)
        rt.register_tool_call('account', 'paused', 'c1')
        sessions[0].close()  # native timeout eventually closes the transport only
        rt.reap_sessions()
        assert 'paused' not in client._conversations
        assert not rt._tool_sessions
        assert rt.session_for('account', {'messages': HISTORY[:6]}) != 'paused'
    finally:
        rt.finish_session('account', 'paused')

@pytest.mark.asyncio
async def test_sessions_ws_admitted_create_backlog_bounded(monkeypatch, runtime):
    # The physical socket reader drains into creates, independently of key admission.
    conn = ws._ResponsesWsConnection(Client())
    async def blocked_handler(lane):
        await asyncio.Event().wait()
    monkeypatch.setattr(ws, '_handle_responses_ws_lane', blocked_handler)
    try:
        for n in range(1000):
            await conn.dispatch(json.dumps(create('same-lane', 'x'*1024)))
        assert len(conn.lanes) == 1
        assert conn.lanes['same-lane'].creates.qsize() == 32
        assert conn.lanes['same-lane'].creates.maxsize == 32
        assert conn.queued_bytes <= ws._WS_CONNECTION_QUEUE_BYTES
        assert any(e.get('error', {}).get('code') == 'websocket_queue_full' for e in conn.websocket.events)
        assert runtime['apikey_limiter'].key_snapshot('ws-key')['waiting'] == 0
    finally:
        conn.closed = True
        for lane in conn.lanes.values():
            lane.task.cancel()
        await asyncio.gather(*(lane.task for lane in conn.lanes.values()), return_exceptions=True)


def test_sessions_cursor_mcp_filePath_argument_matches_schema():
    from src.cursor_bridge import agent_pb2
    from src.cursor_bridge.request_builder import json_to_value_bytes
    from src.cursor_bridge.tool_dispatch import handle_exec_message
    emitted = []
    msg = agent_pb2.ExecServerMessage(id=1, exec_id='exec', mcp_args=agent_pb2.McpArgs(
        tool_call_id='call', tool_name='mcp_pi_open_file', args={'filePath':json_to_value_bytes('/fake/project/file')}))
    handle_exec_message(msg, mcp_tools=[], enabled={'open_file'}, cloud_rule=None,
                        send=lambda _:None, on_mcp_exec=emitted.append)
    assert json.loads(emitted[0].decoded_args) == {'filePath':'/fake/project/file'}

@pytest.mark.asyncio
async def test_sessions_completed_response_processed_ack_is_forwarded(monkeypatch, runtime):
    dispatched, _ = await install_native(monkeypatch, runtime)
    client = Client(); task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create('lane'))
        up, _ = await next_dispatch(dispatched)
        up.feed(terminal('done'))
        await client.until(lambda e: e.get('type') == 'response.completed')
        await asyncio.sleep(.02)  # wait for receive_create() to release active slot
        client.send({'type':'response.processed','stream_id':'lane','response_id':'resp_done'})
        for _ in range(100):
            if any(f.get('type') == 'response.processed' for f in up.sent):
                break
            await asyncio.sleep(.005)
        assert any(f.get('type') == 'response.processed' for f in up.sent)
        assert not any(e.get('type') == 'error' for e in client.events)
    finally:
        await finish(client, task)


def test_sessions_cursor_thinking_tags_split_across_deltas_are_parsed():
    from src.cursor_bridge.thinking import ThinkingTagFilter
    split = ThinkingTagFilter()
    parts = [split.process(piece) for piece in ['<think', 'ing>reason</think', 'ing>answer']]
    assert ''.join(p.content for p in parts) == 'answer'
    assert ''.join(p.reasoning for p in parts) == 'reason'
    together = ThinkingTagFilter().process('<thinking>reason</thinking>answer')
    assert together.reasoning == 'reason' and together.content == 'answer'


def test_sessions_cursor_tool_pause_uses_flushed_timeout(monkeypatch):
    from queue import Queue
    from src.cursor_bridge import agent_pb2
    from src.cursor_bridge.session import CursorSession, StreamState
    from src.cursor_bridge.constants import INACTIVITY_FLUSHED_S, INACTIVITY_STREAMING_S
    import src.cursor_bridge.session as session_module
    monkeypatch.setattr(session_module.time, 'monotonic', lambda: 1000.0)
    s = object.__new__(CursorSession)
    s.state = StreamState()
    s.events = Queue()
    s.pending_execs = [pending()]
    s.batch_state = 'collecting'; s._flushed = []
    s._timer_phase = 'streaming'; s.done_sent = False
    s.on_checkpoint = None
    msg = agent_pb2.AgentServerMessage(conversation_checkpoint_update=agent_pb2.ConversationStateStructure())
    s._on_frame(msg.SerializeToString())
    s._after_parse()
    assert s.batch_state == 'flushed'
    assert s.events.get_nowait().type == 'batchReady'
    assert s._inactivity_deadline == 1000.0 + INACTIVITY_FLUSHED_S


import hashlib
import http.client
import json
import threading
import time
import pytest
from src.cursor_bridge.client import CursorClient
from src.cursor_bridge.runtime import CursorBridgeRuntime, _ACCOUNT_HEADER, _SESSION_HEADER, _PRINCIPAL_HEADER
from src.cursor_bridge.session import SessionEvent
from src.tests.test_protocol_audit_cursor_regressions import HISTORY, local_client, pending


def post(rt, body, principal_anchor):
    from urllib.parse import urlsplit
    u = urlsplit(rt.base_url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=2)
    try:
        conn.request('POST', '/v1/chat/completions', json.dumps(body), headers={
            'Authorization': 'Bearer ' + rt.bearer_secret,
            'Content-Type': 'application/json', _ACCOUNT_HEADER:'shared-account',
            _SESSION_HEADER: hashlib.sha256(principal_anchor.encode()).hexdigest(),
            _PRINCIPAL_HEADER: principal_anchor.split(':')[0],
        })
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def test_sessions_real_http_bridge_principal_isolation_and_same_principal_resume():
    rt = CursorBridgeRuntime()
    client, sessions = local_client([
        SessionEvent(type='toolCall', exec=pending('c1')), SessionEvent(type='batchReady'),
        SessionEvent(type='text', text='ONLY_IN_KEY_A_CONTEXT'), SessionEvent(type='done')],
        [SessionEvent(type='text', text='KEY_B_NEW'), SessionEvent(type='done')])
    rt._clients['shared-account'] = client
    rt.ensure_started()
    try:
        status, first = post(rt, {'model':'M','stream':False,'messages':HISTORY[:4],'tools':[{'type':'function','function':{'name':'Read','parameters':{'type':'object'}}}]}, 'key-A:anchor-A')
        assert status == 200
        assert first['choices'][0]['message']['tool_calls'][0]['id'] == 'c1'
        status, second = post(rt, {'model':'M','stream':False,'messages':[
            {'role':'tool','tool_call_id':'c1','content':'KEY_B_INJECTION'}]}, 'key-B:anchor-B')
        assert status == 200
        assert second['choices'][0]['message']['content'] == 'KEY_B_NEW'
        assert len(sessions) == 2
        assert not sessions[0].results
        assert sessions[0].alive
        status, third = post(rt, {'model':'M','stream':False,'messages':HISTORY[:6], 'tools':[{'type':'function','function':{'name':'Read','parameters':{'type':'object'}}}]}, 'key-A:anchor-A')
        assert status == 200
        assert len(sessions) == 2
        assert third['choices'][0]['message']['content'] == 'ONLY_IN_KEY_A_CONTEXT'
        assert len(sessions[0].results) == 1
    finally:
        rt.stop()


def test_sessions_real_http_bridge_disconnect_cancels_while_silent():
    rt = CursorBridgeRuntime()
    entered = threading.Event(); release = threading.Event(); sessions = []
    class SilentSession:
        def __init__(self, **kw):
            self.alive=True; self.pending_execs=[]; self.cancelled=False
            sessions.append(self)
        def next(self, timeout=None):
            entered.set()
            assert release.wait(3)
            return SessionEvent(type='done')
        def close(self): self.alive=False
        def cancel(self): self.cancelled=True; self.close(); release.set()
    rt._clients['shared-account'] = CursorClient('fake', session_factory=SilentSession, request_timeout_s=2)
    rt.ensure_started()
    from urllib.parse import urlsplit
    u=urlsplit(rt.base_url)
    conn=http.client.HTTPConnection(u.hostname, u.port, timeout=2)
    resp=None
    try:
        conn.request('POST','/v1/chat/completions',json.dumps({'model':'M','stream':True,'messages':[{'role':'user','content':'hello'}]}),headers={
            'Authorization':'Bearer '+rt.bearer_secret,'Content-Type':'application/json',_ACCOUNT_HEADER:'shared-account'})
        resp=conn.getresponse()
        assert resp.status==200
        assert entered.wait(2)
        resp.close(); conn.close()
        time.sleep(.2)
        assert not sessions[0].alive
        assert sessions[0].cancelled
    finally:
        release.set()
        if resp: resp.close()
        conn.close(); rt.stop()


@pytest.mark.parametrize('bad', ['{', '{}', 'null', '', None])
@pytest.mark.parametrize('field', ['input_items', 'output_items'])
def test_sessions_s15_corrupt_store_is_explicit(bad, field):
    from src.openai import store
    row = dict(response_id='r', api_key_name='k', parent_id=None, model='m',
               channel_key='c', created_at=0, expires_at=9999999999,
               input_items='[]', output_items='[{"type":"message","content":[]}]')
    row[field] = bad
    with pytest.raises(store.ResponseHistoryError, match='corrupt response history'):
        store._row_to_response(row, 'r', 'k')


@pytest.mark.parametrize('envelope', ['root', 'response', 'data', 'data.response'])
def test_sessions_ao06_direct_ws_usage_envelopes(envelope):
    event = {'type':'response.completed', 'response': {'id':'r','status':'completed','output':[]}}
    target = event
    for field in ([] if envelope == 'root' else envelope.split('.')):
        target = target.setdefault(field, {})
    target['usage'] = {'input_tokens':100, 'output_tokens':10, 'input_tokens_details':{'cached_tokens':20}}
    tracker = ws._WsTracker()
    tracker.feed_text(json.dumps(event))
    assert tracker.response_completed and tracker.usage_observed
    assert tracker.usage == {'input_tokens':80,'output_tokens':10,'cache_creation':0,'cache_read':20}


def test_sessions_s01_duplicate_ids_never_overwrite_another_conversation():
    from src.cursor_bridge.errors import CursorError
    rt = CursorBridgeRuntime()
    a = rt.session_for('acc', {}, 'a', principal='key')
    b = rt.session_for('acc', {}, 'b', principal='key')
    for sid in (a, b):
        rt.register_tool_call('acc', sid, 'same', principal='key')
    body = {'messages':[{'role':'tool', 'tool_call_id':'same','content':'x'}]}
    assert rt.session_for('acc', body, 'a', principal='key') == a
    assert rt.session_for('acc', body, 'b', principal='key') == b
    with pytest.raises(CursorError, match='Ambiguous'):
        rt.session_for('acc', body, principal='key')
    rt.finish_session('acc', a)
    assert rt.session_for('acc', body, principal='key') == b
    assert rt.session_for('acc', body, principal='other-key') != b


@pytest.mark.asyncio
async def test_sessions_s03_byte_budget_and_full_control_queue_still_allow_cancel(monkeypatch, runtime):
    conn = ws._ResponsesWsConnection(Client())
    entered = asyncio.Event()
    async def blocked_handler(lane):
        lane.owns_slot = True
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(ws, '_handle_responses_ws_lane', blocked_handler)
    try:
        await conn.dispatch(json.dumps(create('busy')))
        await asyncio.wait_for(entered.wait(), 1)
        lane = conn.lanes['busy']
        await conn.dispatch(json.dumps(create('busy', 'x'*(4*1024*1024))))
        assert lane.creates.qsize() == 1
        for i in range(100):
            await conn.dispatch(json.dumps({'type':'response.custom','stream_id':'busy','i':i}))
        assert lane.controls.qsize() == 64
        assert not lane.terminal_sent
        session = lane.session_task
        await conn.dispatch(json.dumps({'type':'response.cancel','stream_id':'busy'}))
        assert lane.cancel_requested
        await asyncio.gather(session, return_exceptions=True)
        assert session.cancelled()
    finally:
        conn.closed = True
        for lane in conn.lanes.values():
            lane.task.cancel()
        await asyncio.gather(*(lane.task for lane in conn.lanes.values()), return_exceptions=True)
        assert conn.queued_bytes == 0


def test_sessions_s05_sync_authoritative_restored_tools_and_removed_parent():
    from src.openai.responses_ws_runtime import sync_translated_body_to_ws_create
    obj = {'type':'response.create','previous_response_id':'stale','generate':False,
           'stream_id':'lane','client_metadata':{'client':'kept'},'input':'delta'}
    body = {'model':'m','input':[{'role':'user','content':'full'}],
            'tools':[{'type':'function','name':'Read','parameters':{'type':'object'}}],
            'generate':False, '_private':'never-forward'}
    sync_translated_body_to_ws_create(obj, body)
    assert obj['generate'] is False and obj['input'] == body['input']
    assert obj['tools'] == body['tools']
    assert obj['client_metadata'] == {'client':'kept'}
    assert 'previous_response_id' not in obj and '_private' not in obj


@pytest.mark.asyncio
async def test_sessions_s06_search_handoff_warmup_never_invokes_http(monkeypatch, runtime):
    captured = []
    async def native(websocket, **kw):
        captured.append(kw)
        return True
    monkeypatch.setattr(ws, '_run_ws_failover', native)
    lease = await runtime['apikey_limiter'].acquire('ws-key', None)
    await ws._run_search_ws_session(Client(), body={'model':'test-model','generate':False},
        schedule_result=object(), request_id='test-warmup', api_key_name='ws-key',
        client_ip='local', start_time=0, start_monotonic=0, allowed_models=None, api_key_lease=lease)
    assert captured[0]['first_obj']['generate'] is False
    assert runtime['apikey_limiter'].key_snapshot('ws-key')['in_flight'] == 0


def test_sessions_s08_runtime_owner_cannot_finish_a_replacement():
    rt = CursorBridgeRuntime()
    client, sessions = local_client()
    rt._clients['acc'] = client
    body = {'messages':[{'role':'user','content':'one'}]}
    sid, owner = rt.begin_session('acc', body, 'anchor', 'key')
    first = client.chat_completions(model='M', messages=body['messages'], session_id=sid)
    branch, owner2 = rt.begin_session('acc', body, 'anchor', 'key')
    second = client.chat_completions(model='M', messages=body['messages'], session_id=branch)
    try:
        assert branch != sid and all(s.alive for s in sessions)
        rt.finish_session('acc', branch, owner=owner)
        assert sessions[1].alive
        rt.finish_session('acc', sid, owner=owner)
        assert not sessions[0].alive and sessions[1].alive
    finally:
        first.close(); second.close(); client.close()


def test_sessions_s08_parallel_threaded_construction_keeps_both_owners():
    from src.tests.test_protocol_audit_cursor_regressions import LocalSession
    entered = threading.Event(); release = threading.Event()
    sessions = []; results = []; errors = []
    def factory(**kwargs):
        session = LocalSession(**kwargs)
        sessions.append(session)
        if len(sessions) == 1:
            entered.set()
            assert release.wait(2)
        return session
    client = CursorClient('fake', session_factory=factory)
    def run():
        try:
            results.append(client.chat_completions(model='M', messages=[{'role':'user','content':'one'}], session_id='same'))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run); thread.start()
    try:
        assert entered.wait(2)
        run()
        release.set(); thread.join(2)
        assert not errors and len(sessions) == 2
        assert all(s.alive for s in sessions)
        results[0].close()
        assert sum(s.alive for s in sessions) == 1
    finally:
        release.set(); thread.join(2)
        for result in results: result.close()
        client.close()


@pytest.mark.parametrize('change', ['system', 'tools', 'long_context'])
def test_sessions_s09_all_effective_constraints_rebuild_without_exec_replies(change):
    first_events = [SessionEvent(type='toolCall', exec=pending()), SessionEvent(type='batchReady')]
    client, sessions = local_client(first_events, [SessionEvent(type='text', text='new'),SessionEvent(type='done')])
    tools = [{'type':'function','function':{'name':'Read','parameters':{'type':'object'}}}]
    try:
        client.chat_completions(model='M', messages=HISTORY[:4], tools=tools, session_id='s', stream=False)
        kwargs = dict(model='M',messages=[dict(i) for i in HISTORY[:6]],tools=tools,session_id='s',stream=False)
        if change == 'system': kwargs['messages'][0]['content'] = 'NEW_RULE'
        if change == 'tools': kwargs['tools'] = []
        if change == 'long_context': kwargs['long_context'] = True
        result = client.chat_completions(**kwargs)
        assert result['choices'][0]['message']['content'] == 'new'
        assert len(sessions) == 2 and not sessions[0].alive
        assert not sessions[0].results and not sessions[1].results
        assert b'TOOL_RESULT_UNIQUE' in sessions[1].kwargs['request_bytes']
    finally:
        client.close()


def test_sessions_s11_cancel_during_late_session_construction():
    from src.tests.test_protocol_audit_cursor_regressions import LocalSession
    from src.cursor_bridge.errors import CursorError
    entered = threading.Event(); release = threading.Event()
    sessions = []; errors = []
    def factory(**kwargs):
        entered.set(); assert release.wait(2)
        session = LocalSession(**kwargs); sessions.append(session)
        return session
    client = CursorClient('fake', session_factory=factory)
    def run():
        try:
            client.chat_completions(model='M', messages=[{'role':'user','content':'hi'}], session_id='s')
        except BaseException as exc: errors.append(exc)
    worker = threading.Thread(target=run); worker.start()
    try:
        assert entered.wait(2)
        client.discard_conversation('s', cancel=True)
        release.set(); worker.join(2)
        assert not worker.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], CursorError)
        assert sessions and not sessions[0].alive
        assert not client._conversations
    finally:
        release.set(); worker.join(2); client.close()


def test_sessions_s12_ttl_and_count_limits_keep_active_conversations():
    from src.cursor_bridge.client import ConversationState
    client, sessions = local_client()
    for i in range(130):
        client._conversations[str(i)] = ConversationState(idle_since=1000+i)
    client._conversations['active'] = ConversationState(idle_since=1,busy=True)
    removed = client.reap_conversations(now=1200)
    assert set(removed) == {'0','1'}
    assert len(client._conversations) == 129
    client.reap_conversations(now=1800)
    assert set(client._conversations) == {'active'}
    client.close()


def test_sessions_s12_blob_byte_budget_reclaims_oversized_idle_state():
    from src.cursor_bridge.client import ConversationState
    client, _ = local_client()
    state = ConversationState(blob_store={'blob': b'x' * (64*1024*1024+1)})
    client._conversations['big'] = state
    assert client.reap_conversations() == ['big']
    assert not state.blob_store and not client._conversations


@pytest.mark.parametrize('split_at', range(1, len('<thinking>reason</thinking>answer')))
def test_sessions_s14_every_tag_boundary_is_equivalent(split_at):
    from src.cursor_bridge.thinking import ThinkingTagFilter
    text = '<thinking>reason</thinking>answer'
    f = ThinkingTagFilter()
    parts = [f.process(text[:split_at]),f.process(text[split_at:]),f.flush()]
    assert ''.join(p.content for p in parts) == 'answer'
    assert ''.join(p.reasoning for p in parts) == 'reason'


from src.tests.test_protocol_audit_ws_regressions import isolated_store


def test_sessions_s15_corrupt_ancestor_fails_real_sqlite_expansion(isolated_store):
    from src.openai import store
    for rid, parent in [('a',None),('b','a')]:
        store.save(rid,parent,api_key_name='k',model='m',channel_key='c',
                   input_items=[{'role':'user','content':rid}],output_items=[])
    conn = store._get_conn()
    conn.execute("UPDATE openai_response_store SET output_items='{' WHERE response_id='a'")
    conn.commit()
    with pytest.raises(store.ResponseHistoryError, match='corrupt response history at a'):
        store.expand_history('b',api_key_name='k')


def test_sessions_s16_partial_and_final_tool_reply_change_timeout_phase(monkeypatch):
    from queue import Queue
    from src.cursor_bridge.session import CursorSession
    from src.cursor_bridge.constants import INACTIVITY_FLUSHED_S, INACTIVITY_STREAMING_S
    import src.cursor_bridge.session as module
    monkeypatch.setattr(module.time, 'monotonic', lambda: 1000.0)
    s = object.__new__(CursorSession)
    s._stream = SimpleNamespace(write=lambda _:None)
    s.events = Queue(); s._timer_phase = 'streaming'
    s.pending_execs = [pending('a'),pending('b')]
    s.send_tool_results([{'tool_call_id':'a','content':'one'}])
    assert s._inactivity_deadline == 1000 + INACTIVITY_FLUSHED_S
    s.send_tool_results([{'tool_call_id':'b','content':'two'}])
    assert s._inactivity_deadline == 1000 + INACTIVITY_STREAMING_S


@pytest.mark.asyncio
@pytest.mark.parametrize('identity_field', ['_api_key_name','_parrot_api_key_name'])
async def test_sessions_s01_channel_always_projects_trusted_principal(monkeypatch, identity_field):
    from src.channel.cursor_oauth_channel import CursorOAuthChannel
    from src.cursor_bridge import runtime as bridge
    from src.tests.test_cursor_oauth_integration import _account
    from src import oauth_manager
    monkeypatch.setattr(bridge, 'base_url', lambda:'http://127.0.0.1:1')
    monkeypatch.setattr(bridge, 'update_account', lambda *args:None)
    async def token(_channel): return 'synthetic-token'
    monkeypatch.setattr(oauth_manager, 'ensure_channel_token', token)
    channel = CursorOAuthChannel(_account())
    payload = {'model':'composer-2.5','messages':[{'role':'user','content':'hi'}],
               identity_field:'key-A','session_id':'user-controlled'}
    a = await channel.build_upstream_request(payload,'composer-2.5',ingress_protocol='chat')
    payload[identity_field] = 'key-B'
    b = await channel.build_upstream_request(payload,'composer-2.5',ingress_protocol='chat')
    assert a.headers[bridge._PRINCIPAL_HEADER] == hashlib.sha256(b'key-A').hexdigest()
    assert b.headers[bridge._PRINCIPAL_HEADER] == hashlib.sha256(b'key-B').hexdigest()
    assert bridge._PRINCIPAL_HEADER not in json.loads(a.body)


def test_sessions_ao06_metadata_without_usage_does_not_erase_observed_tokens():
    tracker = ws._WsTracker()
    tracker.feed_text(json.dumps({'type':'response.in_progress','data':{'usage':{
        'input_tokens':10,'output_tokens':2}}}))
    assert tracker.usage_observed
    tracker.feed_text(json.dumps({'type':'response.in_progress','service_tier':123}))
    assert tracker.usage_observed and tracker.usage['input_tokens'] == 10
    tracker.feed_text(json.dumps({'type':'response.completed','data':{'response':{'usage':{
        'input_tokens':-1,'output_tokens':2}}}}))
    assert not tracker.usage_observed


@pytest.mark.asyncio
async def test_sessions_s06_native_api_warmup_keeps_no_generation(monkeypatch, runtime):
    dispatched, _ = await install_native(monkeypatch, runtime)
    client = Client(); task = asyncio.create_task(ws.handle_responses_ws(client))
    try:
        client.send(create('warmup', generate=False))
        up, wire = await next_dispatch(dispatched)
        assert wire['generate'] is False
        up.feed(terminal('warmup', reason='generate_false'))
        await client.until(lambda event:event.get('type') == 'response.incomplete')
        assert not any(event.get('type') == 'error' for event in client.events)
    finally:
        await finish(client, task)
