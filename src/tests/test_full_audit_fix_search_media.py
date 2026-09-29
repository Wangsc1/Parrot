"""SM-01/02/06 and R-05: correct behavior, isolated fake model/transport only."""
import asyncio
import copy
import hashlib
import json
import sqlite3
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from src import local_web_tools as web, search_service, search_tool_policy as policy, translation
from src.tests.search_stream_fixtures import wire, decode, encode


@pytest.fixture
def managed(monkeypatch):
    cfg = {**search_service.DEFAULTS, 'functionMode': 'managed', 'hostedMode': 'managed', 'maxToolRounds': 4}
    monkeypatch.setattr(search_service, 'settings', lambda: cfg)
    monkeypatch.setattr(policy, '_REPLAY', policy.OrderedDict())
    body = {'model': 'fake-model', 'input': 'search then run', 'tools': [
        {'type': 'function', 'name': 'web_search', 'parameters': {'type': 'object'}},
        {'type': 'custom', 'name': 'shell', 'format': {'type': 'text'}}]}
    obj = {'id': 'resp-a', 'object': 'response', 'status': 'completed', 'output': [
        {'id': 'fc-a', 'type': 'function_call', 'call_id': 'search-a', 'name': 'web_search', 'arguments': '{"query":"python"}'},
        {'id': 'ct-a', 'type': 'custom_tool_call', 'call_id': 'shell-a', 'name': 'shell', 'input': 'printf  "你好"\n'}]}
    searches = []
    async def search(args, **kw):
        searches.append(args)
        return {'results': [], 'answer': 'search-evidence'}
    monkeypatch.setattr(search_service, 'search', search)
    return body, obj, searches


def custom_wire(obj, *, terminal_only=False):
    if terminal_only:
        return encode('response.completed', {'type': 'response.completed', 'response': obj})
    frames = [('response.created', {'type': 'response.created', 'response': {**obj, 'output': [], 'status': 'in_progress'}})]
    for index, item in enumerate(obj['output']):
        custom = item['type'] == 'custom_tool_call'
        field = 'input' if custom else 'arguments'
        event = 'response.custom_tool_call_input' if custom else 'response.function_call_arguments'
        frames.append(('response.output_item.added', {'type': 'response.output_item.added', 'output_index': index,
            'item': {**item, field: '', 'status': 'in_progress'}}))
        for fragment in (item[field][:3], item[field][3:]):
            frames.append((event + '.delta', {'type': event + '.delta', 'output_index': index, 'item_id': item['id'], 'delta': fragment}))
        # Sparse terminal must repair both input.done and output_item.done exactly once.
    frames.append(('response.completed', {'type': 'response.completed', 'response': obj}))
    return b''.join(encode(e, data) for e, data in frames)


async def model_response(obj, stream, terminal_only=False):
    if not stream:
        return JSONResponse(obj)
    async def chunks():
        raw = custom_wire(obj, terminal_only=terminal_only)
        for offset in range(0, len(raw), 19):
            yield raw[offset:offset + 19]
    return StreamingResponse(chunks())


@pytest.mark.asyncio
@pytest.mark.parametrize('stream,terminal_only', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('namespace', ['', 'client'])
async def test_custom_mixed_round_and_full_history_resume(managed, stream, terminal_only, namespace):
    body, obj, searches = managed
    if namespace:
        custom = body['tools'].pop()
        body['tools'].append({'type': 'namespace', 'name': namespace, 'tools': [custom]})
        obj['output'][1]['namespace'] = namespace
    invokes = []
    async def invoke(current):
        invokes.append(copy.deepcopy(current))
        assert len(invokes) == 1, 'must hand client call back before continuing the model'
        return await model_response(obj, stream, terminal_only)
    if stream:
        response = policy.stream(body, 'responses', invoke, api_key_name='fake-owner')
        frames = decode(b''.join([chunk async for chunk in response.body_iterator]))
        visible = next(e['response'] for e in frames if e['type'] == 'response.completed')
        inputs = [e for e in frames if e['type'] == 'response.custom_tool_call_input.delta']
        assert ''.join(e['delta'] for e in inputs) == obj['output'][1]['input']
        assert len([e for e in frames if e['type'] == 'response.custom_tool_call_input.done']) == 1
        assert not any(e.get('item', {}).get('type') == 'function_call' for e in frames)
    else:
        visible = json.loads((await policy.run(body, 'responses', invoke, api_key_name='fake-owner')).body)
    assert len(searches) == 1 and len(visible['output']) == 1
    assert visible['output'][0]['call_id'] == 'shell-a'
    assert visible['output'][0]['input'] == obj['output'][1]['input']
    continuation = copy.deepcopy(body)
    policy._append(continuation, visible, [], 'responses')
    continuation['input'].append({'type': 'custom_tool_call_output', 'call_id': 'shell-a', 'output': 'client-result'})
    restored = policy.restore_replay(continuation, 'responses', 'fake-owner')
    assert 'search-evidence' in json.dumps(restored)
    assert policy.restore_replay(restored, 'responses', 'fake-owner') == restored
    assert policy.restore_replay(continuation, 'responses', 'other-owner') == continuation
    async def finish(current):
        assert 'search-evidence' in json.dumps(current) and 'client-result' in json.dumps(current)
        return JSONResponse({'id': 'resp-b', 'object': 'response', 'status': 'completed', 'output': []})
    assert (await policy.run(continuation, 'responses', finish, api_key_name='fake-owner')).status_code == 200
    assert len(searches) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'failure', 'cancel'])
async def test_custom_stream_waits_for_owned_side_before_commit(managed, monkeypatch, outcome):
    body, obj, _ = managed
    started = asyncio.Event(); release = asyncio.Event(); closed = asyncio.Event()
    frames = []
    async def execute(calls, **kw):
        started.set()
        try:
            await release.wait()
            if outcome == 'failure':
                raise ValueError('synthetic managed failure')
            return [web.LocalToolResult(c.id, 'evidence') for c in calls]
        finally:
            closed.set()
    monkeypatch.setattr(web, 'execute_local_tool_calls', execute)
    async def invoke(current): return await model_response(obj, True)
    response = policy.stream(body, 'responses', invoke, api_key_name='owner')
    async def collect():
        async for chunk in response.body_iterator: frames.extend(decode(chunk))
    task = asyncio.create_task(collect())
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert not any(e.get('item', {}).get('type') == 'custom_tool_call' for e in frames)
        if outcome == 'cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        else:
            release.set()
            await asyncio.wait_for(task, 2)
        assert closed.is_set()
        public = [e for e in frames if e.get('item', {}).get('type') == 'custom_tool_call']
        assert bool(public) is (outcome == 'success')
        if outcome == 'failure': assert frames[-1]['type'] == 'response.failed'
    finally:
        release.set()
        if not task.done(): task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_custom_type_is_not_stolen_by_same_named_managed_function(managed, stream):
    body, obj, searches = managed
    obj['output'] = [{**obj['output'][1], 'name': 'web_search'}]
    async def invoke(current): return await model_response(obj, stream)
    if stream:
        raw = b''.join([part async for part in policy.stream(body, 'responses', invoke).body_iterator])
        output = next(e['response']['output'] for e in decode(raw) if e['type'] == 'response.completed')
    else:
        output = json.loads((await policy.run(body, 'responses', invoke)).body)['output']
    assert len(output) == 1 and output[0]['type'] == 'custom_tool_call'
    assert searches == []


@pytest.mark.asyncio
@pytest.mark.parametrize('stream,terminal_only', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('conflict', [False, True])
async def test_custom_duplicate_ids_checked_before_search_effect(managed, stream, terminal_only, conflict):
    body, obj, searches = managed
    obj['output'].append({**obj['output'][1], 'id': 'ct-b',
        'input': obj['output'][1]['input'] + (' ' if conflict else '')})
    async def invoke(current): return await model_response(obj, stream, terminal_only)
    if stream:
        frames = decode(b''.join([part async for part in policy.stream(body, 'responses', invoke).body_iterator]))
        if conflict:
            assert frames[-1]['type'] == 'response.failed'
            assert not any(e.get('item', {}).get('type') == 'custom_tool_call' for e in frames)
        else:
            assert len(frames[-1]['response']['output']) == 1
    else:
        response = await policy.run(body, 'responses', invoke)
        assert response.status_code == (502 if conflict else 200)
        if not conflict: assert len(json.loads(response.body)['output']) == 1
    assert len(searches) == (0 if conflict else 1)


def test_custom_freeform_identity_and_hosted_xai_evidence(managed):
    _, obj, _ = managed
    a = {**obj['output'][1], 'input': '{"a":1}'}
    b = {**a, 'input': '{"a": 1}'}
    with pytest.raises(ValueError, match='tool_call_id_conflict'):
        policy._unique_calls({'output': [a, b]}, 'responses')
    with pytest.raises(ValueError, match='tool_call_id_conflict'):
        policy._unique_calls({'output': [a, {**obj['output'][0], 'call_id': a['call_id'], 'name': a['name'], 'arguments': a['input']}]}, 'responses')
    hosted = {'type': 'custom_tool_call', 'name': 'x_user_search', 'status': 'completed',
              'call_id': 'xs_call-test', 'input': 'native query'}
    assert list(policy._calls({'output': [hosted]}, 'responses')) == []


@pytest.fixture
def translation_env(monkeypatch):
    cfg = {**translation.DEFAULT_TRANSLATION_CONFIG, 'enabled': True, 'model': 'fake-model', 'failureAlertThreshold': 0}
    monkeypatch.setattr(translation, '_get_cfg', lambda: cfg)
    monkeypatch.setattr(translation, '_translation_output_token_limit', lambda *a: 100)
    cache = {}
    monkeypatch.setattr(translation, '_cache_get', cache.get)
    monkeypatch.setattr(translation, '_cache_put', cache.__setitem__)
    return cfg, cache


_FAILURE_STREAMS = [
    ('openai-chat', [{'choices': [{'delta': {'content': 'partial'}, 'finish_reason': None}]}, {'choices': [{'delta': {}, 'finish_reason': 'length'}]}]),
    ('openai-chat', [{'choices': [{'delta': {'content': 'partial'}, 'finish_reason': 'content_filter'}]}]),
    ('anthropic', [{'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': 'partial'}}, {'type': 'message_delta', 'delta': {'stop_reason': 'max_tokens'}}, {'type': 'message_stop'}]),
    ('openai-responses', [{'type': 'response.output_text.delta', 'delta': 'partial'}, {'type': 'response.failed', 'response': {'status': 'failed', 'error': {'message': 'synthetic'}}}]),
    ('openai-responses', [{'type': 'response.output_text.delta', 'delta': 'partial'}, {'type': 'response.incomplete', 'response': {'status': 'incomplete', 'incomplete_details': {'reason': 'content_filter'}}}]),
    ('openai-responses', [{'type': 'response.output_text.delta', 'delta': 'partial'}]),
    ('openai-chat', [{'choices': [{'delta': {'content': 'partial'}}]}]),
    ('anthropic', [{'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': 'partial'}}]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('protocol,events', _FAILURE_STREAMS)
async def test_failed_translation_sse_preserves_original_and_never_caches(translation_env, monkeypatch, protocol, events):
    cfg, cache = translation_env
    class Channel:
        key = 'api:fake'
        async def build_upstream_request(self, *a, **kw):
            return SimpleNamespace(url='https://fake.invalid/translation', headers={}, body=b'{}')
    ch = Channel(); ch.protocol = protocol
    monkeypatch.setattr(translation, '_find_channel_for_model', lambda *a: (ch, 'fake-model'))
    raw = b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events)
    calls = []
    async def fake(request):
        calls.append(request)
        return httpx.Response(200, content=raw, headers={'content-type': 'text/event-stream'})
    monkeypatch.setattr(translation.network, 'async_client', lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(fake)))
    original = {'model': 'downstream', 'input': '完整原始指令：先做甲，再做乙。'}
    for _ in range(2):
        assert await translation.translate_body(original, ingress_protocol='responses') == original
    assert not cache and len(calls) == 2
    # A completed fallback remains usable; do not replace failures with blanket rejection.
    cfg['fallbackModel'] = 'fallback'
    real_call = translation._call_model
    async def fallback(model, *a, **kw):
        if model == 'fallback': return 'complete fallback translation'
        return await real_call(model, *a, **kw)
    monkeypatch.setattr(translation, '_call_model', fallback)
    assert (await translation.translate_body(original, ingress_protocol='responses'))['input'] == 'complete fallback translation'
    assert len(cache) == 1


@pytest.mark.parametrize('protocol,frames', [
    ('openai-chat', [{'choices': [{'delta': {'content': 'complete'}}]}, '[DONE]']),
    ('openai-chat', [{'choices': [{'delta': {'content': 'complete'}, 'finish_reason': 'stop'}]}]),
    ('anthropic', [{'type': 'content_block_delta', 'delta': {'text': 'complete'}}, {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}}, {'type': 'message_stop'}]),
    ('openai-responses', [{'type': 'response.output_text.delta', 'delta': 'complete'}, {'type': 'response.completed', 'response': {'output': []}}]),
])
def test_translation_success_terminal_forms_and_malformed_tail(protocol, frames):
    raw = b''.join(b'data: ' + (frame.encode() if isinstance(frame, str) else json.dumps(frame).encode()) + b'\n\n' for frame in frames)
    assert translation._extract_text_from_sse(raw, protocol, 'fake') == 'complete'
    assert translation._extract_text_from_sse(raw + b'data: {broken\n\n', protocol, 'fake') is None


@pytest.mark.parametrize('promote', ['preload', 'lookup'])
def test_translation_cache_preload_and_promotion_keep_original_expiry(monkeypatch, promote):
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE translation_cache(cache_key TEXT PRIMARY KEY, translated TEXT, created_at REAL)')
    clock = [200000.0]
    conn.executemany('INSERT INTO translation_cache VALUES(?,?,?)', [('expired', 'old', 0), ('near', 'near expiry', clock[0] - 86400 + 1)])
    monkeypatch.setattr(translation, '_db', conn)
    monkeypatch.setattr(translation, '_mem_cache', translation.OrderedDict())
    monkeypatch.setattr(translation, '_mem_cache_bytes', 0)
    monkeypatch.setattr(translation.time, 'time', lambda: clock[0])
    cfg = {**translation.DEFAULT_TRANSLATION_CONFIG, 'cacheTtlDays': 1, 'memoryCacheTtlSeconds': 0}
    monkeypatch.setattr(translation, '_get_cfg', lambda: cfg)
    if promote == 'preload': translation._preload(100)
    assert translation._cache_get('expired') is None
    assert translation._cache_get('near') == 'near expiry'
    clock[0] += 2
    assert translation._cache_get('near') is None
    translation._preload(100)
    assert not translation._mem_cache
    conn.close()


def test_translation_cache_version_invalidates_unverified_sse_entries():
    old = {'v': 2, 'target_language': 'English', 'prompt_sha256': hashlib.sha256(b'prompt').hexdigest(), 'model_signature': '', 'text': 'original'}
    key = hashlib.sha256(json.dumps(old, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert translation._make_cache_key('English', 'original', 'prompt') != key


@pytest.mark.asyncio
@pytest.mark.parametrize('consume_ping', [False, True])
async def test_real_failover_compact_response_close_cancels_and_drains_task(monkeypatch, consume_ping):
    from src import failover, scheduler, log_db, compact_rescue
    from src.tests.test_routing_review_fixes import setup
    setup(monkeypatch)
    started = asyncio.Event(); finalizing = asyncio.Event(); release_cleanup = asyncio.Event()
    owned = []
    async def rescue(*a, **kw):
        owned.append(asyncio.current_task()); started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalizing.set()
            await release_cleanup.wait()
    monkeypatch.setattr(failover, '_run_compact_map_reduce_rescue', rescue)
    monkeypatch.setattr(compact_rescue, 'is_claude_code_compact_request', lambda body: True)
    body = {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'fixture compact'}]}
    rid = 'fix-compact-close-' + str(consume_ping)
    log_db.insert_pending(rid, '127.0.0.1', 'ws-key', 'test-model', True, 1, 0, {}, body, ingress_protocol='anthropic')
    response = await failover.run_failover(scheduler.ScheduleResult([], None, False), body, rid, 'ws-key', '127.0.0.1', True, time.time(), ingress_protocol='anthropic')
    await asyncio.wait_for(started.wait(), 2)
    if consume_ping: assert b'ping' in await response.body_iterator.__anext__()
    close_task = asyncio.create_task(response.body_iterator.aclose())
    try:
        await asyncio.wait_for(finalizing.wait(), 2)
        close_task.cancel(); await asyncio.sleep(0)
        close_task.cancel(); await asyncio.sleep(0)
        assert not close_task.done() and not owned[0].done()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError): await close_task
        assert owned[0].done() and owned[0].cancelled()
    finally:
        release_cleanup.set()
        await asyncio.gather(close_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_owned_ping_response_asgi_send_failure_before_iteration():
    started = asyncio.Event(); ended = asyncio.Event()
    async def work():
        started.set()
        try: await asyncio.Event().wait()
        finally: ended.set()
    task = asyncio.create_task(work())
    await started.wait()
    response = web.stream_anthropic_response_task_with_pings(task)
    async def send(event): raise RuntimeError('fake send failure')
    async def receive(): await asyncio.Event().wait()
    with pytest.raises(RuntimeError, match='fake send failure'):
        await response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, send)
    assert task.cancelled() and ended.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('wrapper,protocol', [(web.stream_anthropic_response_task_with_pings, 'anthropic'), (web.stream_responses_response_task_with_pings, 'responses')])
async def test_owned_ping_response_preserves_success(wrapper, protocol):
    async def work():
        if protocol == 'anthropic':
            return JSONResponse({'type': 'message', 'role': 'assistant', 'content': [{'type': 'text', 'text': 'complete'}], 'stop_reason': 'end_turn'})
        return JSONResponse({'object': 'response', 'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'complete'}]}]})
    task = asyncio.create_task(work())
    await task
    response = wrapper(task)
    raw = b''.join([chunk async for chunk in response.body_iterator])
    assert b'complete' in raw and not task.cancelled()


@pytest.mark.asyncio
async def test_owned_ping_asgi_disconnect_drains_cancelled_work_under_anyio_scope():
    started = asyncio.Event(); ping = asyncio.Event(); finalizing = asyncio.Event()
    release = asyncio.Event(); ended = asyncio.Event()
    async def work():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalizing.set()
            await release.wait()
            ended.set()
    child = asyncio.create_task(work())
    await started.wait()
    response = web.stream_anthropic_response_task_with_pings(child)
    async def send(event):
        if event.get('type') == 'http.response.body': ping.set()
    async def receive():
        await ping.wait()
        return {'type': 'http.disconnect'}
    serving = asyncio.create_task(response({'type': 'http', 'asgi': {'spec_version': '2.3'}}, receive, send))
    try:
        await asyncio.wait_for(finalizing.wait(), 2)
        assert not serving.done() and not child.done()
        release.set()
        await asyncio.wait_for(serving, 2)
        assert child.cancelled() and ended.is_set()
    finally:
        release.set()
        await asyncio.gather(serving, return_exceptions=True)
