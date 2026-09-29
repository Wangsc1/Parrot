"""SM-03/04/05: real HTTP handlers, isolated stores and controlled disk workers."""
import asyncio
import copy
import json
import threading
import uuid
from pathlib import Path

import httpx
import pytest

from src import channel_state, concurrency, config, cooldown, media_db, oauth_manager, state_db
from src.channel import registry
from src.channel.xai_oauth_channel import XAIOAuthChannel
from src.xai import imagine
from src.tests import conftest
from src.tests.test_xai_imagine import _setup, _install_channel
from src.tests.test_media_followup_video import pending_job, request, video_request


@pytest.mark.asyncio
@pytest.mark.parametrize('poll', [False, True])
@pytest.mark.parametrize('error_shape', ['json', 'text', 'async_failure'])
async def test_video_known_secrets_redacted_from_client_and_error_log(monkeypatch, poll, error_shape):
    channel = _install_channel()
    pending_job(channel)
    account = config._cache['oauthAccounts'][0]
    account.update(refresh_token='fake-private-refresh', access_token='fake-private-before', id_token='fake-private-id')
    sent_token = 'fake-private-sent-token'
    async def token(*a, **kw):
        account['access_token'] = 'fake-private-after-refresh'
        return sent_token
    monkeypatch.setattr(oauth_manager, 'ensure_valid_token', token)
    private = [account['access_token'], account['refresh_token'], account['id_token'], channel.email, channel.subject, sent_token, 'fake-private-after-refresh']
    async def upstream(*a, headers, **kw):
        message = 'rejected ' + headers['authorization'] + ' ' + ' '.join(private)
        hdrs = {'set-cookie': 'synthetic-cookie', 'x-debug-auth': headers['authorization'], 'content-encoding': 'identity'}
        if error_shape == 'text':
            return httpx.Response(401, text=message, headers=hdrs)
        return httpx.Response(200 if error_shape == 'async_failure' else 401,
            json={'request_id': 'job', 'status': 'failed', 'error': {'type': 'bad_auth', 'message': message}}, headers=hdrs)
    monkeypatch.setattr(imagine, '_request_upstream', upstream)
    response = await video_request(poll)
    assert response.status_code == (200 if error_shape == 'async_failure' else 401)
    assert 'set-cookie' not in response.headers and 'content-encoding' not in response.headers
    assert '[private]' in response.text and 'rejected' in response.text
    for secret in private:
        assert secret not in response.text and secret not in str(response.headers)
    for row in media_db.recent():
        for secret in private:
            assert secret not in str(row.get('error_message'))
    if error_shape != 'text':
        assert response.json()['error']['type'] == 'bad_auth'


@pytest.mark.asyncio
@pytest.mark.parametrize('poll', [False, True])
async def test_video_success_payload_and_necessary_headers_stay_intact(monkeypatch, poll):
    channel = _install_channel()
    pending_job(channel)
    obj = {'request_id': 'job', 'status': 'done', 'progress': 100, 'model': 'grok-imagine-video',
           'usage': {'cost_in_usd_ticks': 123}, 'video': {'url': 'https://media.x.ai/signed.mp4?token=public-capability', 'duration': 5}}
    raw = json.dumps(obj, indent=2).encode()
    async def upstream(*a, **kw):
        return httpx.Response(200, content=raw, headers={'content-type': 'application/json', 'x-request-id': 'public-request'})
    monkeypatch.setattr(imagine, '_request_upstream', upstream)
    response = await video_request(poll)
    assert response.status_code == 200 and response.content == raw
    assert response.headers['x-request-id'] == 'public-request'


def replace_account(old):
    channel_state.retire_deleted(old.state_key)
    account = copy.deepcopy(config._cache['oauthAccounts'][0])
    account['generationId'] = uuid.uuid4().hex
    config._cache['oauthAccounts'] = [account]
    fresh = XAIOAuthChannel(account)
    registry._channels[old.key] = fresh
    assert fresh.state_key != old.state_key
    return fresh


@pytest.mark.asyncio
@pytest.mark.parametrize('poll', [False, True])
@pytest.mark.parametrize('status', [200, 401, 429, 503, 'headers_fail'])
async def test_video_late_attempt_health_does_not_change_replacement(monkeypatch, poll, status):
    old = _install_channel()
    pending_job(old)
    config._cache['oauthGraceCount'] = 3
    entered = asyncio.Event(); release = asyncio.Event()
    async def headers():
        entered.set(); await release.wait()
        raise RuntimeError('synthetic token-build failure')
    async def upstream(*a, **kw):
        entered.set(); await release.wait()
        if status == 200:
            return httpx.Response(200, json={'request_id': 'job', 'status': 'pending'})
        return httpx.Response(status, json={'error': {'message': 'synthetic late failure'}})
    if status == 'headers_fail': monkeypatch.setattr(old, 'build_media_headers', headers)
    monkeypatch.setattr(imagine, '_request_upstream', upstream)
    task = asyncio.create_task(video_request(poll))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        fresh = replace_account(old)
        cooldown.record_error(fresh.state_key, 'grok-imagine-video', 'new generation grace error')
        before = copy.deepcopy(cooldown._entries)
        assert before and not cooldown.is_blocked(fresh.state_key, 'grok-imagine-video')
        release.set()
        response = await asyncio.wait_for(task, 2)
        expected = (502 if poll else 503) if status == 'headers_fail' else status
        assert response.status_code == expected
        assert cooldown._entries == before
        assert all(slot.in_flight == 0 for slot in concurrency._slots.values())
    finally:
        release.set()
        if not task.done(): task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [200, 429])
async def test_video_live_generation_still_updates_health(monkeypatch, status):
    channel = _install_channel()
    config._cache['oauthGraceCount'] = 3
    cooldown.record_error(channel.state_key, 'grok-imagine-video', 'grace error')
    async def upstream(*a, **kw):
        return httpx.Response(status, json={'request_id': 'job', 'status': 'pending'})
    monkeypatch.setattr(imagine, '_request_upstream', upstream)
    assert (await video_request()).status_code == status
    entry = cooldown._entries.get((channel.key, 'grok-imagine-video'))
    assert entry is None if status == 200 else entry['error_count'] == 2


async def wait_thread_event(event):
    async def wait():
        while not event.is_set(): await asyncio.sleep(.001)
    await asyncio.wait_for(wait(), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize('poll', [False, True])
@pytest.mark.parametrize('phase', ['write', 'cleanup', 'download_second'])
async def test_cancelled_video_cache_drains_real_worker_and_discards_only_unpublished_batch(monkeypatch, tmp_path, poll, phase):
    channel = _install_channel()
    pending_job(channel)
    root = tmp_path / 'cache'
    root.mkdir()
    retained = root / 'retained.mp4'; retained.write_bytes(b'previous published media')
    config._cache['videos'] = {'enabled': True, 'cacheEnabled': True, 'cachePath': str(root)}
    config._cache['concurrency'] = {'enabled': True, 'defaultMaxConcurrent': 1}
    monkeypatch.setattr(asyncio, 'to_thread', conftest._ORIG_TO_THREAD)
    entered = threading.Event(); release = threading.Event(); finished = threading.Event()
    created = []; calls = []
    real_write = imagine._write_cached_media
    real_cleanup = imagine.media_cache.cleanup
    def barrier():
        entered.set()
        if not release.wait(5): raise TimeoutError('test worker not released')
    def write(*a, **kw):
        if phase == 'write': barrier()
        path = real_write(*a, **kw)
        created.append(path)
        if phase == 'write': finished.set()
        return path
    def cleanup(*a, **kw):
        if phase == 'cleanup': barrier()
        result = real_cleanup(*a, **kw)
        if phase == 'cleanup': finished.set()
        return result
    async def download(*a, **kw):
        calls.append('download')
        if phase == 'download_second' and len(calls) == 2:
            entered.set()
            try: await asyncio.Event().wait()
            finally: finished.set()
        return b'fake video bytes', 'video/mp4'
    async def upstream(*a, **kw):
        return httpx.Response(200, json={'request_id': 'job', 'status': 'done', 'video': {'url': 'https://media.x.ai/fake.mp4'}})
    monkeypatch.setattr(imagine, '_write_cached_media', write)
    monkeypatch.setattr(imagine.media_cache, 'cleanup', cleanup)
    monkeypatch.setattr(imagine, '_download_xai_media', download)
    monkeypatch.setattr(imagine, '_request_upstream', upstream)
    if phase == 'download_second':
        # Multi-artifact helper must also roll back earlier files if next download cancels.
        task = asyncio.create_task(imagine._cache_xai_results([{'url': 'https://media.x.ai/fake.mp4'}] * 2,
            media_type='video', action='generate', channel=channel, model='grok-imagine-video'))
    else:
        task = asyncio.create_task(video_request(poll))
    try:
        await wait_thread_event(entered)
        task.cancel()
        if phase != 'download_second':
            for _ in range(5): await asyncio.sleep(0)
            task.cancel()
            for _ in range(5): await asyncio.sleep(0)
            assert not task.done() and not finished.is_set()
            release.set()
        with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(task, 3)
        assert finished.is_set() and len(created) == 1
        assert all(not Path(path).exists() for path in created)
        assert [p for p in root.rglob('*') if p.is_file()] == [retained]
        if phase != 'download_second':
            row = media_db.get_by_upstream_request_id('job')
            assert row['status'] == ('pending' if poll else 'cancelled')
            assert all(path not in (row.get('cache_paths') or '') for path in created)
        assert all(slot.in_flight == 0 for slot in concurrency._slots.values())
    finally:
        release.set()
        if not task.done(): task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('poll', [False, True])
async def test_successful_real_video_cache_is_retained_and_accounted(monkeypatch, tmp_path, poll):
    channel = _install_channel(); pending_job(channel)
    config._cache['videos'] = {'enabled': True, 'cacheEnabled': True, 'cachePath': str(tmp_path)}
    monkeypatch.setattr(asyncio, 'to_thread', conftest._ORIG_TO_THREAD)
    async def upstream(*a, **kw):
        return httpx.Response(200, json={'request_id': 'job', 'status': 'done', 'video': {'url': 'https://media.x.ai/fake.mp4'}})
    async def download(*a, **kw): return b'fake video bytes', 'video/mp4'
    monkeypatch.setattr(imagine, '_request_upstream', upstream)
    monkeypatch.setattr(imagine, '_download_xai_media', download)
    response = await video_request(poll)
    assert response.status_code == 200
    row = media_db.get_by_upstream_request_id('job')
    paths = imagine._existing_cache_paths(row)
    assert row['status'] == 'success' and len(paths) == 1 and row['image_bytes'] == len(b'fake video bytes')
    assert Path(paths[0]).read_bytes() == b'fake video bytes'


@pytest.mark.asyncio
async def test_poll_cancel_after_cache_transfers_receipt_to_owned_durable_update(monkeypatch, tmp_path):
    channel = _install_channel(); pending_job(channel)
    config._cache['videos'] = {'enabled': True, 'cacheEnabled': True, 'cachePath': str(tmp_path)}
    monkeypatch.setattr(asyncio, 'to_thread', conftest._ORIG_TO_THREAD)
    entered = threading.Event(); release = threading.Event()
    original_update = media_db.update_job
    def update(*a, **kw):
        entered.set()
        if not release.wait(5): raise TimeoutError('test update not released')
        return original_update(*a, **kw)
    async def upstream(*a, **kw):
        return httpx.Response(200, json={'status': 'done', 'video': {'url': 'https://media.x.ai/fake.mp4'}})
    async def download(*a, **kw): return b'fake video bytes', 'video/mp4'
    monkeypatch.setattr(media_db, 'update_job', update)
    monkeypatch.setattr(imagine, '_request_upstream', upstream)
    monkeypatch.setattr(imagine, '_download_xai_media', download)
    task = asyncio.create_task(video_request(True))
    try:
        await wait_thread_event(entered)
        task.cancel()
        for _ in range(5): await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError): await task
        row = media_db.get_by_upstream_request_id('job')
        paths = imagine._existing_cache_paths(row)
        assert row['status'] == 'success' and len(paths) == 1
        assert Path(paths[0]).read_bytes() == b'fake video bytes'
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
