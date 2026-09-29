"""Split P15 cancellation ownership gates; shared fakes live in the support module."""

from src.tests.reaudit2_runtime_cancellation_support import *  # noqa: F401,F403

@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True], ids=["candidate", "queued"])
async def test_cancel_outer_retry_open_update_aborts_unstarted_http_stream_before_slot_release(
    monkeypatch,
    queued,
):
    logs = _StrictLogFakes()
    record_retry = logs.record_retry_attempt
    update_retry = logs.update_retry_attempt
    finish_error = logs.finish_error
    update_proxy = logs.update_proxy_attempt
    monkeypatch.setattr(failover.log_db, "record_retry_attempt", record_retry)
    monkeypatch.setattr(failover.log_db, "update_retry_attempt", update_retry)
    monkeypatch.setattr(failover.log_db, "finish_error", finish_error)
    monkeypatch.setattr(http_runtime.log_db, "update_proxy_attempt", update_proxy)
    monkeypatch.setattr(
        failover.config,
        "get",
        lambda: {
            "timeouts": {"connect": 1, "firstByte": 1, "idle": 1, "total": 5},
            "concurrency": {"queueWaitSeconds": 1},
        },
    )
    monkeypatch.setattr(
        failover,
        "_pick_non_direct_proxy_name",
        lambda ch, resolved_model: None,
    )
    monkeypatch.setattr(
        failover,
        "_should_use_responses_upstream_ws",
        lambda ch, *, ingress_protocol, cfg=None: False,
    )

    channel = SimpleNamespace(
        key="api:one",
        type="api",
        protocol="anthropic",
        provider="anthropic",
        cc_mimicry=False,
        upstream_stream_only=False,
    )
    slot_releases: list[str] = []

    async def try_acquire(ch_key: str) -> bool:
        assert not queued
        assert ch_key == channel.key
        return True

    async def acquire_from_candidates(candidates, timeout: float):
        assert queued
        assert timeout == 1
        return candidates[0]

    def release(ch_key: str) -> None:
        slot_releases.append(ch_key)

    monkeypatch.setattr(failover.concurrency, "try_acquire", try_acquire)
    monkeypatch.setattr(
        failover.concurrency, "acquire_from_candidates", acquire_from_candidates,
    )
    monkeypatch.setattr(failover.concurrency, "release", release)

    context = _CountingContext()
    client = _CountingClient()
    timing = HttpAttemptTiming(response_mode="stream", round_id="round-1")
    timing.mark_connection_complete()
    timing.start_response_body_wait()
    timing.mark_response_body_byte(b"first")
    opened = OpenedHttpResponse(
        ctx=context,
        response=_HttpResponse(),
        connect_ms=timing.snapshot().connect_ms,
        timing=timing,
        proxy_name="proxy-a",
        proxy_bytes={"up": 3, "down": 4},
        proxy_client=client,
        proxy_attempt_id="proxy-1",
        round_timeouts=RoundTimeouts.from_config({
            "connect": 1, "firstByte": 1, "idle": 1, "total": 5,
        }),
    )

    async def prepare_stream_response_start(
        ctx,
        response,
        channel_arg,
        *,
        dynamic_map,
        connect_ms,
        deadline_ts,
        first_byte_timeout,
        idle_timeout,
        ingress_protocol,
        timing=None,
        round_timeouts=None,
        translator_ctx=None,
        partial_state=None,
    ):
        del (
            ctx, response, channel_arg, dynamic_map, connect_ms, deadline_ts,
            first_byte_timeout, idle_timeout, ingress_protocol, timing,
            round_timeouts, translator_ctx, partial_state,
        )
        return HttpStreamStartResult(
            aiter=_NeverIterated(),
            tracker=_Tracker(),
            builder=SimpleNamespace(),
            first_downstream_chunks=[b"first"],
            first_byte_ms=1,
            response_headers={},
            upstream_status=200,
        )

    monkeypatch.setattr(
        failover, "prepare_stream_response_start", prepare_stream_response_start,
    )

    async def try_channel(
        ch,
        resolved_model,
        body,
        is_stream,
        deadline_ts,
        start_time,
        fp_query,
        messages,
        api_key_name,
        client_ip,
        request_id,
        retry_count_so_far,
        affinity_hit,
        *,
        ingress_protocol="anthropic",
        client_key=None,
        retry_attempt_id=None,
        start_monotonic=None,
        attempt_start_monotonic=None,
        terminal_release=None,
    ):
        assert ch is channel
        assert is_stream is True
        return await failover._consume_stream(
            context,
            opened.response,
            ch,
            resolved_model,
            None,
            opened.connect_ms,
            start_time,
            deadline_ts,
            1,
            1,
            request_id,
            messages,
            api_key_name,
            client_ip,
            fp_query,
            retry_count_so_far,
            affinity_hit,
            ingress_protocol=ingress_protocol,
            translator_ctx=None,
            body=body,
            client_key=client_key,
            proxy_name=opened.proxy_name,
            proxy_bytes=opened.proxy_bytes,
            proxy_client=opened.proxy_client,
            timing=opened.timing,
            round_timeouts=opened.round_timeouts,
            opened_response=opened,
            retry_attempt_id=retry_attempt_id,
            start_monotonic=start_monotonic,
            attempt_start_monotonic=attempt_start_monotonic,
            cancel_state={},
            terminal_release=terminal_release,
        )

    monkeypatch.setattr(failover, "_try_channel", try_channel)

    retry_worker_entered = asyncio.Event()
    allow_retry_worker = asyncio.Event()

    async def controlled_to_thread(func, /, *args, **kwargs):
        if func is update_retry and kwargs.get("outcome") == "open":
            retry_worker_entered.set()
            await allow_retry_worker.wait()
        return func(*args, **kwargs)

    monkeypatch.setattr(failover.asyncio, "to_thread", controlled_to_thread)
    task = asyncio.create_task(failover.run_failover(
        _schedule_result(channel, queued=queued),
        {"model": "m", "stream": True, "messages": []},
        "request-1",
        "key-1",
        "127.0.0.1",
        True,
        time.time(),
        ingress_protocol="anthropic",
    ))
    await retry_worker_entered.wait()
    task.cancel()
    allow_retry_worker.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    await http_runtime._drain_proxy_attempt_persistence()

    assert context.exit_calls == 1
    assert client.close_calls == 1
    assert slot_releases == [channel.key]
    assert [row["outcome"] for row in logs.retry_updates] == [
        "open", "client_disconnected",
    ]
    assert all(
        row["attempt_id"] == logs.retry_records[0]["handle"]
        for row in logs.retry_updates
    )
    assert [
        (row["status"], row["http_status"]) for row in logs.request_terminals
    ] == [("cancelled", 499)]
    assert [row["outcome"] for row in logs.proxy_updates] == [
        "client_disconnected",
    ]
    assert logs.proxy_updates[0]["proxy_attempt_id"] == opened.proxy_attempt_id
    assert logs.proxy_updates[0]["ended_at"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("queued", "blocked_write"),
    [
        (False, "record_retry_attempt"),
        (False, "update_pending"),
        (False, "update_retry_attempt"),
        (True, "record_retry_attempt"),
        (True, "update_retry_attempt"),
    ],
    ids=[
        "candidate-retry-record",
        "candidate-pending-update",
        "candidate-retry-terminal",
        "queued-retry-record",
        "queued-retry-terminal",
    ],
)
@pytest.mark.parametrize("repeat_cancel", [False, True], ids=["single-cancel", "repeated-cancel"])
async def test_responses_ws_cancel_during_preaccept_log_update_terminalizes_all_owners_once(
    monkeypatch,
    queued,
    blocked_write,
    repeat_cancel,
):
    logs = _StrictLogFakes()
    insert_pending = logs.insert_pending
    record_retry = logs.record_retry_attempt
    update_retry = logs.update_retry_attempt
    finish_error = logs.finish_error

    def update_pending(*_args, **_kwargs):
        return None

    monkeypatch.setattr(responses_ws.log_db, "insert_pending", insert_pending)
    monkeypatch.setattr(responses_ws.log_db, "record_retry_attempt", record_retry)
    monkeypatch.setattr(responses_ws.log_db, "update_pending", update_pending)
    monkeypatch.setattr(responses_ws.log_db, "update_retry_attempt", update_retry)
    monkeypatch.setattr(responses_ws.log_db, "finish_error", finish_error)

    channel = SimpleNamespace(
        key="oauth:one",
        type="oauth",
        protocol="openai-responses",
        provider="openai",
        account_key="account-1",
        email="unit@example.invalid",
    )
    schedule = _schedule_result(channel, queued=queued)
    monkeypatch.setattr(
        responses_ws.auth,
        "validate",
        lambda headers: ("key-1", None, None),
    )
    monkeypatch.setattr(
        responses_ws,
        "_request_body_from_ws_create",
        lambda obj: {"model": obj["model"], "input": obj.get("input", [])},
    )
    monkeypatch.setattr(
        responses_ws.model_mapping,
        "apply_default",
        lambda body, ingress_protocol: None,
    )
    monkeypatch.setattr(
        responses_ws.model_mapping,
        "apply_mapping",
        lambda body, ingress_protocol: None,
    )
    monkeypatch.setattr(
        responses_ws,
        "guard_responses_ingress",
        lambda body, *, store_enabled: None,
    )
    monkeypatch.setattr(
        responses_ws.local_web_tools,
        "prepare_openai_responses_local_web_tools",
        lambda body: False,
    )
    monkeypatch.setattr(
        responses_ws.fingerprint,
        "fingerprint_query_responses",
        lambda api_key_name, client_ip, input_items: "fp-1",
    )

    def maybe_apply_auto_prompt_cache_key(
        body,
        *,
        fp_query,
        api_key_name="",
        client_ip="",
        model="",
        ingress_protocol="chat",
        claude_code_session_id=None,
        claude_code_agent_id=None,
    ):
        del (
            body, fp_query, api_key_name, client_ip, model, ingress_protocol,
            claude_code_session_id, claude_code_agent_id,
        )
        return None

    monkeypatch.setattr(
        responses_ws,
        "_maybe_apply_auto_prompt_cache_key",
        maybe_apply_auto_prompt_cache_key,
    )

    def schedule_request(
        body,
        api_key_name,
        client_ip,
        ingress_protocol="anthropic",
        fp_query=None,
    ):
        del body, api_key_name, client_ip, ingress_protocol, fp_query
        return schedule

    monkeypatch.setattr(responses_ws.scheduler, "schedule", schedule_request)

    async def translate_body(body, *, ingress_protocol, route=None):
        del ingress_protocol, route
        return body

    monkeypatch.setattr(responses_ws.translation, "translate_body", translate_body)

    lease = _FakeLease()

    async def acquire_api_key(key_name, request=None, *, receive=None):
        assert key_name == "key-1"
        assert request is None and receive is None
        return lease

    monkeypatch.setattr(responses_ws.apikey_limiter, "acquire", acquire_api_key)
    monkeypatch.setattr(
        responses_ws.config,
        "get",
        lambda: {
            "timeouts": {"total": 5},
            "concurrency": {"queueWaitSeconds": 1},
        },
    )
    monkeypatch.setattr(
        responses_ws,
        "_is_ws_capable_channel",
        lambda ch: ch is channel,
    )
    monkeypatch.setattr(
        responses_ws,
        "_pick_non_direct_proxy_name",
        lambda ch, resolved_model: (
            "proxy-a" if blocked_write == "update_pending" else None
        ),
    )

    slot_releases: list[str] = []

    async def try_acquire(ch_key: str) -> bool:
        assert not queued
        assert ch_key == channel.key
        return True

    async def acquire_from_candidates(candidates, timeout: float):
        assert queued
        assert timeout == 1
        return candidates[0]

    def release(ch_key: str) -> None:
        slot_releases.append(ch_key)

    monkeypatch.setattr(responses_ws.concurrency, "try_acquire", try_acquire)
    monkeypatch.setattr(
        responses_ws.concurrency, "acquire_from_candidates", acquire_from_candidates,
    )
    monkeypatch.setattr(responses_ws.concurrency, "release", release)

    upstream_ws = _FakeUpstreamWebSocket()

    async def try_ws_channel(
        websocket,
        *,
        first_obj,
        ch,
        resolved_model,
        body,
        allowed_models,
        deadline_ts,
        start_time,
        request_id,
        retry_count_so_far,
        affinity_hit,
        api_key_name,
        client_ip,
        fp_query,
        client_key,
        retry_attempt_id,
        start_monotonic,
        attempt_start_monotonic,
        turn_capacity,
    ):
        del (
            websocket, first_obj, resolved_model, body, allowed_models,
            deadline_ts, start_time, request_id, retry_count_so_far,
            affinity_hit, api_key_name, client_ip, fp_query, client_key,
            retry_attempt_id, start_monotonic, attempt_start_monotonic,
            turn_capacity,
        )
        assert ch is channel
        # Faithful pre-accept failure contract: the attempt closes its upstream
        # owner before returning a candidate-failover result.
        await upstream_ws.close()
        return responses_ws._WsAttemptResult(
            outcome="connect_error",
            error_detail="upstream rejected before accept",
            proxy_name="proxy-a",
            upstream_protocol="openai-responses",
            upstream_transport="ws",
        )

    monkeypatch.setattr(responses_ws, "_try_ws_channel", try_ws_channel)

    retry_worker_entered = asyncio.Event()
    allow_retry_worker = asyncio.Event()

    blocked_func = {
        "record_retry_attempt": record_retry,
        "update_pending": update_pending,
        "update_retry_attempt": update_retry,
    }[blocked_write]

    async def controlled_to_thread(func, /, *args, **kwargs):
        if func is blocked_func:
            retry_worker_entered.set()
            await allow_retry_worker.wait()
        return func(*args, **kwargs)

    monkeypatch.setattr(responses_ws.asyncio, "to_thread", controlled_to_thread)

    reader_stopped = asyncio.Event()

    class PersistentDownstream(_FakeDownstreamWebSocket):
        async def receive(self):
            if self.receive_calls:
                # The connection mux owns a persistent reader after first create.
                # A second receive is not an assertion failure/disconnect.
                try:
                    await asyncio.Event().wait()
                finally:
                    reader_stopped.set()
            return await super().receive()

    websocket = PersistentDownstream()
    task = asyncio.create_task(responses_ws.handle_responses_ws(websocket))
    await retry_worker_entered.wait()
    task.cancel()
    if repeat_cancel:
        await asyncio.wait_for(reader_stopped.wait(), 1)
        task.cancel()
        assert not task.done()  # The persistence owner is still blocked.
    allow_retry_worker.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert upstream_ws.close_calls == (
        1 if blocked_write == "update_retry_attempt" else 0
    )
    assert slot_releases == [channel.key]
    assert lease.release_calls == 1
    assert [row["outcome"] for row in logs.retry_updates] == [
        "connect_error" if blocked_write == "update_retry_attempt" else "cancelled"
    ]
    assert logs.retry_updates[0]["attempt_id"] == logs.retry_records[0]["handle"]
    assert logs.retry_updates[0]["ended_at"] is not None
    assert [
        (row["status"], row["http_status"]) for row in logs.request_terminals
    ] == [("cancelled", 499)]
