# Downstream API keys bound to exact upstream sources

`apiKeys.<name>.allowedChannels` is an optional array of registry source IDs,
for example `oauth:cursor:<account identity>` or `api:<configured channel name>`.
Obtain exact IDs from the management channel inventory rather than constructing
an ID from a display name. Empty/missing arrays preserve the shared pool by default.
`channelBindingEnabled: false` explicitly restores the shared pool while keeping
the selected IDs. Set it to `true` to enable the selection. Enabling an empty
selection is rejected by management; invalid enabled configurations fail closed.
Malformed bindings and removed IDs deny all model routes; they never fall back
to the shared pool. Multiple IDs permit failover only between those sources.

The management API accepts `allowedChannels` in API-key updates and returns it
in key views. It rejects newly added unknown, duplicate and empty IDs. Existing
bindings to unavailable/deleted sources may be retained; clear explicitly with
`[]` to restore the shared pool. API-key secrets remain masked in management views.

`/v1/models` intersects source-bound discovery with the existing `allowedModels`
grants. Text Messages, Chat, Responses (HTTP/WS), Realtime, images and video
routing all honor source bindings, including saturated queues and failover.
Image/video MCP calls share those HTTP executors. Bindings restrict AI model
sources; separately configured search backends retain their own tool permissions.

A Cursor-bound key can expose Claude, GPT, Gemini and Grok names unchanged.
The same model ID may appear in each independent OpenBear provider. OAuth token
refresh does not change the binding. Account deletion/identity changes require
an explicit rebind. This feature does not revoke already-running requests.

## Manual enable/disable

Update a key using the management API:

```json
{"allowedChannels": ["api:example"], "channelBindingEnabled": true}
```

Send `{"channelBindingEnabled": false}` to use the existing shared scheduler.
The selection remains saved, so sending `{"channelBindingEnabled": true}`
restores it. Unbound keys retain existing model compatibility checks, channel
selection, cooldown, queues and safe failover. This does not translate an
unsupported model into another model or change streaming retry rules.

Telegram key details display the selection and whether binding is disabled.
