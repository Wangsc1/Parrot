"""下游 API Key 验证（常数时间比较，防止时序侧信道）。

返回三元组 (key_name, allowed_models, err)：
  - 验证通过：allowed_models 为列表（空 = 无限制，非空 = 白名单）
  - 验证失败：allowed_models 置空，err 为原因字符串
"""

import hmac
from typing import Optional

from . import config, model_names


def validate(headers) -> tuple[Optional[str], list[str], Optional[str]]:
    """验证请求头中的 API Key。

    headers: 类 dict，支持 `.get(key)`，key 大小写不敏感。

    返回:
      (key_name, allowed_models, None)  — 验证通过
      (None,     [],             err)   — 验证失败
    """
    auth_h = headers.get("authorization") or ""
    api_key = headers.get("x-api-key") or ""

    token = ""
    if auth_h.lower().startswith("bearer "):
        token = auth_h[7:].strip()
    elif api_key:
        token = api_key.strip()

    if not token:
        return None, [], "Missing API key"

    cfg = config.get()
    for name, entry in (cfg.get("apiKeys") or {}).items():
        if not isinstance(entry, dict):
            continue
        key_value = entry.get("key", "")
        if not key_value:
            continue
        if hmac.compare_digest(str(key_value), token):
            if entry.get("enabled") is False:
                return None, [], "API key is disabled"
            allowed = model_names.expand_legacy_permissions(list(entry.get("allowedModels") or []))
            return name, allowed, None

    return None, [], "Invalid API key"


def images_allowed(key_name: Optional[str]) -> bool:
    """该 Key 是否允许调用 Parrot 图片生成/编辑接口。默认 False。"""
    if not key_name:
        return False
    cfg = config.get()
    entry = (cfg.get("apiKeys") or {}).get(key_name)
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("allowImages", False))


def videos_allowed(key_name: Optional[str]) -> bool:
    """该 Key 是否允许调用 Parrot 视频生成/编辑接口。默认 False。"""
    if not key_name:
        return False
    cfg = config.get()
    entry = (cfg.get("apiKeys") or {}).get(key_name)
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("allowVideos", False))


def api_key_entry(key_name: Optional[str]) -> Optional[dict]:
    """该 Key 的原始配置项；不存在时返回 None。"""
    if not key_name:
        return None
    entry = (config.get().get("apiKeys") or {}).get(key_name)
    return entry if isinstance(entry, dict) else None


def allowed_channels(key_name: Optional[str], *, cfg: Optional[dict] = None) -> Optional[frozenset[str]]:
    """Exact source IDs permitted by this key; None means the legacy shared pool.

    Missing/empty arrays preserve existing behavior unless explicitly enabled.
    An explicit false switch retains the selection but uses the shared pool. A malformed nonempty
    binding fails closed, as does a binding to a removed account. Never resolve
    these IDs by model family, display name, affinity or provider prefix.
    """
    cfg = config.get() if cfg is None else cfg
    entry = (cfg.get("apiKeys") or {}).get(key_name)
    if not isinstance(entry, dict):
        return None
    enabled = entry.get("channelBindingEnabled")
    if enabled is False:
        return None
    if "channelBindingEnabled" in entry and not isinstance(enabled, bool):
        return frozenset()
    raw = entry.get("allowedChannels", [])
    if isinstance(raw, list) and not raw and enabled is not True:
        return None
    if not isinstance(raw, list) or any(not isinstance(v, str) or not v.strip() for v in raw):
        return frozenset()
    return frozenset(raw)


def channel_allowed(key_name: Optional[str], channel_key: str, *, cfg: Optional[dict] = None) -> bool:
    selected = allowed_channels(key_name, cfg=cfg)
    return selected is None or channel_key in selected


def mcp_allowed(key_name: Optional[str]) -> bool:
    """该 Key 是否允许访问 MCP 服务。默认 False，需显式开启。"""
    entry = api_key_entry(key_name)
    return bool(entry and entry.get("allowMcp", False))


def mcp_selected_tools(key_name: Optional[str]) -> list[str]:
    """该 Key 显式选择的 MCP 工具名；空列表 = 使用全局开关。"""
    entry = api_key_entry(key_name)
    if not entry:
        return []
    raw = entry.get("mcpTools")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str) and item]


def get_allowed_protocols(key_name: Optional[str]) -> list[str]:
    """Deprecated compatibility shim.

    API Keys no longer gate Anthropic/OpenAI protocol entrances.  Route safety is
    decided by ProtocolMatrix + provider capabilities; model access remains
    controlled by ``allowedModels`` returned from ``validate()``.
    """
    return []
