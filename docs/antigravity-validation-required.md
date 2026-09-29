# Antigravity 账号「需完成 Google 账号验证」处理

Google 会不定期要求 Antigravity OAuth 账号重新验证身份。撞上这道墙时：

- 文本/图片请求返回 **HTTP 403**，报错是 `Verify your account to continue.`，`details[].reason = VALIDATION_REQUIRED`，`metadata.validation_url` 是一条一次性的 Google 登录续接链接；
- TG「🔐 管理 OAuth」账户详情的模型配额一行显示「需完成 Google 账号验证」（查额度接口返回同一个 403，Parrot 只做分类，不缓存链接）；
- **token 刷新照常成功**，账户不会被标 `auth_error`，所以「🧨 移除失效」列表里不会出现它。

这是 Google 侧的账号验证，不是 token 失效，也不是 Parrot 的问题。

## 不要做

- 删号重加、重新授权、覆盖导入、重启容器：都解不开这道墙。删号重加还会丢掉该账户的停用模型列表等账户级设置。
- 反复请求这个账户：每多一次 403，Google 那边就多一条风控记录。验证完成前可以先在 TG 里停用该账户（「🚫 停用」），让流量走其他账户。

## 1. 取验证链接（只读）

`scripts/antigravity_validation_link.py` 只读地打开 `logs/YYYY-MM.db`，按账户列出可完整提取的最新一条 `validation_url`（同一 URL 重复出现时保留最新时间），并把顶层查询参数 `authuser` 设为该账户邮箱，移除重复的同名参数，保留其他参数及 fragment 原样（不补的话，Google 会按浏览器里的默认账号去验证）。只用标准库，不联网、不读 config，以 SQLite `mode=ro` 打开，不修改业务记录。WAL 数据库读取时可能创建或更新 `-wal` / `-shm` 辅助文件，因此不保证目录中完全没有写入；同时仍会读取尚未 checkpoint 的最新日志。镜像里没有 `scripts/` 也能通过 stdin 运行：

```bash
# Docker（容器名以 docker ps 为准；在 Parrot 源码目录或下载了该脚本的目录执行）
docker exec -i parrot python3 - < scripts/antigravity_validation_link.py

# 只看某个账户 / 列出全部记录 / JSON 输出
docker exec -i parrot python3 - --email user@gmail.com < scripts/antigravity_validation_link.py
docker exec -i parrot python3 - --all --json < scripts/antigravity_validation_link.py

# 宿主机读 Docker 挂载的数据目录（一键脚本默认安装目录）
python3 scripts/antigravity_validation_link.py --data-dir /opt/parrot/data

# 源码部署：未设置数据目录环境变量时，数据默认在源码根目录
python3 scripts/antigravity_validation_link.py --data-dir /opt/parrot

# 自定义了 logDir 时直接指定实际日志目录
python3 scripts/antigravity_validation_link.py --log-dir /path/to/logs
```

说明：

- 不指定路径时，依次采用 `$ANTHROPIC_PROXY_DATA_DIR`、存在日志目录的 `/app/data`、脚本所在源码仓库的根目录；独立下载的脚本或 stdin 运行则以当前目录为准。脚本不读取 `config.json`，自定义 `logDir` 必须显式传 `--log-dir`。
- `--email` 将邮箱中的 `_`、`%` 等按字面匹配，不会把它们当作 SQL 通配符。
- `--months` 只接受非负整数，默认 2，`0` 表示全部月库；负数按参数错误退出，不会扫描日志。
- 退出码：`0` = 找到可完整提取的链接，`1` = 查询成功但没有可完整提取的链接，`2` = 参数错误、找不到日志库或读取失败。数据库损坏、锁冲突、权限或 I/O 错误会明确报错，不会提示「没有验证记录，发一次请求再试」。旧日志缺少某张表或字段时仍跳过该来源。
- 每次被拒 Google 都会发一条新的 `plt` 令牌，**只用最新那条**；旧链接用过就对不上了。
- 如果该账户在这条记录之后已有成功请求，脚本会提示「很可能已用过或不再需要」。这是按邮箱（可能跨项目）汇总的历史提示，不代表账号当前已恢复。
- 日志可能在 4000 字符处截断。URL 字符串已经完整、仅后面的 JSON 截断时仍可提取；URL 本身未结束时不会猜补或输出残缺链接。部分验证记录缺少完整链接时会向 stderr 警告，并说明展示的旧链接不保证对应最新验证要求；`--json` 的 stdout 仍保持 JSON 格式。空值或类型异常字段不会阻止继续检查其他链接候选。
- 日志里没有记录时脚本返回 1：账户刚被要求验证、但还没有请求落到它上面时，日志里不会有链接。给它发 1 次请求再运行。
- TG「📋 最近日志 → 💬 请求日志」的详情只显示错误摘要（`HTTP 403 — Verify your account to continue.`），可以用来确认是哪个账户撞墙，但看不到链接，链接要用脚本取。手动拼接时，把链接末尾的 `authuser` 改成 `authuser=邮箱`（`@` 写成 `%40`）。
- 这条链接能直接续接该账户的登录，**不要发到公开场合**。

## 2. 打开链接验证

实践中成功率最高的做法：

- 代理开全局，固定一个节点，与 Parrot 出口在同一地区，验证过程中不要切换；
- 用**无痕窗口，只登录这一个 Google 账号**（多账号同时登录时会按默认账号走，页面上切换不了）；
- 打开刚取出、还没用过的链接，按提示完成验证，直到页面显示成功。

以上任一条不满足（没开全局、多账号同登、复用旧链接），都可能失败。

## 3. 页面提示「我们无法验证您的信息」

反复点「重试」没用，说明验证方式本身没过：

- 按 Google 帮助页 <https://support.google.com/accounts?p=al_alert> 换一种方式：短信（号码可以不是账号绑定的那个）、另一个 Google 账号担保、政府证件。二维码/设备检查需要装有 Google Play 服务的安卓设备，iPhone 基本过不了。
- 在官方 Antigravity 桌面应用（<https://antigravity.google/download>，只有 macOS / Windows / Linux 版）里登录该账号并发一句话，从应用弹出的验证入口走（社区经验）。
- 每次重试前，先给该账户发 1 次请求拿到新链接，再回到第 1 步。

## 4. 验收

页面显示成功不等于已恢复。用只会路由到该账户的模型（或临时只启用这一个 Antigravity 账户），发 1 次 `POST /v1/chat/completions`（`max_tokens` ≥ 64）：

- 返回 200：已恢复，在 TG 里重新启用账户，点「📊 刷新额度」确认配额一行恢复正常；
- 仍是 403 `VALIDATION_REQUIRED`：回到第 1 步取新链接。

## 风险提示

Google 在识别反代使用的账号：验证会越来越频繁，最终可能被判定违反服务条款而禁用。实践中出现过验证通过十几分钟后查额度接口又要求验证的情况。连续失败时建议停用该账户 1–2 天，期间不发请求、不刷新额度，之后再走官方桌面应用验证。
