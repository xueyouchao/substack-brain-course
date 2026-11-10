# 模型层开放化改造 · 设计稿（Spec）

> 状态：**已实施完成**（分支 `spec/model-provider-config`，四张工单全部合入）。本文保留为设计与决策依据；使用说明见 `.env.example` 的 `# --- Model providers ---` 段与 `README.md` 的 `### Picking a model backend`。
> 决策来源：与你的访谈结论（见文末「已敲定决策」）。

## 1. 目标 / 非目标

**目标**
- 用 `provider:model` 前缀让现有 5 个 `*_MODEL` 设置可指向任意 provider，支持 Anthropic、OpenAI、**Ollama Cloud（含本地 Ollama）**、**OpenCode Go / Console**。
- 新增 provider 不需要改业务代码，只需在能力表加一条 + 配 env。
- 保持向后兼容：现有只配 `ANTHROPIC_API_KEY` 的 `.env` 零改动可跑。

**非目标（本次明确不做）**
- embeddings 层不动：`kb/sources/embedder.py` 继续走 OpenAI `text-embedding-3-small` / 1536 维。换 embedding 模型涉及 pgvector `vector(1536)` 表结构与全量重 embed，另立议题。
- 不支持 OpenAI **Responses API**。只做两条 API 族：`openai_chat`（Chat Completions）与 `anthropic_messages`（Messages）。因此 OpenCode Go 里仅走 Responses 的 GPT 系模型暂不可用，其余（GLM / Kimi / DeepSeek / Qwen / MiniMax / Claude Haiku 等）均可用。
- 不新增 YAML 注册表文件，不引入 LiteLLM 之类统一网关依赖。

## 2. 分层

```
业务调用点（17 处，全部零改动）
        │  call_llm(system_prompt=, user_input=, model="ollama:gemma4:31b", ...)
        ▼
kb/agents/providers.py
  ├─ resolve_model(ref) -> ResolvedModel(provider, model_id, spec)     ← 新增
  ├─ _structured_output_mode() 选档；三档分支落在各自 transport 里       ← 新增
  ├─ transport: AnthropicMessages / OpenAIChatCompletions             ← 新增
  └─ estimate_llm_cost_usd(provider, model_id, in, out, mode)          ← 改签名(内部)
        ▲
kb/config.py  Settings: 新增 default_provider + 各 provider 的 base_url/key/pricing 覆盖项
```

`call_llm()` 的公开签名**完全不变**，所以 20 处 `output_schema=` 与 17 处调用点都不用改。

## 3. 模型引用解析

格式：`[provider:]model_id`，**按第一个冒号切分且只切一次**。

```
"claude-haiku-5-5"        -> provider=DEFAULT_PROVIDER, model="claude-haiku-5-5"
"anthropic:claude-haiku-5-5" -> provider=anthropic,      model="claude-haiku-5-5"
"ollama:gemma4:31b"       -> provider=ollama,            model="gemma4:31b"   ← tag 里的冒号保留
"opencode:glm-5.3"        -> provider=opencode,          model="glm-5.3"
"openai:gpt-4o-mini"      -> provider=openai,            model="gpt-4o-mini"
```

规则：
1. 前缀**不在能力表里** → 视为模型名本身（如 `openai:gpt-4o` 的 `openai` 恰好在表里，正常路由；而 `gpt-4o` 无前缀则走默认 provider）。判定用「前缀 ∈ 能力表」而非「含冒号」。
2. provider 未知且形似前缀 → `ValueError`，错误信息列出已知 provider。
3. provider 已解析但缺 API key（云端 provider）→ 保留现有风格 `RuntimeError`，信息含 provider 与所需 env 变量名。

## 4. provider 能力表（代码内，无 YAML）

放在 `kb/agents/providers.py`，一个 `final` dict：

| provider | api_family | 默认 base_url | key env | structured 默认档 | pricing 默认 |
|---|---|---|---|---|---|
| `anthropic` | anthropic_messages | SDK 默认 | `ANTHROPIC_API_KEY` | `native` | `per_token` |
| `openai` | openai_chat | `https://api.openai.com/v1` | `OPENAI_API_KEY` | `native` | `per_token` |
| `ollama` | openai_chat | `https://ollama.com/v1` | `OLLAMA_API_KEY` | `json_mode` | `unknown` |
| `ollama-anthropic` | anthropic_messages | `https://ollama.com` | `OLLAMA_API_KEY` | `prompt_only` | `unknown` |
| `opencode` | openai_chat | `https://opencode.ai/inference/openai/v1` | `OPENCODE_API_KEY` | `prompt_only` | `subscription` |
| `opencode-anthropic` | anthropic_messages | `https://opencode.ai/inference/anthropic` | `OPENCODE_API_KEY` | `prompt_only` | `subscription` |

每条的字段：`api_family` / `base_url_setting` / `key_setting` / `structured_setting` / `pricing_setting` / `default_base_url` / `default_structured` / `default_pricing`。`*_setting` 存的是 `Settings` 属性名；为 `None` 表示该 provider 没有自己的 env 开关，改用 `default_*` 字段。

**为什么同一网关占两个条目**：OpenCode Console 与 Ollama 都同时暴露两种 API 族（`/inference/openai/v1` 与 `/inference/anthropic`、`ollama.com/v1` 与 `ollama.com`），而 OpenCode **按模型**分配族——Claude 系走 Anthropic Messages，GLM / Kimi / DeepSeek / Qwen 走 Chat Completions。把 `provider → api_family` 定成 1:1 就表达不了这种「同 key 不同族」。拆成两个条目后，选族这件事退化成选 provider 名，不需要维护模型→族映射表，也不需要运行时探测：

```
OPENCODE 模型（Chat Completions 族）：模型名 => opencode:glm-5.3
OPENCODE 模型（Anthropic Messages 族）：模型名 => opencode-anthropic:claude-haiku-5-5
```

两个条目共用 `OPENCODE_API_KEY`，所以用户只需一个 key；`OPENCODE_STRUCTURED_OUTPUT` / `OPENCODE_PRICING_MODE` 也同时作用于两个条目（实现上就是两行的 `*_setting` 指向同一批 env 开关，不需要额外的别名机制）。

**本地 Ollama**：无需新 provider，覆盖 base_url 即可
`OLLAMA_BASE_URL=http://localhost:11434/v1`，key 任意值（Ollama 忽略）。

**env 覆盖**（每 provider 一组，全部可缺省）。两条规则：

1. **空字符串环境变量等于「未设置」**，回落到字段默认值（纯空格同理）。因此 URL 类变量**不要**写成 `FOO=` 空赋值 —— 那会把真实默认值清空，`cp .env.example .env` 就会踩到。下表用注释形式给出规范值，需要改端点时才取消注释。
2. 默认值本来就是 `""` 的变量（两个 API key、`ANTHROPIC_BASE_URL`）留空赋值无害：它们没有默认值可丢。

```
DEFAULT_PROVIDER=anthropic                      # 无前缀时用哪个
# OLLAMA_BASE_URL=https://ollama.com/v1         # 本地 Ollama 改成 http://localhost:11434/v1
# OLLAMA_ANTHROPIC_BASE_URL=https://ollama.com  # ollama-anthropic 族，与 ollama 共用同一把 key
OLLAMA_API_KEY=                                 # 空赋值无害：key 无默认值可丢
OLLAMA_STRUCTURED_OUTPUT=json_mode              # native | json_mode | prompt_only
OLLAMA_PRICING_MODE=unknown                     # per_token | subscription | unknown
# OPENCODE_BASE_URL=https://opencode.ai/inference/openai/v1
# OPENCODE_ANTHROPIC_BASE_URL=https://opencode.ai/inference/anthropic
OPENCODE_API_KEY=                               # opencode 与 opencode-anthropic 共用
OPENCODE_STRUCTURED_OUTPUT=prompt_only
OPENCODE_PRICING_MODE=subscription
# OPENAI_BASE_URL=https://api.openai.com/v1     # 仅影响 openai provider 的对话调用；embeddings 仍直连 OpenAI
ANTHROPIC_BASE_URL=                             # 例外：默认即「让 SDK 自己选端点」，空赋值正确
```

第二族的两个 `*_ANTHROPIC_BASE_URL` 属于「同 key 不同端点」：`ollama` / `opencode` 读各自的开源族 URL，`ollama-anthropic` / `opencode-anthropic` 读 `*_ANTHROPIC_BASE_URL`，key 共用一份。

## 5. 传输层

两个轻量 adapter，输入同一组参数（system / user / model / max_tokens / schema）:

**AnthropicMessages**（现有行为不变）
- `anthropic.AsyncAnthropic(api_key=..., base_url=<可选>)`，走 `messages.create`。
- 有 schema 且档位 `native` → `output_config={format:{type:"json_schema", schema: anthropic.transform_schema(S)}}`。
- 解析：优先 `tool_use.input`，否则 text → `_extract_json_object` → `json.loads`。

**OpenAIChatCompletions**
- `openai.AsyncOpenAI(api_key=..., base_url=...)`，走 `chat.completions.create`。
- system 变成 `messages[0]={"role":"system"}`；user 是 `messages[1]`。
- `native` → `response_format={"type":"json_schema","json_schema":{"name":...,"schema":S.model_json_schema(),"strict":false}}`
- `json_mode` → `response_format={"type":"json_object"}` + schema 注入 prompt
- `prompt_only` → 只注入 prompt
- usage 取 `response.usage.prompt_tokens/completion_tokens`（映射到现有 input/output_tokens 字段名）。

`strict:false` 是有意的：Ollama / OpenCode 对 strict 模式支持不齐，留宽松以免 400。

## 6. 结构化输出三档

| 档位 | 机制 | 用于 |
|---|---|---|
| `native` | 厂商原生 json_schema | anthropic, openai |
| `json_mode` | `response_format=json_object` + schema 注入 prompt | ollama |
| `prompt_only` | 仅 schema 注入 prompt | opencode（默认） |

schema 注入文本统一由 `_schema_instruction(schema) -> str` 生成，附在 system_prompt 末尾：
> 只输出符合以下 JSON Schema 的单个 JSON 对象，不要包含解释或 markdown 围栏：`<schema json>`

兜底链条对三档**始终生效**：`_extract_json_object` 剥围栏 → `json.loads` → `output_schema.model_validate`。
解析失败时抛带 provider/model/原始片段前 200 字符的 `RuntimeError`（现在是无提示的 `json.JSONDecodeError`）。

## 7. 成本模型

`pricing` 三态，按 provider 默认 + env 覆盖，模型级价格仍用现有 `MODEL_PRICING_PER_MTOK` 表：

- `per_token` → 现有算法：`tokens/1e6 * price`；模型不在价格表 → 落回 `anthropic_*_price_per_million_tokens` 默认值（现行为）。
- `subscription` → `est_usd = 0.0`，`usage` 加 `pricing_mode:"subscription"`。
- `unknown` → `est_usd = 0.0`，`usage` 加 `pricing_mode:"unknown"`，并 `logger.warning` 一次（按 provider+model 去重，不刷屏）。

`usage` 字典保持 4 个必填键不变（`kb/inngest_client.py:43` 是子集判断，加键安全），新增可选键 `provider` / `pricing_mode`。

**必须在文档里写明的后果**：`kb/agents/core.py:118` 用 `est_usd` 累加来卡 `RESEARCH_MAX_USD`，订阅/未知档恒为 0 → 该预算在 Ollama/OpenCode 上**失效**。补偿控制：把 `RESEARCH_MAX_ITERATIONS`（现默认 3）作为主约束写进 `.env.example` 注释与 `docs/`，而非新增机制。

## 8. 改动清单

| 文件 | 改动 |
|---|---|
| `kb/config.py` | 加 `default_provider` 与 4 组 provider 覆盖项；5 个 `*_MODEL` 默认值不变 |
| `kb/agents/providers.py` | 加能力表 `PROVIDERS`、`resolve_model()`、`_schema_instruction()`、两个 adapter、`pricing_mode` 支持；`estimate_llm_cost_usd` 内部改签名；`_anthropic_client` 单例按 `(provider, base_url)` 分键 |
| `kb/functions/ask.py` | **删除** 私有 `_anthropic_client` / `get_anthropic_client` / `_synthesize` 里的直连调用，改走 `call_llm`；保留「失败返回错误字符串而非抛异常」的现有语义。**并补上 `persist_cost=True`**：`/ask` 不经 Inngest step（`kb/main.py` 直接 `await`），`CostCaptureMiddleware` 在 `result.step is None` 时直接 return，所以往响应里塞 `usage` 永远不会被记录，只有直写 costs 表这条路有效 |
| `.env.example` | 加 `DEFAULT_PROVIDER` 与 provider 覆盖项样例 + 订阅档预算失效的注释 |
| `docs/model-provider-config.md` | 本文件，确认后转正为使用说明 |
| `tests/unit/test_model_providers.py` | 新增（见 §9） |
| `tests/unit/test_config_settings.py` | 新增：配置面默认值、env 覆盖、`Literal` 拒绝非法档位、空值回退 |
| `tests/unit/test_ask_provider.py` | 新增：`SYNTHESIS_MODEL` 透传、`persist_cost`、成本落库、`/ask` 响应契约 |
| `README.md` | 配置章节补一段指向本文 |

**不改**：`kb/sources/embedder.py`、`kb/inngest_client.py`、17 处 `call_llm` 调用点、`kb/agents/core.py`、任何 alembic 迁移。

## 9. 测试计划

默认**不联网、不需要真 key**。

> **实现更正（T2 实证，本 spec 原方案有误）**：本仓 pin 的 `anthropic==1.7.0` / `openai==3.16.2` 走 `httpx2`，而 `respx` 只 patch `httpx` —— 用 respx 拦这两个 SDK 会**漏到真实网络**（探针实测打到了 api.anthropic.com 并拿到真 401）。且 `respx` 在本仓库的 Python 文件里**从未被使用**，只是 dev 依赖里躺着。因此实际测试改用：monkeypatch `providers.get_anthropic_client` / `get_openai_client` 注入记录型假 client 断言请求 kwargs，并直接断言真实 client 的 `base_url` / `api_key` 钉住路由。

1. `test_resolve_model`：无前缀→默认 provider；`ollama:gemma4:31b` 切成 `ollama`/`gemma4:31b`；`opencode:glm-5.3`；未知前缀抛出且错误信息含已知 provider 列表。
2. `test_anthropic_native_still_emits_output_config`：**向后兼容护栏** — 无前缀模型必须仍发 `output_config.format.type == "json_schema"`。
3. `test_openai_chat_native` / `test_ollama_json_mode` / `test_opencode_prompt_only`：注入假 openai client 捕获 `chat.completions.create` 的 kwargs，断言 `response_format` 三档差异与 schema 注入仅在非 native 档出现。
4. `test_base_url_and_auth_header`：Ollama → `https://ollama.com/v1` + `Bearer`；OpenCode → `https://opencode.ai/inference/openai/v1`。
5. `test_subscription_cost_is_zero_and_flagged`：`est_usd == 0.0` 且 `usage["pricing_mode"] == "subscription"`。
6. `test_usage_dict_shape`：4 个必填键恒在（守住 `CostCaptureMiddleware` 契约）。
7. `test_malformed_json_raises_with_context`：prompt_only 档返回带围栏/垃圾文本时的报错含 provider/model。
8. 回归：现有 `tests/unit/test_research_agent.py`、`tests/test_week4.py` 里 monkeypatch `call_llm` 的用例必须全绿（证明签名未破坏）。

## 10. 风险

| 风险 | 影响 | 处置 |
|---|---|---|
| Ollama/OpenCode 对 `json_schema` 支持不齐 | 结构化输出降档，JSON 合法率下降 | 三档 + 统一 schema 注入 + 解析失败带上下文 |
| 订阅档 `est_usd=0` | `RESEARCH_MAX_USD` 失效 | 文档写明，主约束改为 `RESEARCH_MAX_ITERATIONS` |
| OpenCode 免费档模型在非 OpenCode 客户端 403 | 用户误配后报错不明 | §3 错误信息带上「免费档模型仅限 OpenCode 内使用，请用 Go 订阅或付费模型」提示 |
| `anthropic.transform_schema` 是 SDK 内部函数 | 版本升级可能消失 | 仅 native 档使用；加一条注释说明来源，异常时降级到 prompt_only 并告警 |
| 一次请求里 provider 混用（synthesis 用 ollama、judges 用 anthropic）| 成本表混入 0 值 | 已由 `pricing_mode` 字段区分，报表可按它过滤 |

## 11. 已敲定决策（访谈结论）

1. 范围：只做对话/LLM 层，embeddings 保持 OpenAI。
2. 配置面：env `provider:model` 前缀，不建 YAML 注册表。
3. 兼容：无前缀 → `DEFAULT_PROVIDER`，默认 `anthropic`。
4. 结构化输出：三档 native / json_mode / prompt_only，按 provider 自动降级。
5. Responses API：暂不实现，只做 Chat Completions + Anthropic Messages。
6. 成本：provider 标 per_token / subscription / unknown，订阅档 `est_usd=0.0` 并打标。
7. 元数据落点：代码内能力表 + env 覆盖。
8. 交付：先出 spec，确认后才动代码。

## 12. 确认后的实施顺序

1. `config.py` 加配置项（无行为变更，现有测试应全绿）
2. `providers.py` 加能力表 + `resolve_model` + 两个 adapter + 三档结构化输出
3. 新增 `tests/unit/test_model_providers.py`（先红后绿）；`test_config_settings.py` 随第 1 步、`test_ask_provider.py` 随第 4 步
4. `ask.py` 收敛去重
5. `.env.example` / `README.md` / `docs/` 收尾
6. `uv run pytest` + `ruff check` + `mypy kb scripts tests` 全绿
