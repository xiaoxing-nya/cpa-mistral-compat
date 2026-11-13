# cpa-mistral-compat (v0.2.2)

CLIProxyAPI (CPA) 专用 Mistral **协议格式（Wire-Format）**双向兼容插件。

## 职责范围
1. **响应侧 Thinking 数组拆解（流式 SSE + 非流式 JSON）**：
   将 Mistral 返回的多态数组 `content: [{"type":"thinking",...}, {"type":"text",...}]` 还原为标准 OpenAI 的 `reasoning_content` + 纯字符串 `content`，并将 `finish_reason: "model_length"` 转为 `"length"`。兼容 Chat / Responses / Claude / Gemini 所有下游转换。
2. **请求侧字段映射与 `422 extra_forbidden` 清洗**：
   `seed` -> `random_seed`；剥离非 Mistral Chat 协议字段（`store`、`user`、`verbosity`、`reasoning`、`options`、非 tool 消息上的 `name`、content 块上的 `cache_control` 等）；仅保留 `stream_options.include_usage`。
3. **请求侧角色、历史思考、工具 ID 与 Schema 格式转换**：
   - `role: "developer"` -> `role: "system"`；
   - 历史 `assistant` 消息中的 `reasoning_content` / `reasoning` / `reasoning_details` -> 转换为 Mistral 原生 `{"type":"thinking","closed":true,...}` 内容块回传；
   - `tool_call_id` 确定性哈希归一化为 9 位字母数字 `[A-Za-z0-9]{9}`；
   - 修正 `assistant.prefix` 约束；
   - 递归将 Gemini 工具参数中的全大写类型（`OBJECT`/`STRING` 等）及 `nullable` 规范化为标准 JSON Schema。
4. **基于选中凭证的精准隔离（不与其他渠道打架）**：
   通过 `host.auth.list` + `host.auth.models` 识别 Mistral 凭证的 `id` / `auth_index`，并在 `request.intercept_after` 校验宿主传入的 `selected_auth_id` / `selected_auth_index`（`before` 阶段保持 `omit_body_before: true` 零拷贝）。非 Mistral 凭证（如 `cline-pass` 等）的请求哪怕模型名或别名完全相同也会零改动直接放行。
5. **不支持的推理档位自动使用上游默认值**：
   对已知受限模型，仅删除其不支持的 `reasoning_effort`（包含不应直接发往 Mistral 的 `auto`），让上游按模型默认档位推理；受支持的值与未知模型的值原样保留，绝不映射成其他档位。

## 推理档位支持矩阵（已知受限模型）
| 模型族 | 保留的档位 | 其他值（含 `auto`） |
|---|---|---|
| `*deepseek*` | `none`、`low`、`high`、`xhigh`、`max` | 剥离 `reasoning_effort`（使用上游默认） |
| `*glm-5?2*`（`zai-glm-5-2`、`glm-5-2`、`glm-5.2`） | `none`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max` | 剥离 `reasoning_effort`（使用上游默认） |
| `*glm-5?3*` / `*zai-glm*`（`zai-glm-5-3`、`zai-glm-5-3-flash`、`zai-glm-5`、`zai-glm-latest`） | `low`、`high`、`max` | 剥离 `reasoning_effort`（使用上游默认） |
| `mistral-large-4*`、`mistral-large-latest` | `low`、`medium`、`high` | 剥离 `reasoning_effort`（使用上游默认） |
| `mistral-small-2603*`、`mistral-small-4*`、`mistral-small-latest`、`mistral-medium*`、`magistral-*`、`mistral-vibe-*` | `none`、`high` | 剥离 `reasoning_effort`（使用上游默认） |
| `mistral-large*`（旧版 `2407`/`2411`/`2512`）、`codestral*`、`ministral*`、`mistral-code*` | *（不支持）* | 剥离 `reasoning_effort` |
| 其他未知模型 | 原样透传 | 原样透传 |

> **关于 CPA `model(effort)` 思考后缀：** CPA 宿主会在插件拦截与翻译之后执行 `ApplyRequestThinkingDoc` 解析 `model(medium)` 后缀；若对受限 Mistral 模型使用括号后缀语法，请在 CPA `config.yaml` 对应模型下同时声明 `thinking.levels`。

## 部署路径
- ARM64 容器：`<CPA>/plugins/linux/arm64/cpa-mistral-compat.so`
- AMD64 容器：`<CPA>/plugins/linux/amd64/cpa-mistral-compat.so`
