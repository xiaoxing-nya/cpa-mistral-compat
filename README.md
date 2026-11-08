# cpa-mistral-compat (v0.2.0)

CLIProxyAPI (CPA) 专用 Mistral **纯粹协议格式（Wire-Format）**双向兼容插件。

## 职责范围（仅处理协议格式差异，绝不篡改推理档位）
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
4. **轻量零拷贝自动模型发现**：
   启用 `schema_version: 7` 的 `request_interceptor_omit_body_before/after`（不拷贝请求体），仅通过 `host_callback_id` 查询 `host.auth.list` + `host.auth.models` 自动发现绑定在 `mistral` 渠道下的模型与 `alias`。
5. **不干涉推理档位**：
   `reasoning_effort` 的取值完全交由调用方、CPA 原生模型 `thinking` 定义及上游校验，插件绝不暗中改写档位。

## 部署路径
- ARM64 容器：`<CPA>/plugins/linux/arm64/cpa-mistral-compat.so`
- AMD64 容器：`<CPA>/plugins/linux/amd64/cpa-mistral-compat.so`
