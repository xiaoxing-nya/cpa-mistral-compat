package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"
)

func TestTransformMistralChatRequest_PureWireFormat(t *testing.T) {
	cfg := DefaultConfig()
	raw := []byte(`{
		"model": "mistral-large-4",
		"store": false,
		"seed": 12345,
		"user": "user_99",
		"verbosity": "medium",
		"reasoning_effort": "medium",
		"prompt_mode": "reasoning",
		"stream_options": {"include_usage": true, "include_obfuscation": true},
		"messages": [
			{"role": "developer", "content": "You are a helpful assistant.", "name": "dev"},
			{"role": "user", "name": "alice", "content": [{"type": "text", "text": "What is 1+1?", "cache_control": {"type": "ephemeral"}}]},
			{
				"role": "assistant",
				"content": "",
				"reasoning_content": "Let me call the calc tool.",
				"prefix": true,
				"tool_calls": [
					{
						"id": "call_abc_123456_long_id",
						"type": "function",
						"function": {"name": "calc", "arguments": "{\"expr\":\"1+1\"}"},
						"extra_content": {"google": {"thought_signature": "sig"}}
					}
				]
			},
			{"role": "tool", "tool_call_id": "call_abc_123456_long_id", "name": "calc", "content": "2"},
			{"role": "assistant", "content": "The answer is "}
		],
		"tools": [
			{
				"type": "function",
				"cache_control": {"type": "ephemeral"},
				"function": {
					"name": "calc",
					"description": "calculator",
					"parameters": {"type": "OBJECT", "properties": {"expr": {"type": "STRING", "nullable": true}}},
					"strict": false,
					"extra_field": 1
				}
			},
			{
				"type": "web_search_preview"
			}
		]
	}`)

	out, changed := TransformMistralChatRequest(raw, "", cfg)
	if !changed {
		t.Fatal("expected request to be changed")
	}

	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal output: %v", err)
	}

	for _, forbidden := range []string{"store", "seed", "user", "verbosity", "prompt_mode"} {
		if _, exists := m[forbidden]; exists {
			t.Fatalf("forbidden field %q still exists in output: %s", forbidden, out)
		}
	}
	if m["random_seed"] != float64(12345) {
		t.Fatalf("random_seed = %v, want 12345", m["random_seed"])
	}
	// reasoning_effort MUST remain untouched ("medium")!
	if m["reasoning_effort"] != "medium" {
		t.Fatalf("reasoning_effort = %v, want untouched 'medium'", m["reasoning_effort"])
	}

	// Verify Gemini uppercase schema normalization
	fnParams := m["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)["parameters"].(map[string]any)
	if fnParams["type"] != "object" {
		t.Fatalf("parameters.type = %v, want object", fnParams["type"])
	}

	msgs := m["messages"].([]any)
	if len(msgs) != 5 {
		t.Fatalf("messages len = %d, want 5", len(msgs))
	}
	if msgs[0].(map[string]any)["role"] != "system" {
		t.Fatalf("msg0 role = %v, want system", msgs[0].(map[string]any)["role"])
	}
	msg2 := msgs[2].(map[string]any)
	if _, hasRC := msg2["reasoning_content"]; hasRC {
		t.Fatal("msg2 still has reasoning_content")
	}
	tcID := msg2["tool_calls"].([]any)[0].(map[string]any)["id"].(string)
	if !validMistralToolID.MatchString(tcID) {
		t.Fatalf("tool_call id = %q, want 9-char alphanumeric", tcID)
	}
	if msgs[3].(map[string]any)["tool_call_id"] != tcID {
		t.Fatal("tool_call_id mismatch")
	}
	if msgs[4].(map[string]any)["prefix"] != true {
		t.Fatal("trailing assistant missing prefix: true")
	}
}

func TestTransformMistralChatRequest_UnsupportedReasoningEffortUsesUpstreamDefault(t *testing.T) {
	tests := []struct {
		name          string
		model         string
		envelopeModel string
		effort        string
		wantPresent   bool
	}{
		{name: "DeepSeek medium falls back to model default", model: "deepseek-v4-1-flash-1017", effort: "medium", wantPresent: false},
		{name: "DeepSeek auto also falls back instead of sending invalid wire value", model: "deepseek-v4.1-flash", effort: "auto", wantPresent: false},
		{name: "DeepSeek supported high stays untouched", model: "deepseek-v4-1-flash-1017", effort: "high", wantPresent: true},
		{name: "DeepSeek supported xhigh stays untouched", model: "deepseek-v4-1-flash-1017", effort: "xhigh", wantPresent: true},
		{name: "GLM 5.3 supported max stays untouched", model: "zai-glm-5-3", effort: "max", wantPresent: true},
		{name: "GLM 5.3 supported low stays untouched", model: "glm-5.3", effort: "low", wantPresent: true},
		{name: "GLM 5.3 none is unsupported and falls back", model: "zai-glm-5-3", effort: "none", wantPresent: false},
		{name: "GLM 5.3 medium is unsupported and falls back", model: "glm-5.3-flash", effort: "medium", wantPresent: false},
		{name: "GLM 5.3 Flash supported low stays untouched", model: "zai-glm-5-3-flash", effort: "low", wantPresent: true},
		{name: "GLM 5.3 Flash supported max stays untouched", model: "zai-glm-5-3-flash", effort: "max", wantPresent: true},
		{name: "GLM 5.3 Flash none is unsupported and falls back", model: "zai-glm-5-3-flash", effort: "none", wantPresent: false},
		{name: "GLM 5.2 zai-id supported none stays untouched", model: "zai-glm-5-2", effort: "none", wantPresent: true},
		{name: "GLM 5.2 zai-id supported high stays untouched", model: "zai-glm-5-2", effort: "high", wantPresent: true},
		{name: "GLM 5.2 zai-id supported max stays untouched", model: "zai-glm-5-2", effort: "max", wantPresent: true},
		{name: "GLM 5.2 zai-id low is now supported and kept", model: "zai-glm-5-2", effort: "low", wantPresent: true},
		{name: "GLM 5.2 zai-id medium is now supported and kept", model: "zai-glm-5-2", effort: "medium", wantPresent: true},
		{name: "GLM 5.2 alias-id supported none stays untouched", model: "glm-5-2", effort: "none", wantPresent: true},
		{name: "GLM 5.2 alias-id low is now supported and kept", model: "glm-5-2", effort: "low", wantPresent: true},
		{name: "GLM 5.2 dot-alias supported high stays untouched", model: "glm-5.2", effort: "high", wantPresent: true},
		{name: "GLM latest alias supported low stays untouched", model: "zai-glm-latest", effort: "low", wantPresent: true},
		{name: "GLM latest alias none is unsupported and falls back", model: "zai-glm-latest", effort: "none", wantPresent: false},
		{name: "GLM 5 alias supported high stays untouched", model: "zai-glm-5", effort: "high", wantPresent: true},
		{name: "Mistral Large 4 supported medium stays untouched", model: "mistral-large-4", effort: "medium", wantPresent: true},
		{name: "Mistral Large latest maps to Large 4 and keeps medium", model: "mistral-large-latest", effort: "medium", wantPresent: true},
		{name: "Legacy Mistral Large 2512 drops any effort", model: "mistral-large-2512", effort: "high", wantPresent: false},
		{name: "Legacy Mistral Large 2411 drops any effort", model: "mistral-large-2411", effort: "low", wantPresent: false},
		{name: "Mistral Small 4 supported high stays untouched", model: "mistral-small-2603", effort: "high", wantPresent: true},
		{name: "Mistral Medium 3.5 unsupported medium falls back", model: "mistral-medium-3-5", effort: "medium", wantPresent: false},
		{name: "unknown model capabilities stay untouched", model: "future-model", effort: "medium", wantPresent: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw, err := json.Marshal(map[string]any{
				"model": tt.model, "reasoning_effort": tt.effort,
				"messages": []any{map[string]any{"role": "user", "content": "hi"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			out, changed := TransformMistralChatRequest(raw, tt.envelopeModel, DefaultConfig())
			var got map[string]any
			if err := json.Unmarshal(out, &got); err != nil {
				t.Fatalf("unmarshal output: %v", err)
			}
			value, present := got["reasoning_effort"]
			if present != tt.wantPresent {
				t.Fatalf("reasoning_effort presence = %v (value %v), want %v; output: %s", present, value, tt.wantPresent, out)
			}
			if present && value != tt.effort {
				t.Fatalf("reasoning_effort = %v, want untouched %q", value, tt.effort)
			}
			if changed == tt.wantPresent {
				t.Fatalf("changed = %v, want %v", changed, !tt.wantPresent)
			}
		})
	}
}

func TestTransformMistralChatRequest_AliasFallsBackToEnvelopeModel(t *testing.T) {
	tests := []struct {
		name          string
		bodyModel     string
		envelopeModel string
		effort        string
		wantPresent   bool
	}{
		{name: "custom alias + GLM 5.3 upstream: none should drop", bodyModel: "my-custom-glm", envelopeModel: "zai-glm-5-3", effort: "none", wantPresent: false},
		{name: "custom alias + GLM 5.3 upstream: low should keep", bodyModel: "my-custom-glm", envelopeModel: "zai-glm-5-3", effort: "low", wantPresent: true},
		{name: "custom alias + GLM 5.2 upstream: auto should drop", bodyModel: "vibe-glm", envelopeModel: "zai-glm-5-2", effort: "auto", wantPresent: false},
		{name: "custom alias + GLM 5.2 upstream: none should keep", bodyModel: "vibe-glm", envelopeModel: "zai-glm-5-2", effort: "none", wantPresent: true},
		{name: "custom alias + DeepSeek upstream: medium should drop", bodyModel: "ds-flash", envelopeModel: "deepseek-v4-1-flash-1017", effort: "medium", wantPresent: false},
		{name: "custom alias + DeepSeek upstream: xhigh should keep", bodyModel: "ds-flash", envelopeModel: "deepseek-v4-1-flash-1017", effort: "xhigh", wantPresent: true},
		{name: "custom alias + legacy Mistral Large 2512: any effort should drop", bodyModel: "big-model", envelopeModel: "mistral-large-2512", effort: "high", wantPresent: false},
		{name: "body model matches pattern: envelope model ignored", bodyModel: "zai-glm-5-3", envelopeModel: "some-other-model", effort: "none", wantPresent: false},
		{name: "both models unknown: pass-through", bodyModel: "custom-a", envelopeModel: "custom-b", effort: "medium", wantPresent: true},
		{name: "empty envelope model: body model only", bodyModel: "zai-glm-5-3", envelopeModel: "", effort: "none", wantPresent: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw, err := json.Marshal(map[string]any{
				"model": tt.bodyModel, "reasoning_effort": tt.effort,
				"messages": []any{map[string]any{"role": "user", "content": "hi"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			out, changed := TransformMistralChatRequest(raw, tt.envelopeModel, DefaultConfig())
			var got map[string]any
			if err := json.Unmarshal(out, &got); err != nil {
				t.Fatalf("unmarshal output: %v", err)
			}
			value, present := got["reasoning_effort"]
			if present != tt.wantPresent {
				t.Fatalf("reasoning_effort presence = %v (value %v), want %v; output: %s", present, value, tt.wantPresent, out)
			}
			if present && value != tt.effort {
				t.Fatalf("reasoning_effort = %v, want untouched %q", value, tt.effort)
			}
			_ = changed
		})
	}
}

func TestTransformMistralResponse_StreamAndNonStream(t *testing.T) {
	nonStream := []byte(`{
		"id": "cmpl-1",
		"object": "chat.completion",
		"choices": [{
			"index": 0,
			"finish_reason": "model_length",
			"message": {
				"role": "assistant",
				"content": [
					{"type": "thinking", "thinking": [{"type": "text", "text": "Step 1. "}, {"type": "text", "text": "Step 2."}], "closed": true},
					{"type": "text", "text": "Final answer."}
				]
			}
		}]
	}`)
	out, chg := TransformMistralResponse(nonStream, false)
	if !chg {
		t.Fatal("expected non-stream response to change")
	}
	var parsed map[string]any
	_ = json.Unmarshal(out, &parsed)
	choice := parsed["choices"].([]any)[0].(map[string]any)
	if choice["finish_reason"] != "length" {
		t.Fatalf("finish_reason = %v, want length", choice["finish_reason"])
	}
	msg := choice["message"].(map[string]any)
	if msg["content"] != "Final answer." || msg["reasoning_content"] != "Step 1. Step 2." {
		t.Fatalf("unexpected message: %#v", msg)
	}

	sseThinkOnly := []byte("data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"thinking\",\"thinking\":[{\"type\":\"text\",\"text\":\"thinking chunk\"}]}]},\"finish_reason\":null}]}\n\n")
	outSSE, chgSSE := TransformMistralResponse(sseThinkOnly, true)
	if !chgSSE || !bytes.HasPrefix(outSSE, []byte("data: ")) {
		t.Fatalf("unexpected SSE output: %s", outSSE)
	}
}

func TestTransformMistralChatRequest_EnvelopeModelOverridesConflictingAlias(t *testing.T) {
	// Client aliases GLM 5.3 (supports low/high/max, NOT none) as "glm-5.2" (which allows none).
	// Upstream envelope model MUST take priority over client alias.
	raw := []byte(`{"model":"glm-5.2","reasoning_effort":"none","messages":[{"role":"user","content":"hi"}]}`)
	out, changed := TransformMistralChatRequest(raw, "zai-glm-5-3", DefaultConfig())
	if !changed {
		t.Fatal("expected request to be changed")
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, hasRE := got["reasoning_effort"]; hasRE {
		t.Fatalf("expected none reasoning_effort to be stripped for zai-glm-5-3, got %s", out)
	}
}

func TestStripUnsupportedSourceThinking(t *testing.T) {
	tests := []struct {
		name           string
		sourceFormat   string
		model          string
		requestedModel string
		body           string
		wantChanged    bool
		check          func(t *testing.T, got map[string]any)
	}{
		{
			name:         "responses unsupported medium stripped",
			sourceFormat: "openai-response",
			model:        "zai-glm-5-3",
			body:         `{"model":"glm-5.3","reasoning":{"effort":"medium","summary":"auto"},"input":"hi"}`,
			wantChanged:  true,
			check: func(t *testing.T, got map[string]any) {
				r, _ := got["reasoning"].(map[string]any)
				if _, hasEffort := r["effort"]; hasEffort {
					t.Fatalf("expected reasoning.effort to be stripped: %#v", got)
				}
				if r["summary"] != "auto" {
					t.Fatalf("expected reasoning.summary preserved: %#v", r)
				}
			},
		},
		{
			name:         "responses supported high preserved",
			sourceFormat: "openai-response",
			model:        "zai-glm-5-3",
			body:         `{"model":"glm-5.3","reasoning":{"effort":"high"},"input":"hi"}`,
			wantChanged:  false,
			check: func(t *testing.T, got map[string]any) {
				r, _ := got["reasoning"].(map[string]any)
				if r["effort"] != "high" {
					t.Fatalf("expected reasoning.effort=high preserved: %#v", got)
				}
			},
		},
		{
			name:         "claude budget 8192 maps to unsupported medium and is stripped on glm-5.3",
			sourceFormat: "claude",
			model:        "zai-glm-5-3",
			body:         `{"model":"glm-5.3","thinking":{"type":"enabled","budget_tokens":8192},"messages":[]}`,
			wantChanged:  true,
			check: func(t *testing.T, got map[string]any) {
				if _, has := got["thinking"]; has {
					t.Fatalf("expected thinking stripped: %#v", got)
				}
			},
		},
		{
			name:         "claude disabled maps to none which is unsupported on glm-5.3",
			sourceFormat: "claude",
			model:        "zai-glm-5-3",
			body:         `{"model":"glm-5.3","thinking":{"type":"disabled"},"messages":[]}`,
			wantChanged:  true,
			check: func(t *testing.T, got map[string]any) {
				if _, has := got["thinking"]; has {
					t.Fatalf("expected thinking stripped: %#v", got)
				}
			},
		},
		{
			name:         "claude disabled is preserved on glm-5.2",
			sourceFormat: "claude",
			model:        "zai-glm-5-2",
			body:         `{"model":"glm-5.2","thinking":{"type":"disabled"},"messages":[]}`,
			wantChanged:  false,
			check: func(t *testing.T, got map[string]any) {
				th, _ := got["thinking"].(map[string]any)
				if th["type"] != "disabled" {
					t.Fatalf("expected thinking.type=disabled preserved: %#v", got)
				}
			},
		},
		{
			name:         "gemini unsupported medium thinkingLevel stripped on glm-5.3",
			sourceFormat: "gemini",
			model:        "zai-glm-5-3",
			body:         `{"generationConfig":{"thinkingConfig":{"thinkingLevel":"medium","includeThoughts":true}},"contents":[]}`,
			wantChanged:  true,
			check: func(t *testing.T, got map[string]any) {
				gc, _ := got["generationConfig"].(map[string]any)
				tc, _ := gc["thinkingConfig"].(map[string]any)
				if _, has := tc["thinkingLevel"]; has {
					t.Fatalf("expected thinkingLevel stripped: %#v", got)
				}
				if tc["includeThoughts"] != true {
					t.Fatalf("expected includeThoughts preserved: %#v", tc)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, changed := StripUnsupportedSourceThinking([]byte(tt.body), tt.sourceFormat, tt.model, tt.requestedModel)
			if changed != tt.wantChanged {
				t.Fatalf("changed = %v, want %v; out=%s", changed, tt.wantChanged, out)
			}
			var got map[string]any
			if err := json.Unmarshal(out, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			tt.check(t, got)
		})
	}
}

func TestTransformMistralResponse_PreservesUnknownMultimodalParts(t *testing.T) {
	nonMistral := []byte(`{
		"id": "cmpl-2",
		"object": "chat.completion",
		"choices": [{
			"index": 0,
			"finish_reason": "stop",
			"message": {
				"role": "assistant",
				"content": [
					{"type": "text", "text": "Here is the image:"},
					{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
				]
			}
		}]
	}`)
	out, changed := TransformMistralResponse(nonMistral, false)
	if changed {
		t.Fatalf("expected non-Mistral multimodal response to stay untouched, got: %s", out)
	}
}

func TestStripUnsupportedSourceThinking_FallsBackToBodyModelWhenArgsEmpty(t *testing.T) {
	body := []byte(`{"model":"zai-glm-5-3","reasoning":{"effort":"medium"},"input":"hi"}`)
	out, changed := StripUnsupportedSourceThinking(body, "openai-response", "", "")
	if !changed {
		t.Fatalf("expected effort stripped using body.model fallback, got unchanged: %s", out)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if _, hasR := got["reasoning"]; hasR {
		t.Fatalf("expected reasoning stripped, got: %s", out)
	}
}

func TestTransformMistralChatRequest_CaseInsensitiveSupportedReasoningEffort(t *testing.T) {
	raw := []byte(`{"model":"zai-glm-5-3","reasoning_effort":" High ","messages":[{"role":"user","content":"hi"}]}`)
	out, changed := TransformMistralChatRequest(raw, "zai-glm-5-3", DefaultConfig())
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if got["reasoning_effort"] != "high" {
		t.Fatalf("expected normalized supported effort 'high', got %v (changed=%v, out=%s)", got["reasoning_effort"], changed, out)
	}
}

func TestTransformMistralChatRequest_DoesNotDuplicateExistingThinkingBlockInContentArray(t *testing.T) {
	raw := []byte(`{
		"model": "zai-glm-5-3",
		"messages": [
			{
				"role": "assistant",
				"reasoning_content": "Step 1.",
				"content": [
					{"type": "thinking", "closed": true, "thinking": [{"type": "text", "text": "Step 1."}]},
					{"type": "text", "text": "Answer"}
				]
			},
			{"role": "user", "content": "Next?"}
		]
	}`)
	out, changed := TransformMistralChatRequest(raw, "zai-glm-5-3", DefaultConfig())
	if !changed {
		t.Fatal("expected request to change (reasoning_content removed)")
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	msg0 := got["messages"].([]any)[0].(map[string]any)
	parts := msg0["content"].([]any)
	thinkingCount := 0
	for _, p := range parts {
		if pm, ok := p.(map[string]any); ok && pm["type"] == "thinking" {
			thinkingCount++
		}
	}
	if thinkingCount != 1 {
		t.Fatalf("expected exactly 1 thinking block, got %d: %s", thinkingCount, out)
	}
}

func TestTransformMistralChatRequest_MistralCatalogEffortCoverage(t *testing.T) {
	cases := []struct {
		name       string
		model      string
		effort     string
		wantKept   bool
		wantEffort string
	}{
		// GLM 5.2 now supports none/minimal/low/medium/high/xhigh/max, only rejects auto.
		{"GLM 5.2 low kept", "zai-glm-5-2", "low", true, "low"},
		{"GLM 5.2 medium kept", "glm-5-2", "medium", true, "medium"},
		{"GLM 5.2 xhigh kept", "zai-glm-5-2", "xhigh", true, "xhigh"},
		{"GLM 5.2 auto dropped", "zai-glm-5-2", "auto", false, ""},
		// Mistral Medium 3.5 (dot & dash) & latest & Magistral & Vibe: none / high only.
		{"Mistral Medium 3.5 dot medium dropped", "mistral-medium-3.5", "medium", false, ""},
		{"Mistral Medium 3.5 dot high kept", "mistral-medium-3.5", "high", true, "high"},
		{"Mistral Medium latest low dropped", "mistral-medium-latest", "low", false, ""},
		{"Mistral Medium 2604 none kept", "mistral-medium-2604", "none", true, "none"},
		{"Mistral Small latest medium dropped", "mistral-small-latest", "medium", false, ""},
		{"Magistral Medium latest medium dropped", "magistral-medium-latest", "medium", false, ""},
		{"Magistral Small latest high kept", "magistral-small-latest", "high", true, "high"},
		{"Mistral Vibe CLI fast low dropped", "mistral-vibe-cli-fast", "low", false, ""},
		// Codestral / Ministral / Mistral-Code: reasoning_effort disabled.
		{"Codestral high dropped", "codestral-latest", "high", false, ""},
		{"Ministral 14b medium dropped", "ministral-14b-latest", "medium", false, ""},
		{"Mistral Code none dropped", "mistral-code-latest", "none", false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(fmt.Sprintf(`{"model":%q,"reasoning_effort":%q,"messages":[{"role":"user","content":"hi"}]}`, tc.model, tc.effort))
			out, _ := TransformMistralChatRequest(raw, tc.model, DefaultConfig())
			var got map[string]any
			if err := json.Unmarshal(out, &got); err != nil {
				t.Fatal(err)
			}
			val, exists := got["reasoning_effort"]
			if tc.wantKept {
				if !exists || val != tc.wantEffort {
					t.Fatalf("expected reasoning_effort=%q kept, got exists=%v val=%v out=%s", tc.wantEffort, exists, val, out)
				}
			} else if exists {
				t.Fatalf("expected reasoning_effort stripped, got %v out=%s", val, out)
			}
		})
	}
}

func TestTransformMistralChatRequest_SanitizesRealWorld422Hazards(t *testing.T) {
	raw := []byte(`{
		"model": "zai-glm-5-3",
		"max_tokens": 32,
		"max_completion_tokens": null,
		"stream_options": null,
		"response_format": null,
		"parallel_tool_calls": null,
		"tool_choice": null,
		"service_tier": "default",
		"truncation": "auto",
		"background": false,
		"previous_response_id": "resp_123",
		"max_output_tokens": 64,
		"instructions": "be brief",
		"include": ["reasoning.encrypted_content"],
		"conversation": "conv_1",
		"prompt": {"id": "p1"},
		"input": "hi",
		"text": {"format": {"type": "text"}},
		"audio": {"voice": "alloy", "format": "wav"},
		"modalities": ["text"],
		"tools": [{
			"type": "function",
			"function": {
				"name": "ping",
				"description": null,
				"strict": null,
				"parameters": null
			}
		}],
		"messages": [
			{"role": "system", "content": null},
			{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
		]
	}`)
	out, changed := TransformMistralChatRequest(raw, "zai-glm-5-3", DefaultConfig())
	if !changed {
		t.Fatal("expected request to be sanitized")
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	for _, mustDrop := range []string{
		"max_completion_tokens", "stream_options", "response_format", "parallel_tool_calls",
		"tool_choice", "service_tier", "truncation", "background", "previous_response_id",
		"max_output_tokens", "instructions", "include", "conversation", "prompt", "input", "text",
	} {
		if _, exists := got[mustDrop]; exists {
			t.Fatalf("expected %q removed, out=%s", mustDrop, out)
		}
	}
	if _, hasAudio := got["audio"]; !hasAudio {
		t.Fatalf("expected valid Mistral audio field preserved, out=%s", out)
	}
	if _, hasMod := got["modalities"]; !hasMod {
		t.Fatalf("expected valid Mistral modalities field preserved, out=%s", out)
	}
	tools := got["tools"].([]any)
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	if _, hasDesc := fn["description"]; hasDesc {
		t.Fatalf("expected null description removed: %s", out)
	}
	if _, hasStrict := fn["strict"]; hasStrict {
		t.Fatalf("expected null strict removed: %s", out)
	}
	if _, hasParams := fn["parameters"].(map[string]any); !hasParams {
		t.Fatalf("expected default parameters object synthesized when null/missing: %s", out)
	}
	msgs := got["messages"].([]any)
	if msgs[0].(map[string]any)["content"] != "" {
		t.Fatalf("expected system null content coerced to empty string: %s", out)
	}
	uParts := msgs[1].(map[string]any)["content"].([]any)
	if uParts[0].(map[string]any)["type"] != "text" {
		t.Fatalf("expected input_text normalized to text: %s", out)
	}
}

func TestStripUnsupportedSourceThinking_GeminiSnakeCaseAndClaudeMixedEffort(t *testing.T) {
	// 1. Gemini snake_case generation_config.thinking_config
	gBody := []byte(`{"generation_config":{"thinking_config":{"thinking_level":"medium"}},"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`)
	gOut, gChg := StripUnsupportedSourceThinking(gBody, "gemini", "zai-glm-5-3", "m-glm53")
	if !gChg {
		t.Fatalf("expected snake_case gemini thinking_level stripped: %s", gOut)
	}
	var gMap map[string]any
	_ = json.Unmarshal(gOut, &gMap)
	gc := gMap["generation_config"].(map[string]any)
	if _, ok := gc["thinking_config"]; ok {
		t.Fatalf("expected thinking_config removed: %s", gOut)
	}

	// 2. Claude with supported thinking budget (16000 -> high) but unsupported output_config.effort ("medium")
	cBody := []byte(`{"thinking":{"type":"enabled","budget_tokens":16000},"output_config":{"effort":"medium"},"messages":[{"role":"user","content":"hi"}]}`)
	cOut, cChg := StripUnsupportedSourceThinking(cBody, "claude", "zai-glm-5-3", "m-glm53")
	if !cChg {
		t.Fatalf("expected unsupported output_config.effort stripped while keeping supported thinking: %s", cOut)
	}
	var cMap map[string]any
	_ = json.Unmarshal(cOut, &cMap)
	if _, hasOC := cMap["output_config"]; hasOC {
		t.Fatalf("expected unsupported output_config removed: %s", cOut)
	}
	if _, hasThinking := cMap["thinking"]; !hasThinking {
		t.Fatalf("expected supported thinking preserved: %s", cOut)
	}
}

func TestTransformMistralResponse_PrettyPrintedMultilineJSON(t *testing.T) {
	raw := []byte("{\n  \"id\": \"1\",\n  \"choices\": [\n    {\n      \"index\": 0,\n      \"finish_reason\": \"stop\",\n      \"message\": {\n        \"role\": \"assistant\",\n        \"content\":\n        [\n          {\"type\": \"thinking\", \"thinking\": [{\"type\": \"text\", \"text\": \"step\"}]},\n          {\"type\": \"text\", \"text\": \"done\"}\n        ]\n      }\n    }\n  ]\n}")
	out, chg := TransformMistralResponse(raw, false)
	if !chg {
		t.Fatalf("expected multiline pretty-printed content array to be transformed")
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	msg := got["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "done" || msg["reasoning_content"] != "step" {
		t.Fatalf("unexpected message: %s", out)
	}
}

func TestTransformMistralChatRequest_StructuredOutputSchemaAndFileToDocumentURL(t *testing.T) {
	raw := []byte(`{
		"model": "mistral-large-4",
		"response_format": {
			"type": "json_schema",
			"json_schema": {
				"name": "answer",
				"strict": null,
				"schema": {
					"type": "OBJECT",
					"properties": {
						"ok": {"type": "BOOLEAN"},
						"note": {"type": "STRING", "nullable": true}
					},
					"required": ["ok"]
				}
			}
		},
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "Read this PDF"},
					{"type": "file", "file": {"filename": "doc.pdf", "file_data": "data:application/pdf;base64,JVBERi0xLjQK"}},
					{"type": "file", "file": {"file_url": "https://example.com/a.pdf"}}
				]
			}
		]
	}`)
	out, changed := TransformMistralChatRequest(raw, "mistral-large-4", DefaultConfig())
	if !changed {
		t.Fatal("expected request to be changed")
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	js := got["response_format"].(map[string]any)["json_schema"].(map[string]any)
	if _, hasStrict := js["strict"]; hasStrict {
		t.Fatalf("expected null strict removed from response_format.json_schema: %s", out)
	}
	schema := js["schema"].(map[string]any)
	if schema["type"] != "object" {
		t.Fatalf("expected uppercase schema.type normalized to 'object', got %v", schema["type"])
	}
	props := schema["properties"].(map[string]any)
	if props["ok"].(map[string]any)["type"] != "boolean" {
		t.Fatalf("expected ok.type='boolean', got %v", props["ok"])
	}
	noteTypes, ok := props["note"].(map[string]any)["type"].([]any)
	if !ok || len(noteTypes) != 2 || noteTypes[0] != "string" || noteTypes[1] != "null" {
		t.Fatalf("expected nullable note normalized to [string null], got %v", props["note"])
	}

	parts := got["messages"].([]any)[0].(map[string]any)["content"].([]any)
	p1 := parts[1].(map[string]any)
	if p1["type"] != "document_url" || p1["document_url"] != "data:application/pdf;base64,JVBERi0xLjQK" || p1["document_name"] != "doc.pdf" {
		t.Fatalf("expected file part converted to document_url with document_name, got %v", p1)
	}
	if _, hasFile := p1["file"]; hasFile {
		t.Fatalf("expected legacy file key removed: %v", p1)
	}
	p2 := parts[2].(map[string]any)
	if p2["type"] != "document_url" || p2["document_url"] != "https://example.com/a.pdf" {
		t.Fatalf("expected file_url part converted to document_url, got %v", p2)
	}
}

func TestTransformMistralChatRequest_ImageInFilePartConvertsToImageURLAndNormalImageUntouched(t *testing.T) {
	raw := []byte(`{
		"model": "mistral-large-4",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "Inspect both images"},
					{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
					{"type": "file", "file": {"filename": "shot.png", "file_data": "data:image/png;base64,BBB="}},
					{"type": "file", "file": {"filename": "doc.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}}
				]
			}
		]
	}`)
	out, changed := TransformMistralChatRequest(raw, "mistral-large-4", DefaultConfig())
	if !changed {
		t.Fatal("expected request to change")
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	parts := got["messages"].([]any)[0].(map[string]any)["content"].([]any)
	// 1. Existing image_url untouched
	p1 := parts[1].(map[string]any)
	if p1["type"] != "image_url" || p1["image_url"].(map[string]any)["url"] != "data:image/png;base64,AAA=" {
		t.Fatalf("existing image_url mutated unexpectedly: %v", p1)
	}
	// 2. image/png wrapped in file part -> image_url (NOT document_url)
	p2 := parts[2].(map[string]any)
	if p2["type"] != "image_url" || p2["image_url"].(map[string]any)["url"] != "data:image/png;base64,BBB=" {
		t.Fatalf("expected image file part converted to image_url, got %v", p2)
	}
	if _, hasDoc := p2["document_url"]; hasDoc {
		t.Fatalf("image must not become document_url: %v", p2)
	}
	// 3. PDF file part -> document_url
	p3 := parts[3].(map[string]any)
	if p3["type"] != "document_url" || p3["document_url"] != "data:application/pdf;base64,JVBERi0=" || p3["document_name"] != "doc.pdf" {
		t.Fatalf("expected pdf file part converted to document_url, got %v", p3)
	}
}

func TestTransformMistralChatRequest_BareBase64InFilePartGetsDataURLPrefix(t *testing.T) {
	raw := []byte(`{
		"model": "mistral-large-4",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "file", "file": {"filename": "document.pdf", "file_data": "JVBERi0xLjQK"}},
				{"type": "file", "file": {"filename": "note.txt", "file_data": "SGVsbG8="}},
				{"type": "file", "file": {"filename": "img.png", "file_data": "iVBORw0KGgo="}}
			]
		}]
	}`)
	out, changed := TransformMistralChatRequest(raw, "mistral-large-4", DefaultConfig())
	if !changed {
		t.Fatal("expected request to change")
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	parts := got["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if parts[0].(map[string]any)["document_url"] != "data:application/pdf;base64,JVBERi0xLjQK" {
		t.Fatalf("expected application/pdf data URL prefix, got %v", parts[0])
	}
	if parts[1].(map[string]any)["document_url"] != "data:text/plain;base64,SGVsbG8=" {
		t.Fatalf("expected text/plain data URL prefix, got %v", parts[1])
	}
	p2 := parts[2].(map[string]any)
	if p2["type"] != "image_url" || p2["image_url"].(map[string]any)["url"] != "data:image/png;base64,iVBORw0KGgo=" {
		t.Fatalf("expected bare base64 png converted to image_url with data:image/png prefix, got %v", p2)
	}
}

func TestTransformMistralResponse_ToolCallsThinkingOnlySetsNullContent(t *testing.T) {
	raw := []byte(`{"id":"1","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"call12345","type":"function","function":{"name":"ping","arguments":"{}"}}],"content":[{"type":"thinking","thinking":[{"type":"text","text":"need tool"}],"closed":true}]}}]}`)
	out, chg := TransformMistralResponse(raw, false)
	if !chg {
		t.Fatal("expected rewrite")
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	msg := got["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	val, exists := msg["content"]
	if !exists || val != nil {
		t.Fatalf("expected content: null when thinking-only + tool_calls, got exists=%v val=%#v out=%s", exists, val, out)
	}
	if msg["reasoning_content"] != "need tool" {
		t.Fatalf("reasoning_content = %v, want 'need tool'", msg["reasoning_content"])
	}
}

func TestTransformMistralResponse_StripsResponseErrorAndNoHTMLEscape(t *testing.T) {
	// 1. Plain-string content chunk carrying non-standard Mistral response_error
	plainWithErr := []byte(`data: {"id":"1","choices":[{"index":0,"delta":{"content":"a<b && c>d"},"finish_reason":"stop","response_error":{"format_error":null,"extra_tags":[]}}]}`)
	out, chg := TransformMistralResponse(plainWithErr, true)
	if !chg {
		t.Fatal("expected response_error to be stripped")
	}
	if bytes.Contains(out, []byte(`"response_error"`)) {
		t.Fatalf("response_error still present: %s", out)
	}
	if !bytes.Contains(out, []byte(`"a<b && c>d"`)) {
		t.Fatalf("HTML chars were unexpectedly escaped in plain chunk: %s", out)
	}
	// 2. HTML characters in both response and request outputs must NOT be escaped to < / > / &
	arrWithHTML := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"x<y && y>z"}]},{"type":"text","text":"a<b & c>d"}]}}]}`)
	out2, chg2 := TransformMistralResponse(arrWithHTML, false)
	if !chg2 {
		t.Fatal("expected rewrite")
	}
	if !bytes.Contains(out2, []byte(`"a<b & c>d"`)) || !bytes.Contains(out2, []byte(`"x<y && y>z"`)) {
		t.Fatalf("HTML chars were unexpectedly escaped in response: %s", out2)
	}
	reqHTML := []byte(`{"model":"mistral-large-4","store":false,"messages":[{"role":"user","content":"a<b & c>d"}]}`)
	outReq, chgReq := TransformMistralChatRequest(reqHTML, "mistral-large-4", DefaultConfig())
	if !chgReq {
		t.Fatal("expected request rewrite")
	}
	if !bytes.Contains(outReq, []byte(`"a<b & c>d"`)) {
		t.Fatalf("HTML chars were unexpectedly escaped in request: %s", outReq)
	}
}

func TestTransformMistralChatRequest_OutputTextEmptyToolsAndDisabledPromptMode(t *testing.T) {
	// 1. output_text -> text, empty tools[] + tool_choice -> dropped, disabled reasoning model drops prompt_mode
	raw := []byte(`{
		"model": "codestral-latest",
		"prompt_mode": "reasoning",
		"tools": [],
		"tool_choice": "auto",
		"messages": [
			{"role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
			{"role": "user", "content": "next"}
		]
	}`)
	out, chg := TransformMistralChatRequest(raw, "codestral-latest", DefaultConfig())
	if !chg {
		t.Fatal("expected request to be sanitized")
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if _, hasPM := got["prompt_mode"]; hasPM {
		t.Fatalf("expected prompt_mode stripped on non-reasoning model codestral-latest: %s", out)
	}
	if _, hasTools := got["tools"]; hasTools {
		t.Fatalf("expected empty tools[] removed: %s", out)
	}
	if _, hasTC := got["tool_choice"]; hasTC {
		t.Fatalf("expected orphaned tool_choice removed when tools is empty: %s", out)
	}
	msg0 := got["messages"].([]any)[0].(map[string]any)
	part0 := msg0["content"].([]any)[0].(map[string]any)
	if part0["type"] != "text" {
		t.Fatalf("expected output_text normalized to text, got %v", part0["type"])
	}

	// 2. Reasoning-capable model without reasoning_effort keeps prompt_mode
	raw2 := []byte(`{"model":"mistral-large-4","prompt_mode":"reasoning","messages":[{"role":"user","content":"hi"}]}`)
	out2, chg2 := TransformMistralChatRequest(raw2, "mistral-large-4", DefaultConfig())
	if chg2 {
		t.Fatalf("expected reasoning model without reasoning_effort to preserve prompt_mode, got %s", out2)
	}
}

func TestTransformMistralChatRequest_DropsNullToolsAndServiceTier(t *testing.T) {
	raw := []byte(`{"model":"mistral-large-4","tools":null,"service_tier":null,"messages":[{"role":"user","content":"hi"}]}`)
	out, chg := TransformMistralChatRequest(raw, "mistral-large-4", DefaultConfig())
	if !chg {
		t.Fatal("expected null tools and service_tier to be stripped")
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if _, exists := got["tools"]; exists {
		t.Fatalf("tools:null survived: %s", out)
	}
	if _, exists := got["service_tier"]; exists {
		t.Fatalf("service_tier:null survived: %s", out)
	}
}
