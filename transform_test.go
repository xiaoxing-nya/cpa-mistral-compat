package main

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestTransformMistralChatRequest_PureWireFormat(t *testing.T) {
	cfg := DefaultConfig()
	raw := []byte(`{
		"model": "mistral-small-latest",
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

	out, changed := TransformMistralChatRequest(raw, cfg)
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
		name        string
		model       string
		effort      string
		wantPresent bool
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
		{name: "GLM 5.2 zai-id low is unsupported and falls back", model: "zai-glm-5-2", effort: "low", wantPresent: false},
		{name: "GLM 5.2 zai-id medium is unsupported and falls back", model: "zai-glm-5-2", effort: "medium", wantPresent: false},
		{name: "GLM 5.2 alias-id supported none stays untouched", model: "glm-5-2", effort: "none", wantPresent: true},
		{name: "GLM 5.2 alias-id low is unsupported and falls back", model: "glm-5-2", effort: "low", wantPresent: false},
		{name: "GLM 5.2 dot-alias supported high stays untouched", model: "glm-5.2", effort: "high", wantPresent: true},
		{name: "GLM latest alias supported low stays untouched", model: "zai-glm-latest", effort: "low", wantPresent: true},
		{name: "GLM latest alias none is unsupported and falls back", model: "zai-glm-latest", effort: "none", wantPresent: false},
		{name: "GLM 5 alias supported high stays untouched", model: "zai-glm-5", effort: "high", wantPresent: true},
		{name: "Mistral Large 4 supported medium stays untouched", model: "mistral-large-4", effort: "medium", wantPresent: true},
		{name: "Legacy Mistral Large drops any effort", model: "mistral-large-latest", effort: "high", wantPresent: false},
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
			out, changed := TransformMistralChatRequest(raw, DefaultConfig())
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
