package main

import (
	"encoding/json"
	"testing"
	"time"
)

func resetPluginStateForTest() {
	cfgMu.Lock()
	activeCfg = DefaultConfig()
	cfgMu.Unlock()

	discoveredMu.Lock()
	mistralCredIDs = map[string]bool{}
	knownCredIDs = map[string]bool{}
	crossProtoMistralTargets = map[string]time.Time{}
	lastDiscovery = time.Now()
	discoveredMu.Unlock()
}

func callPluginMethodForTest(t *testing.T, method string, req any) envelope {
	t.Helper()
	var reqBytes []byte
	if req != nil {
		var err error
		reqBytes, err = json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal %s req: %v", method, err)
		}
	}
	raw, err := handleMethod(method, reqBytes)
	if err != nil {
		t.Fatalf("handleMethod(%s) error = %v", method, err)
	}
	var env envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatalf("unmarshal %s envelope: %v", method, err)
	}
	if !env.OK {
		t.Fatalf("handleMethod(%s) not OK: %+v", method, env.Error)
	}
	return env
}

func TestRequestInterceptAfter_CredentialIsolationAndAliasNormalization(t *testing.T) {
	resetPluginStateForTest()
	discoveredMu.Lock()
	mistralCredIDs["openai-compatibility:mistral:aaa"] = true
	mistralCredIDs["idx-mistral"] = true
	knownCredIDs["openai-compatibility:mistral:aaa"] = true
	knownCredIDs["idx-mistral"] = true
	knownCredIDs["openai-compatibility:cline-pass:bbb"] = true
	knownCredIDs["idx-cline"] = true
	discoveredMu.Unlock()

	// 1. Check registration disables OmitBodyAfter so intercept_after receives Body.
	regEnv := callPluginMethodForTest(t, "plugin.register", nil)
	var reg registrationResponse
	if err := json.Unmarshal(regEnv.Result, &reg); err != nil {
		t.Fatalf("unmarshal reg: %v", err)
	}
	if !reg.Capabilities.RequestInterceptorOmitBodyBefore {
		t.Fatal("expected RequestInterceptorOmitBodyBefore=true")
	}
	if reg.Capabilities.RequestInterceptorOmitBodyAfter {
		t.Fatal("expected RequestInterceptorOmitBodyAfter=false so request.intercept_after receives Body")
	}

	// 2. Non-Mistral credential (cline-pass) with shared model/name MUST be completely untouched.
	clineBody := []byte(`{"model":"glm-5.3","store":true,"reasoning_effort":"medium","messages":[{"role":"developer","content":"sys"}]}`)
	clineEnv := callPluginMethodForTest(t, "request.intercept_after", map[string]any{
		"SourceFormat":   "openai",
		"ToFormat":       "openai",
		"Model":          "glm-5.3",
		"RequestedModel": "glm-5.3",
		"Body":           clineBody,
		"Metadata": map[string]any{
			"selected_auth_id":    "openai-compatibility:cline-pass:bbb",
			"selected_auth_index": "idx-cline",
		},
	})
	var clineResp requestInterceptResp
	if err := json.Unmarshal(clineEnv.Result, &clineResp); err != nil {
		t.Fatalf("unmarshal clineResp: %v", err)
	}
	if len(clineResp.Body) != 0 {
		t.Fatalf("expected non-Mistral credential to remain untouched, got body: %s", clineResp.Body)
	}

	// 3. Mistral credential with custom alias + upstream zai-glm-5-3 + store + unsupported medium effort:
	// MUST strip store, convert developer->system, and drop medium in intercept_after.
	mistralBody := []byte(`{"model":"custom-alias","store":true,"reasoning_effort":"medium","messages":[{"role":"developer","content":"sys"},{"role":"user","content":"hi"}]}`)
	mistralEnv := callPluginMethodForTest(t, "request.intercept_after", map[string]any{
		"SourceFormat":   "openai",
		"ToFormat":       "openai",
		"Model":          "zai-glm-5-3",
		"RequestedModel": "custom-alias",
		"Body":           mistralBody,
		"Metadata": map[string]any{
			"selected_auth_id":    "openai-compatibility:mistral:aaa",
			"selected_auth_index": "idx-mistral",
		},
	})
	var mistralResp requestInterceptResp
	if err := json.Unmarshal(mistralEnv.Result, &mistralResp); err != nil {
		t.Fatalf("unmarshal mistralResp: %v", err)
	}
	if len(mistralResp.Body) == 0 {
		t.Fatal("expected Mistral credential request body to be normalized in intercept_after")
	}
	var got map[string]any
	if err := json.Unmarshal(mistralResp.Body, &got); err != nil {
		t.Fatalf("unmarshal normalized body: %v", err)
	}
	if _, hasStore := got["store"]; hasStore {
		t.Fatalf("store should be stripped: %s", mistralResp.Body)
	}
	if _, hasRE := got["reasoning_effort"]; hasRE {
		t.Fatalf("unsupported medium reasoning_effort should be dropped for zai-glm-5-3: %s", mistralResp.Body)
	}
	msgs := got["messages"].([]any)
	if msgs[0].(map[string]any)["role"] != "system" {
		t.Fatalf("developer role should convert to system: %s", mistralResp.Body)
	}

	// 4. Cross-protocol (claude -> openai): intercept_after strips bad thinking & marks target,
	// then request.normalize finishes Wire-Format normalization ONLY for that cross-protocol Mistral target.
	claudeBody := []byte(`{"model":"custom-alias","thinking":{"type":"enabled","budget_tokens":8192},"messages":[{"role":"user","content":"hi"}]}`)
	claudeAfterEnv := callPluginMethodForTest(t, "request.intercept_after", map[string]any{
		"SourceFormat":   "claude",
		"ToFormat":       "openai",
		"Model":          "zai-glm-5-3",
		"RequestedModel": "custom-alias",
		"Body":           claudeBody,
		"Metadata": map[string]any{
			"selected_auth_id":    "openai-compatibility:mistral:aaa",
			"selected_auth_index": "idx-mistral",
		},
	})
	var claudeAfterResp requestInterceptResp
	_ = json.Unmarshal(claudeAfterEnv.Result, &claudeAfterResp)
	if len(claudeAfterResp.Body) == 0 {
		t.Fatal("expected unsupported Claude thinking budget_tokens=8192 (medium) stripped in intercept_after")
	}
	var strippedClaude map[string]any
	_ = json.Unmarshal(claudeAfterResp.Body, &strippedClaude)
	if _, hasThinking := strippedClaude["thinking"]; hasThinking {
		t.Fatalf("expected thinking stripped from Claude body: %s", claudeAfterResp.Body)
	}

	// Subsequent request.normalize after CPA translation (FromFormat="claude", ToFormat="openai", Model="zai-glm-5-3"):
	translatedOpenAI := []byte(`{"model":"zai-glm-5-3","store":true,"messages":[{"role":"developer","content":"sys"},{"role":"user","content":"hi"}]}`)
	normMistralEnv := callPluginMethodForTest(t, "request.normalize", map[string]any{
		"FromFormat": "claude",
		"ToFormat":   "openai",
		"Model":      "zai-glm-5-3",
		"Body":       translatedOpenAI,
	})
	var normMistralResp payloadResp
	_ = json.Unmarshal(normMistralEnv.Result, &normMistralResp)
	if len(normMistralResp.Body) == 0 {
		t.Fatal("expected cross-protocol request.normalize to normalize confirmed Mistral model")
	}

	// Cross-protocol for non-Mistral credential (cline-pass) MUST NOT trigger request.normalize:
	_ = callPluginMethodForTest(t, "request.intercept_after", map[string]any{
		"SourceFormat":   "claude",
		"ToFormat":       "openai",
		"Model":          "deepseek-v4.1-flash",
		"RequestedModel": "deepseek-v4.1-flash",
		"Body":           []byte(`{"model":"deepseek-v4.1-flash","messages":[]}`),
		"Metadata": map[string]any{
			"selected_auth_id":    "openai-compatibility:cline-pass:bbb",
			"selected_auth_index": "idx-cline",
		},
	})
	normClineEnv := callPluginMethodForTest(t, "request.normalize", map[string]any{
		"FromFormat": "claude",
		"ToFormat":   "openai",
		"Model":      "deepseek-v4.1-flash",
		"Body":       []byte(`{"model":"deepseek-v4.1-flash","store":true,"messages":[{"role":"developer","content":"sys"}]}`),
	})
	var normClineResp payloadResp
	_ = json.Unmarshal(normClineEnv.Result, &normClineResp)
	if len(normClineResp.Body) != 0 {
		t.Fatalf("expected non-Mistral cross-protocol request untouched, got: %s", normClineResp.Body)
	}
}

func TestRefreshCredentials_HotReloadsUnseenAuthID(t *testing.T) {
	resetPluginStateForTest()
	origCaller := hostAPICaller
	defer func() { hostAPICaller = origCaller }()

	callCount := 0
	hostAPICaller = func(method string, payload []byte) ([]byte, bool) {
		callCount++
		switch method {
		case "host.auth.list":
			return []byte(`{"files":[]}`), true
		case "host.auth.models":
			if callCount <= 2 {
				return []byte(`{"credentials":[{"id":"openai-compatibility:cline-pass:bbb","auth_index":"idx-cline","provider":"openai-compatible-cline-pass","models":["glm-5.3"]}]}`), true
			}
			// Hot-added Mistral credential appears on second refresh.
			return []byte(`{"credentials":[
				{"id":"openai-compatibility:cline-pass:bbb","auth_index":"idx-cline","provider":"openai-compatible-cline-pass","models":["glm-5.3"]},
				{"id":"openai-compatibility:mistral:new","auth_index":"idx-new","provider":"openai-compatible-mistral","models":["my-alias"]}
			]}`), true
		}
		return nil, false
	}

	// First request warms discovery (only cline-pass known).
	_ = callPluginMethodForTest(t, "request.intercept_before", map[string]any{
		"host_callback_id": "cb-1",
	})

	// Second request within 15s targets a brand-new unseen Mistral auth_id -> MUST force refresh and normalize!
	env := callPluginMethodForTest(t, "request.intercept_after", map[string]any{
		"SourceFormat":   "openai",
		"ToFormat":       "openai",
		"Model":          "zai-glm-5-3",
		"RequestedModel": "my-alias",
		"Body":           []byte(`{"model":"my-alias","store":true,"reasoning_effort":"medium","messages":[{"role":"user","content":"hi"}]}`),
		"Metadata": map[string]any{
			"selected_auth_id":    "openai-compatibility:mistral:new",
			"selected_auth_index": "idx-new",
		},
		"host_callback_id": "cb-2",
	})
	var resp requestInterceptResp
	if err := json.Unmarshal(env.Result, &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Body) == 0 {
		t.Fatalf("expected hot-added Mistral credential to be recognized immediately and normalized")
	}
	var got map[string]any
	if err := json.Unmarshal(resp.Body, &got); err != nil {
		t.Fatal(err)
	}
	if _, hasStore := got["store"]; hasStore {
		t.Fatalf("store should be stripped: %s", resp.Body)
	}
	if _, hasRE := got["reasoning_effort"]; hasRE {
		t.Fatalf("medium effort should be stripped for zai-glm-5-3: %s", resp.Body)
	}
}
