package main

import (
	"bytes"
	"encoding/json"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func resetPluginStateForTest() {
	defCfg := DefaultConfig()
	activeCfg.Store(&defCfg)

	discoveredMu.Lock()
	mistralCredIDs = map[string]bool{}
	knownCredIDs = map[string]bool{}
	mistralResponseModels = map[string]bool{}
	nonMistralResponseModels = map[string]bool{}
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

func TestRefreshCredentials_ConcurrentSingleFlight(t *testing.T) {
	resetPluginStateForTest()
	discoveredMu.Lock()
	lastDiscovery = time.Time{} // force stale
	discoveredMu.Unlock()

	origCaller := hostAPICaller
	defer func() { hostAPICaller = origCaller }()

	var listCalls atomic.Int32
	var modelsCalls atomic.Int32
	hostAPICaller = func(method string, payload []byte) ([]byte, bool) {
		time.Sleep(15 * time.Millisecond)
		switch method {
		case "host.auth.list":
			listCalls.Add(1)
			return []byte(`{"files":[{"id":"auth-1","auth_index":"idx-1","provider":"openai-compatible-mistral","base_url":"https://api.mistral.ai/v1"}]}`), true
		case "host.auth.models":
			modelsCalls.Add(1)
			return []byte(`{"credentials":[{"id":"auth-1","auth_index":"idx-1","provider":"openai-compatible-mistral","models":["zai-glm-5-3"]}]}`), true
		}
		return nil, false
	}

	cfg := DefaultConfig()
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			refreshCredentials("cb-sf", "auth-1", "idx-1", cfg)
		}()
	}
	wg.Wait()

	if got := listCalls.Load(); got != 1 {
		t.Fatalf("expected single-flight host.auth.list calls = 1, got %d", got)
	}
	if got := modelsCalls.Load(); got != 1 {
		t.Fatalf("expected single-flight host.auth.models calls = 1, got %d", got)
	}
	if !isMistralCredential("auth-1", "idx-1") {
		t.Fatal("expected auth-1/idx-1 to be registered as Mistral credential")
	}
}

func TestBuildRegistration_OmitsRedundantResponseInterceptors(t *testing.T) {
	resetPluginStateForTest()
	regEnv := callPluginMethodForTest(t, "plugin.register", nil)
	var reg map[string]any
	if err := json.Unmarshal(regEnv.Result, &reg); err != nil {
		t.Fatalf("unmarshal reg: %v", err)
	}
	caps, _ := reg["capabilities"].(map[string]any)
	if caps["response_before_translator"] != true {
		t.Fatalf("expected response_before_translator=true, got %#v", caps)
	}
	if v, exists := caps["response_interceptor"]; exists && v == true {
		t.Fatalf("expected response_interceptor omitted/false, got %#v", v)
	}
	if v, exists := caps["response_stream_interceptor"]; exists && v == true {
		t.Fatalf("expected response_stream_interceptor omitted/false, got %#v", v)
	}
}

func TestHandleMethod_ResponseNormalizeBefore_FromFormatAndLargeRequests(t *testing.T) {
	resetPluginStateForTest()
	bigReq := bytes.Repeat([]byte(`{"role":"user","content":"padding"},`), 4000)

	// 1. Non-OpenAI upstream format (e.g. claude) MUST pass through untouched
	nonOpenAIEnv := callPluginMethodForTest(t, "response.normalize_before", map[string]any{
		"FromFormat":        "claude",
		"ToFormat":          "claude",
		"Model":             "claude-sonnet-4",
		"Stream":            true,
		"OriginalRequest":   bigReq,
		"TranslatedRequest": bigReq,
		"Body":              []byte(`data: {"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"x"}]},"response_error":{"x":1}}]}`),
	})
	var nonOpenAIResp payloadResp
	if err := json.Unmarshal(nonOpenAIEnv.Result, &nonOpenAIResp); err != nil {
		t.Fatal(err)
	}
	if len(nonOpenAIResp.Body) != 0 {
		t.Fatalf("expected non-openai FromFormat untouched, got: %s", nonOpenAIResp.Body)
	}

	// 2. OpenAI upstream with large request bodies + Mistral thinking chunk -> normalized
	mistralEnv := callPluginMethodForTest(t, "response.normalize_before", map[string]any{
		"FromFormat":        "openai",
		"ToFormat":          "openai-response",
		"Model":             "mistral-large-4",
		"Stream":            true,
		"OriginalRequest":   bigReq,
		"TranslatedRequest": bigReq,
		"Body":              []byte(`data: {"id":"1","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"step"}]},{"type":"text","text":"ok"}]},"response_error":{"format_error":null}}]}`),
	})
	var mistralResp payloadResp
	if err := json.Unmarshal(mistralEnv.Result, &mistralResp); err != nil {
		t.Fatal(err)
	}
	if len(mistralResp.Body) == 0 || bytes.Contains(mistralResp.Body, []byte("response_error")) {
		t.Fatalf("expected Mistral response normalized without response_error, got: %s", mistralResp.Body)
	}

	// 3. Reordered JSON fields fallback (Body before OriginalRequest) still works
	reorderedRaw, _ := json.Marshal(struct {
		Body            []byte `json:"Body"`
		FromFormat      string `json:"FromFormat"`
		OriginalRequest []byte `json:"OriginalRequest"`
		Stream          bool   `json:"Stream"`
	}{
		Body:            []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":[{"type":"thinking","thinking":"ok"},{"type":"text","text":"42"}]}}]}`),
		FromFormat:      "openai",
		OriginalRequest: []byte(`{}`),
		Stream:          false,
	})
	rawOut, err := handleMethod("response.normalize_before", reorderedRaw)
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	_ = json.Unmarshal(rawOut, &env)
	var reorderedResp payloadResp
	_ = json.Unmarshal(env.Result, &reorderedResp)
	if !bytes.Contains(reorderedResp.Body, []byte(`"reasoning_content":"ok"`)) {
		t.Fatalf("reordered fallback failed: %s", reorderedResp.Body)
	}
}

func TestPluginReconfigure_InvalidYAMLPreservesPreviousConfig(t *testing.T) {
	resetPluginStateForTest()
	t.Cleanup(resetPluginStateForTest)

	// Apply valid custom config first (including comma-separated provider list)
	callPluginMethodForTest(t, "plugin.reconfigure", map[string]any{
		"config_yaml": []byte("enabled: true\nproviders:\n  - custom-mistral\nnormalize_tool_call_ids: false\n"),
	})
	cfg := getConfig()
	if cfg.NormalizeToolCallIDs {
		t.Fatal("expected normalize_tool_call_ids=false after valid reconfigure")
	}

	// Send malformed YAML -> must return an error and keep previous custom config intact
	badReq, _ := json.Marshal(map[string]any{
		"config_yaml": []byte("enabled: [broken"),
	})
	rawOut, err := handleMethod("plugin.reconfigure", badReq)
	if err == nil {
		var env envelope
		_ = json.Unmarshal(rawOut, &env)
		if env.OK {
			t.Fatalf("expected invalid YAML reconfigure to fail, got OK: %s", rawOut)
		}
	}
	after := getConfig()
	if after.NormalizeToolCallIDs {
		t.Fatal("previous config was wiped after failed reconfigure")
	}
	foundCustom := false
	for _, p := range after.Providers {
		if p == "custom-mistral" {
			foundCustom = true
			break
		}
	}
	if !foundCustom {
		t.Fatalf("expected custom-mistral provider preserved, got %#v", after.Providers)
	}
}

type benchHostResponseTransformReq struct {
	FromFormat        string
	ToFormat          string
	Model             string
	Stream            bool
	OriginalRequest   []byte
	TranslatedRequest []byte
	Body              []byte
}

func BenchmarkResponseNormalizeBefore_UnrelatedLargeConv(b *testing.B) {
	resetPluginStateForTest()
	big := bytes.Repeat([]byte(`{"role":"user","content":"lorem ipsum dolor sit amet"},`), 5000)
	req, _ := json.Marshal(benchHostResponseTransformReq{
		FromFormat:        "openai",
		ToFormat:          "openai-response",
		Model:             "gpt-5",
		Stream:            true,
		OriginalRequest:   big,
		TranslatedRequest: big,
		Body:              []byte(`data: {"id":"1","choices":[{"index":0,"delta":{"content":"hello world"},"finish_reason":null}]}`),
	})
	b.SetBytes(int64(len(req)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = handleMethod("response.normalize_before", req)
	}
}

func BenchmarkResponseNormalizeBefore_TransitionChunk(b *testing.B) {
	resetPluginStateForTest()
	big := bytes.Repeat([]byte(`{"role":"user","content":"lorem ipsum dolor sit amet"},`), 5000)
	req, _ := json.Marshal(benchHostResponseTransformReq{
		FromFormat:        "openai",
		ToFormat:          "openai-response",
		Model:             "mistral-large-4",
		Stream:            true,
		OriginalRequest:   big,
		TranslatedRequest: big,
		Body:              []byte(`data: {"id":"1","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"reason"}]},{"type":"text","text":"ans"}]},"finish_reason":"stop","response_error":{"format_error":null}}]}`),
	})
	b.SetBytes(int64(len(req)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = handleMethod("response.normalize_before", req)
	}
}

func TestResponseNormalizeBefore_ModelScopingAndTextArrayProtection(t *testing.T) {
	resetPluginStateForTest()
	t.Cleanup(resetPluginStateForTest)

	// 1. Non-Mistral model (gpt-5) with plain text content array MUST remain untouched.
	gptBody := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":[{"type":"text","text":"hello"}]}}]}`)
	gptEnv := callPluginMethodForTest(t, "response.normalize_before", map[string]any{
		"FromFormat": "openai",
		"ToFormat":   "openai",
		"Model":      "gpt-5",
		"Stream":     false,
		"Body":       gptBody,
	})
	var gptResp payloadResp
	if err := json.Unmarshal(gptEnv.Result, &gptResp); err != nil {
		t.Fatal(err)
	}
	if len(gptResp.Body) != 0 {
		t.Fatalf("expected non-Mistral text-only content array untouched, got: %s", gptResp.Body)
	}

	// 2. Mistral-family model with plain text content array MUST still be flattened.
	misTextBody := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":[{"type":"text","text":"hello"}]}}]}`)
	misEnv := callPluginMethodForTest(t, "response.normalize_before", map[string]any{
		"FromFormat": "openai",
		"ToFormat":   "openai",
		"Model":      "mistral-small-latest",
		"Stream":     false,
		"Body":       misTextBody,
	})
	var misResp payloadResp
	if err := json.Unmarshal(misEnv.Result, &misResp); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(misResp.Body, []byte(`"content":"hello"`)) {
		t.Fatalf("expected Mistral text-only content array flattened, got: %s", misResp.Body)
	}

	// 3. Unknown/alias model on a confirmed Mistral credential with text-only content array MUST be flattened.
	discoveredMu.Lock()
	mistralCredIDs["openai-compatibility:mistral:aaa"] = true
	knownCredIDs["openai-compatibility:mistral:aaa"] = true
	discoveredMu.Unlock()
	_ = callPluginMethodForTest(t, "request.intercept_after", map[string]any{
		"SourceFormat":   "openai",
		"ToFormat":       "openai",
		"Model":          "zai-glm-5-3",
		"RequestedModel": "zai-glm-5-3",
		"Body":           []byte(`{"model":"zai-glm-5-3","messages":[{"role":"user","content":"hi"}]}`),
		"Metadata":       map[string]any{"selected_auth_id": "openai-compatibility:mistral:aaa"},
	})
	glmEnv := callPluginMethodForTest(t, "response.normalize_before", map[string]any{
		"FromFormat": "openai",
		"ToFormat":   "openai",
		"Model":      "zai-glm-5-3",
		"Stream":     false,
		"Body":       misTextBody,
	})
	var glmResp payloadResp
	if err := json.Unmarshal(glmEnv.Result, &glmResp); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(glmResp.Body, []byte(`"content":"hello"`)) {
		t.Fatalf("expected credential-confirmed zai-glm-5-3 text-only array flattened, got: %s", glmResp.Body)
	}

	// 4. Mistral thinking array on non-pattern model (e.g. deepseek-v4-1-flash-1017 without prior intercept) MUST still flatten.
	dsThinkBody := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"step"}]},{"type":"text","text":"ans"}]}}]}`)
	dsEnv := callPluginMethodForTest(t, "response.normalize_before", map[string]any{
		"FromFormat": "openai",
		"ToFormat":   "openai",
		"Model":      "deepseek-v4-1-flash-1017",
		"Stream":     false,
		"Body":       dsThinkBody,
	})
	var dsResp payloadResp
	if err := json.Unmarshal(dsEnv.Result, &dsResp); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(dsResp.Body, []byte(`"reasoning_content":"step"`)) || !bytes.Contains(dsResp.Body, []byte(`"content":"ans"`)) {
		t.Fatalf("expected Mistral thinking chunk flattened on deepseek, got: %s", dsResp.Body)
	}
}

func TestPluginReconfigure_EmptyPayloadResetsToDefaults(t *testing.T) {
	resetPluginStateForTest()
	t.Cleanup(resetPluginStateForTest)

	callPluginMethodForTest(t, "plugin.reconfigure", map[string]any{
		"config_yaml": []byte("enabled: false\nnormalize_tool_call_ids: false\n"),
	})
	if getConfig().Enabled {
		t.Fatal("expected enabled=false after custom reconfigure")
	}

	// Empty payload or nil reqBytes resets back to DefaultConfig().
	rawOut, err := handleMethod("plugin.reconfigure", nil)
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	_ = json.Unmarshal(rawOut, &env)
	if !env.OK {
		t.Fatalf("empty reconfigure failed: %+v", env.Error)
	}
	cfg := getConfig()
	if !cfg.Enabled || !cfg.NormalizeToolCallIDs {
		t.Fatalf("expected empty reconfigure to reset to defaults, got %+v", cfg)
	}
}
