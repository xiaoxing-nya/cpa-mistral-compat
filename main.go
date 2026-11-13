package main

/*
#include "cpa_bridge.h"
*/
import "C"

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"gopkg.in/yaml.v3"
)

const (
	abiVersion      uint32 = 1
	schemaVersion   uint32 = 7
	pluginName             = "cpa-mistral-compat"
	pluginVersion          = "0.2.2"
	maxRequestBytes        = 1 << 30
)

type envelope struct {
	OK     bool            `json:"ok"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  *envelopeError  `json:"error,omitempty"`
}

type envelopeError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type lifecycleRequest struct {
	ConfigYAML    []byte `json:"config_yaml"`
	SchemaVersion uint32 `json:"schema_version"`
}

type configField struct {
	Name        string   `json:"Name"`
	Type        string   `json:"Type"`
	Default     any      `json:"Default,omitempty"`
	EnumValues  []string `json:"EnumValues,omitempty"`
	Description string   `json:"Description,omitempty"`
}

type metadata struct {
	Name             string        `json:"Name"`
	Version          string        `json:"Version"`
	Author           string        `json:"Author"`
	GitHubRepository string        `json:"GitHubRepository"`
	Logo             string        `json:"Logo,omitempty"`
	ConfigFields     []configField `json:"ConfigFields"`
}

type capabilities struct {
	RequestNormalizer                bool `json:"request_normalizer"`
	RequestInterceptor               bool `json:"request_interceptor"`
	RequestInterceptorOmitBodyBefore bool `json:"request_interceptor_omit_body_before,omitempty"`
	RequestInterceptorOmitBodyAfter  bool `json:"request_interceptor_omit_body_after,omitempty"`
	ResponseBeforeTranslator         bool `json:"response_before_translator"`
}

type registrationResponse struct {
	SchemaVersion uint32       `json:"schema_version"`
	Metadata      metadata     `json:"metadata"`
	Capabilities  capabilities `json:"capabilities"`
}

var (
	activeCfg atomic.Pointer[Config]

	refreshMu sync.Mutex

	discoveredMu             sync.RWMutex
	mistralCredIDs           = map[string]bool{}
	knownCredIDs             = map[string]bool{}
	crossProtoMistralTargets = map[string]time.Time{}
	lastDiscovery            time.Time

	hostAPICaller = callHostAPI
)

var (
	emptyPayloadEnvelope   []byte
	emptyInterceptEnvelope []byte
)

func init() {
	def := DefaultConfig()
	activeCfg.Store(&def)
	emptyPayloadEnvelope, _ = okEnvelope(payloadResp{})
	emptyInterceptEnvelope, _ = okEnvelope(requestInterceptResp{})
}

func main() {}

//export cliproxy_plugin_init
func cliproxy_plugin_init(host *C.cliproxy_host_api, plugin *C.cliproxy_plugin_api) (rc C.int) {
	defer func() {
		if recover() != nil {
			rc = 1
		}
	}()
	if plugin == nil {
		return 1
	}
	if host != nil && uint32(host.abi_version) != abiVersion {
		return 1
	}
	C.cpa_store_host_api(host)
	plugin.abi_version = C.uint32_t(abiVersion)
	plugin.call = C.cliproxy_plugin_call_fn(C.cliproxyPluginCall)
	plugin.free_buffer = C.cliproxy_plugin_free_fn(C.cliproxyPluginFree)
	plugin.shutdown = C.cliproxy_plugin_shutdown_fn(C.cliproxyPluginShutdown)
	return 0
}

//export cliproxyPluginCall
func cliproxyPluginCall(method *C.char, request *C.uint8_t, requestLen C.size_t, response *C.cliproxy_buffer) (rc C.int) {
	defer func() {
		if recover() != nil {
			rc = 1
			func() {
				defer func() { _ = recover() }()
				writeResponse(response, errorEnvelope("internal_error", "internal plugin panic"))
			}()
		}
	}()
	if response != nil {
		response.ptr = nil
		response.len = 0
	}
	if method == nil {
		writeResponse(response, errorEnvelope("invalid_method", "method is required"))
		return 1
	}
	if uint64(requestLen) > maxRequestBytes {
		writeResponse(response, errorEnvelope("request_too_large", "request exceeds maximum size"))
		return 1
	}
	var reqBytes []byte
	if request != nil && requestLen > 0 {
		// Read-only view valid only for the duration of this synchronous call.
		reqBytes = unsafe.Slice((*byte)(unsafe.Pointer(request)), int(requestLen))
	}
	raw, err := handleMethod(C.GoString(method), reqBytes)
	if err != nil {
		writeResponse(response, errorEnvelope("plugin_error", err.Error()))
		return 1
	}
	if !writeResponse(response, raw) && len(raw) > 0 {
		return 1
	}
	return 0
}

//export cliproxyPluginFree
func cliproxyPluginFree(ptr unsafe.Pointer, length C.size_t) {
	defer func() { _ = recover() }()
	if ptr != nil {
		C.free(ptr)
	}
	_ = length
}

//export cliproxyPluginShutdown
func cliproxyPluginShutdown() {
	defer func() { _ = recover() }()
	C.cpa_store_host_api(nil)
}

func getConfig() Config {
	if ptr := activeCfg.Load(); ptr != nil {
		return *ptr
	}
	return DefaultConfig()
}

type pluginConfigYAML struct {
	Enabled              *bool `yaml:"enabled"`
	Models               any   `yaml:"models"`
	Providers            any   `yaml:"providers"`
	NormalizeToolCallIDs *bool `yaml:"normalize_tool_call_ids"`
	ConvertThinking      *bool `yaml:"convert_thinking_history"`
}

func parseStringListField(fieldName string, raw any) ([]string, error) {
	switch v := raw.(type) {
	case nil:
		return nil, nil
	case string:
		var out []string
		for _, part := range strings.Split(v, ",") {
			if s := strings.TrimSpace(part); s != "" {
				out = append(out, s)
			}
		}
		return out, nil
	case []any:
		var out []string
		for _, item := range v {
			s, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("%s: every entry must be a string", fieldName)
			}
			for _, part := range strings.Split(s, ",") {
				if trimmed := strings.TrimSpace(part); trimmed != "" {
					out = append(out, trimmed)
				}
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("%s: expected a list of strings or a comma-separated string", fieldName)
	}
}

func applyPluginConfigStruct(cfg *Config, pc pluginConfigYAML) error {
	if pc.Enabled != nil {
		cfg.Enabled = *pc.Enabled
	}
	models, err := parseStringListField("models", pc.Models)
	if err != nil {
		return err
	}
	if len(models) > 0 {
		cfg.Models = append(cfg.Models, models...)
	}
	providers, err := parseStringListField("providers", pc.Providers)
	if err != nil {
		return err
	}
	if len(providers) > 0 {
		cfg.Providers = append(cfg.Providers, providers...)
	}
	if pc.NormalizeToolCallIDs != nil {
		cfg.NormalizeToolCallIDs = *pc.NormalizeToolCallIDs
	}
	if pc.ConvertThinking != nil {
		cfg.ConvertThinkingHistory = *pc.ConvertThinking
	}
	return nil
}

func updateConfigFromYAML(raw []byte) error {
	cfg := DefaultConfig()
	if len(bytes.TrimSpace(raw)) > 0 {
		var root struct {
			Plugins *struct {
				Configs map[string]pluginConfigYAML `yaml:"configs"`
			} `yaml:"plugins"`
		}
		if err := yaml.Unmarshal(raw, &root); err != nil {
			return fmt.Errorf("invalid config yaml: %w", err)
		}
		if root.Plugins != nil {
			if pc, ok := root.Plugins.Configs[pluginName]; ok {
				if err := applyPluginConfigStruct(&cfg, pc); err != nil {
					return err
				}
			}
		} else {
			var direct pluginConfigYAML
			if err := yaml.Unmarshal(raw, &direct); err != nil {
				return fmt.Errorf("invalid config yaml: %w", err)
			}
			if err := applyPluginConfigStruct(&cfg, direct); err != nil {
				return err
			}
		}
	}
	activeCfg.Store(&cfg)

	discoveredMu.Lock()
	lastDiscovery = time.Time{}
	discoveredMu.Unlock()
	return nil
}

func buildRegistration() ([]byte, error) {
	defCfg := DefaultConfig()
	reg := registrationResponse{
		SchemaVersion: schemaVersion,
		Metadata: metadata{
			Name:             pluginName,
			Version:          pluginVersion,
			Author:           "xiaoxing",
			GitHubRepository: "https://github.com/xiaoxing-nya/cpa-mistral-compat",
			ConfigFields: []configField{
				{
					Name:        "models",
					Type:        "array",
					Default:     defCfg.Models,
					Description: "Fallback model patterns used only when host omits selected_auth_id metadata.",
				},
				{
					Name:        "providers",
					Type:        "array",
					Default:     defCfg.Providers,
					Description: "Extra openai-compatibility provider names routed to Mistral API (names containing mistral match automatically).",
				},
				{
					Name:        "normalize_tool_call_ids",
					Type:        "boolean",
					Default:     true,
					Description: "Deterministically normalize tool_call_id to 9-char alphanumeric for Mistral.",
				},
				{
					Name:        "convert_thinking_history",
					Type:        "boolean",
					Default:     true,
					Description: "Convert assistant reasoning_content into Mistral typed thinking content blocks.",
				},
			},
		},
		Capabilities: capabilities{
			RequestNormalizer:                true,
			RequestInterceptor:               true,
			RequestInterceptorOmitBodyBefore: true,
			RequestInterceptorOmitBodyAfter:  false,
			ResponseBeforeTranslator:         true,
		},
	}
	return okEnvelope(reg)
}

type requestTransformReq struct {
	FromFormat string `json:"FromFormat"`
	ToFormat   string `json:"ToFormat"`
	Model      string `json:"Model"`
	Stream     bool   `json:"Stream"`
	Body       []byte `json:"Body"`
}

type responseTransformReq struct {
	FromFormat string `json:"FromFormat"`
	ToFormat   string `json:"ToFormat"`
	Model      string `json:"Model"`
	Stream     bool   `json:"Stream"`
	Body       []byte `json:"Body"`
}

type payloadResp struct {
	Body []byte `json:"Body,omitempty"`
}

type requestInterceptReq struct {
	SourceFormat   string         `json:"SourceFormat"`
	ToFormat       string         `json:"ToFormat"`
	Model          string         `json:"Model"`
	RequestedModel string         `json:"RequestedModel"`
	Body           []byte         `json:"Body"`
	Metadata       map[string]any `json:"Metadata"`
	HostCallbackID string         `json:"host_callback_id"`
}

type requestInterceptResp struct {
	Body []byte `json:"Body,omitempty"`
}

var (
	markOriginal = []byte(`,"OriginalRequest":`)
	markBody     = []byte(`,"Body":`)
)

// decodeResponseTransformRequest extracts header fields and trailing Body
// without scanning large OriginalRequest / TranslatedRequest base64 payloads.
// Falls back to full json.Unmarshal if field order differs from the host encoder.
func decodeResponseTransformRequest(request []byte) (responseTransformReq, error) {
	var req responseTransformReq
	trimmed := bytes.TrimSpace(request)
	i := bytes.Index(trimmed, markOriginal)
	j := bytes.LastIndex(trimmed, markBody)
	if i > 0 && j > i && bytes.HasSuffix(trimmed, []byte("}")) {
		header := make([]byte, 0, i+1)
		header = append(header, trimmed[:i]...)
		header = append(header, '}')
		var head struct {
			FromFormat string `json:"FromFormat"`
			ToFormat   string `json:"ToFormat"`
			Model      string `json:"Model"`
			Stream     bool   `json:"Stream"`
		}
		if json.Unmarshal(header, &head) == nil {
			req.FromFormat = head.FromFormat
			req.ToFormat = head.ToFormat
			req.Model = head.Model
			req.Stream = head.Stream
			fromFmt := strings.ToLower(strings.TrimSpace(head.FromFormat))
			if fromFmt != "" && fromFmt != "openai" {
				return req, nil
			}
			var body []byte
			if json.Unmarshal(trimmed[j+len(markBody):len(trimmed)-1], &body) == nil {
				req.Body = body
				return req, nil
			}
		}
	}
	err := json.Unmarshal(request, &req)
	return req, err
}

func stringFromMetadata(meta map[string]any, key string) string {
	if len(meta) == 0 {
		return ""
	}
	if v, ok := meta[key].(string); ok {
		return strings.TrimSpace(v)
	}
	return ""
}

func crossProtoTargetKey(fromFmt, model string) string {
	m := strings.ToLower(strings.TrimSpace(stripThinkingSuffix(model)))
	if fromFmt == "" || m == "" {
		return ""
	}
	return strings.ToLower(strings.TrimSpace(fromFmt)) + "|" + m
}

func markCrossProtoMistralTarget(fromFmt, model string) {
	k := crossProtoTargetKey(fromFmt, model)
	if k == "" {
		return
	}
	now := time.Now()
	discoveredMu.Lock()
	for oldKey, ts := range crossProtoMistralTargets {
		if now.Sub(ts) > 60*time.Second {
			delete(crossProtoMistralTargets, oldKey)
		}
	}
	if len(crossProtoMistralTargets) >= 512 {
		clear(crossProtoMistralTargets)
	}
	crossProtoMistralTargets[k] = now
	discoveredMu.Unlock()
}

func clearCrossProtoMistralTarget(fromFmt, model string) {
	k := crossProtoTargetKey(fromFmt, model)
	if k == "" {
		return
	}
	discoveredMu.Lock()
	delete(crossProtoMistralTargets, k)
	discoveredMu.Unlock()
}

func isCrossProtoMistralTarget(fromFmt, model string) bool {
	k := crossProtoTargetKey(fromFmt, model)
	if k == "" {
		return false
	}
	now := time.Now()
	discoveredMu.RLock()
	ts, ok := crossProtoMistralTargets[k]
	discoveredMu.RUnlock()
	if !ok {
		return false
	}
	return now.Sub(ts) <= 60*time.Second
}

func handleMethod(method string, reqBytes []byte) ([]byte, error) {
	switch method {
	case "plugin.register", "plugin.reconfigure":
		var lr lifecycleRequest
		if len(reqBytes) > 0 {
			if err := json.Unmarshal(reqBytes, &lr); err != nil {
				return errorEnvelope("bad_request", "decode lifecycle request"), nil
			}
			if err := updateConfigFromYAML(lr.ConfigYAML); err != nil {
				return errorEnvelope("invalid_config", err.Error()), nil
			}
		}
		return buildRegistration()

	case "plugin.quiesce", "plugin.shutdown":
		return okEnvelope(struct{}{})

	case "request.intercept_before":
		// Lightweight metadata-only hook (omit_body_before=true) to warm credential discovery.
		var req requestInterceptReq
		if json.Unmarshal(reqBytes, &req) == nil {
			cfg := getConfig()
			if cfg.Enabled && req.HostCallbackID != "" {
				refreshCredentials(req.HostCallbackID, "", "", cfg)
			}
		}
		return emptyInterceptEnvelope, nil

	case "request.intercept_after":
		var req requestInterceptReq
		if json.Unmarshal(reqBytes, &req) != nil {
			return emptyInterceptEnvelope, nil
		}
		cfg := getConfig()
		if !cfg.Enabled {
			return emptyInterceptEnvelope, nil
		}
		toFmt := strings.ToLower(strings.TrimSpace(req.ToFormat))
		if toFmt != "" && toFmt != "openai" {
			return emptyInterceptEnvelope, nil
		}

		authID := stringFromMetadata(req.Metadata, "selected_auth_id")
		authIndex := stringFromMetadata(req.Metadata, "selected_auth_index")
		if req.HostCallbackID != "" {
			refreshCredentials(req.HostCallbackID, authID, authIndex, cfg)
		}

		var isMistral bool
		if authID != "" || authIndex != "" {
			isMistral = isMistralCredential(authID, authIndex)
		} else {
			// Legacy fallback when host omits selected_auth_* metadata.
			isMistral = IsMistralTarget(req.Model, req.RequestedModel, "", req.Body, cfg)
		}

		srcFmt := strings.ToLower(strings.TrimSpace(req.SourceFormat))
		if !isMistral {
			if srcFmt != "" && srcFmt != "openai" {
				clearCrossProtoMistralTarget(srcFmt, req.Model)
			}
			return emptyInterceptEnvelope, nil
		}

		if srcFmt == "" || srcFmt == "openai" {
			if len(req.Body) == 0 {
				return emptyInterceptEnvelope, nil
			}
			model := req.Model
			if model == "" {
				model = req.RequestedModel
			}
			if out, changed := TransformMistralChatRequest(req.Body, model, cfg); changed {
				return okEnvelope(requestInterceptResp{Body: out})
			}
			return emptyInterceptEnvelope, nil
		}

		// Cross-protocol request (claude / openai-response / gemini -> openai):
		// strip unsupported source thinking here (before ApplyRequestThinkingDoc reads OriginalRequest),
		// and mark the (SourceFormat, upstreamModel) pair for post-translation normalization in request.normalize.
		markCrossProtoMistralTarget(srcFmt, req.Model)
		if len(req.Body) > 0 {
			if out, changed := StripUnsupportedSourceThinking(req.Body, srcFmt, req.Model, req.RequestedModel); changed {
				return okEnvelope(requestInterceptResp{Body: out})
			}
		}
		return emptyInterceptEnvelope, nil

	case "request.normalize":
		var req requestTransformReq
		if err := json.Unmarshal(reqBytes, &req); err != nil {
			return emptyPayloadEnvelope, nil
		}
		cfg := getConfig()
		if !cfg.Enabled || len(req.Body) == 0 {
			return emptyPayloadEnvelope, nil
		}
		toFmt := strings.ToLower(strings.TrimSpace(req.ToFormat))
		if toFmt != "" && toFmt != "openai" {
			return emptyPayloadEnvelope, nil
		}
		fromFmt := strings.ToLower(strings.TrimSpace(req.FromFormat))
		// Same-protocol OpenAI requests are already gated by selected_auth_id and normalized in request.intercept_after.
		if fromFmt == "" || fromFmt == "openai" {
			return emptyPayloadEnvelope, nil
		}
		if !isCrossProtoMistralTarget(fromFmt, req.Model) {
			return emptyPayloadEnvelope, nil
		}
		if out, changed := TransformMistralChatRequest(req.Body, req.Model, cfg); changed {
			return okEnvelope(payloadResp{Body: out})
		}
		return emptyPayloadEnvelope, nil

	case "response.normalize_before":
		cfg := getConfig()
		if !cfg.Enabled {
			return emptyPayloadEnvelope, nil
		}
		req, err := decodeResponseTransformRequest(reqBytes)
		if err != nil || len(req.Body) == 0 {
			return emptyPayloadEnvelope, nil
		}
		fromFmt := strings.ToLower(strings.TrimSpace(req.FromFormat))
		if fromFmt != "" && fromFmt != "openai" {
			return emptyPayloadEnvelope, nil
		}
		if out, changed := TransformMistralResponse(req.Body, req.Stream); changed {
			return okEnvelope(payloadResp{Body: out})
		}
		return emptyPayloadEnvelope, nil

	default:
		return errorEnvelope("unknown_method", "unknown method: "+method), nil
	}
}

func callHostAPI(method string, payload []byte) ([]byte, bool) {
	cMethod := C.CString(method)
	defer C.free(unsafe.Pointer(cMethod))
	var resp C.cliproxy_buffer
	var reqPtr *C.uint8_t
	if len(payload) > 0 {
		reqPtr = (*C.uint8_t)(C.cpa_copy_out(unsafe.Pointer(&payload[0]), C.size_t(len(payload))))
		if reqPtr == nil {
			return nil, false
		}
		defer C.free(unsafe.Pointer(reqPtr))
	}
	rc := C.cpa_call_host_api(cMethod, reqPtr, C.size_t(len(payload)), &resp)
	if rc != 0 || resp.ptr == nil {
		return nil, false
	}
	defer C.cpa_free_host_buffer(resp.ptr, resp.len)
	out := C.GoBytes(resp.ptr, C.int(resp.len))
	var env envelope
	if json.Unmarshal(out, &env) != nil || !env.OK {
		return nil, false
	}
	return env.Result, true
}

func isMistralProviderKey(provider, baseURL string, cfg Config) bool {
	p := strings.ToLower(strings.TrimSpace(provider))
	b := strings.ToLower(strings.TrimSpace(baseURL))
	if strings.Contains(b, "mistral.ai") || strings.Contains(p, "mistral") {
		return true
	}
	for _, cp := range cfg.Providers {
		cpLow := strings.ToLower(strings.TrimSpace(cp))
		if cpLow != "" && (p == cpLow || p == "openai-compatible-"+cpLow) {
			return true
		}
	}
	return false
}

func isMistralCredential(authID, authIndex string) bool {
	discoveredMu.RLock()
	defer discoveredMu.RUnlock()
	if authID != "" && mistralCredIDs[authID] {
		return true
	}
	if authIndex != "" && mistralCredIDs[authIndex] {
		return true
	}
	return false
}

func isCredentialsFreshLocked(unseenID1, unseenID2 string) bool {
	fresh := time.Since(lastDiscovery) < 15*time.Second && len(knownCredIDs) > 0
	if fresh && unseenID1 != "" && !knownCredIDs[unseenID1] {
		fresh = false
	}
	if fresh && unseenID2 != "" && !knownCredIDs[unseenID2] {
		fresh = false
	}
	return fresh
}

func refreshCredentials(callbackID, unseenID1, unseenID2 string, cfg Config) {
	if callbackID == "" {
		return
	}
	discoveredMu.RLock()
	fresh := isCredentialsFreshLocked(unseenID1, unseenID2)
	discoveredMu.RUnlock()
	if fresh {
		return
	}

	refreshMu.Lock()
	defer refreshMu.Unlock()

	discoveredMu.RLock()
	fresh = isCredentialsFreshLocked(unseenID1, unseenID2)
	discoveredMu.RUnlock()
	if fresh {
		return
	}

	reqBody, _ := json.Marshal(map[string]string{"host_callback_id": callbackID})

	nextMistral := map[string]bool{}
	nextKnown := map[string]bool{}

	if listRaw, ok := hostAPICaller("host.auth.list", reqBody); ok {
		var listResp struct {
			Files []struct {
				ID        string `json:"id"`
				AuthIndex string `json:"auth_index"`
				Provider  string `json:"provider"`
				BaseURL   string `json:"base_url"`
			} `json:"files"`
		}
		if json.Unmarshal(listRaw, &listResp) == nil {
			for _, f := range listResp.Files {
				if f.ID != "" {
					nextKnown[f.ID] = true
				}
				if f.AuthIndex != "" {
					nextKnown[f.AuthIndex] = true
				}
				if isMistralProviderKey(f.Provider, f.BaseURL, cfg) {
					if f.ID != "" {
						nextMistral[f.ID] = true
					}
					if f.AuthIndex != "" {
						nextMistral[f.AuthIndex] = true
					}
				}
			}
		}
	}

	modelsRaw, ok := hostAPICaller("host.auth.models", reqBody)
	if !ok {
		return
	}
	var modelsResp struct {
		Credentials []struct {
			ID        string   `json:"id"`
			AuthIndex string   `json:"auth_index"`
			Provider  string   `json:"provider"`
			Models    []string `json:"models"`
		} `json:"credentials"`
	}
	if json.Unmarshal(modelsRaw, &modelsResp) != nil {
		return
	}

	for _, c := range modelsResp.Credentials {
		if c.ID != "" {
			nextKnown[c.ID] = true
		}
		if c.AuthIndex != "" {
			nextKnown[c.AuthIndex] = true
		}
		if nextMistral[c.ID] || nextMistral[c.AuthIndex] || isMistralProviderKey(c.Provider, "", cfg) {
			if c.ID != "" {
				nextMistral[c.ID] = true
			}
			if c.AuthIndex != "" {
				nextMistral[c.AuthIndex] = true
			}
		}
	}

	discoveredMu.Lock()
	mistralCredIDs = nextMistral
	knownCredIDs = nextKnown
	lastDiscovery = time.Now()
	discoveredMu.Unlock()
}

func okEnvelope(v any) ([]byte, error) {
	raw, err := marshalNoEscape(v)
	if err != nil {
		return nil, err
	}
	return marshalNoEscape(envelope{OK: true, Result: raw})
}

func errorEnvelope(code, message string) []byte {
	raw, _ := marshalNoEscape(envelope{OK: false, Error: &envelopeError{Code: code, Message: message}})
	return raw
}

func writeResponse(response *C.cliproxy_buffer, raw []byte) bool {
	if response == nil || len(raw) == 0 {
		return false
	}
	ptr := C.cpa_copy_out(unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	if ptr == nil {
		return false
	}
	response.ptr = ptr
	response.len = C.size_t(len(raw))
	return true
}
