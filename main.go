package main

/*
#include "cpa_bridge.h"
*/
import "C"

import (
	"encoding/json"
	"strings"
	"sync"
	"time"
	"unsafe"

	"gopkg.in/yaml.v3"
)

const (
	abiVersion    uint32 = 1
	schemaVersion uint32 = 7
	pluginName           = "cpa-mistral-compat"
	pluginVersion        = "0.2.2"
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
	ResponseInterceptor              bool `json:"response_interceptor"`
	StreamChunkInterceptor           bool `json:"response_stream_interceptor"`
}

type registrationResponse struct {
	SchemaVersion uint32       `json:"schema_version"`
	Metadata      metadata     `json:"metadata"`
	Capabilities  capabilities `json:"capabilities"`
}

var (
	cfgMu     sync.RWMutex
	activeCfg = DefaultConfig()

	discoveredMu             sync.RWMutex
	mistralCredIDs           = map[string]bool{}
	knownCredIDs             = map[string]bool{}
	crossProtoMistralTargets = map[string]time.Time{}
	lastDiscovery            time.Time

	hostAPICaller = callHostAPI
)

func main() {}

//export cliproxy_plugin_init
func cliproxy_plugin_init(host *C.cliproxy_host_api, plugin *C.cliproxy_plugin_api) C.int {
	if plugin == nil {
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
func cliproxyPluginCall(method *C.char, request *C.uint8_t, requestLen C.size_t, response *C.cliproxy_buffer) C.int {
	if response != nil {
		response.ptr = nil
		response.len = 0
	}
	if method == nil {
		writeResponse(response, errorEnvelope("invalid_method", "method is required"))
		return 1
	}
	var reqBytes []byte
	if request != nil && requestLen > 0 {
		reqBytes = C.GoBytes(unsafe.Pointer(request), C.int(requestLen))
	}
	raw, err := handleMethod(C.GoString(method), reqBytes)
	if err != nil {
		writeResponse(response, errorEnvelope("plugin_error", err.Error()))
		return 1
	}
	writeResponse(response, raw)
	return 0
}

//export cliproxyPluginFree
func cliproxyPluginFree(ptr unsafe.Pointer, length C.size_t) {
	if ptr != nil {
		C.free(ptr)
	}
	_ = length
}

//export cliproxyPluginShutdown
func cliproxyPluginShutdown() {}

func getConfig() Config {
	cfgMu.RLock()
	defer cfgMu.RUnlock()
	return activeCfg
}

type pluginConfigYAML struct {
	Enabled              *bool    `yaml:"enabled"`
	Models               []string `yaml:"models"`
	Providers            []string `yaml:"providers"`
	NormalizeToolCallIDs *bool    `yaml:"normalize_tool_call_ids"`
	ConvertThinking      *bool    `yaml:"convert_thinking_history"`
}

func applyPluginConfigStruct(cfg *Config, pc pluginConfigYAML) {
	if pc.Enabled != nil {
		cfg.Enabled = *pc.Enabled
	}
	if len(pc.Models) > 0 {
		cfg.Models = append(cfg.Models, pc.Models...)
	}
	if len(pc.Providers) > 0 {
		cfg.Providers = append(cfg.Providers, pc.Providers...)
	}
	if pc.NormalizeToolCallIDs != nil {
		cfg.NormalizeToolCallIDs = *pc.NormalizeToolCallIDs
	}
	if pc.ConvertThinking != nil {
		cfg.ConvertThinkingHistory = *pc.ConvertThinking
	}
}

func updateConfigFromYAML(raw []byte) {
	cfg := DefaultConfig()
	if len(raw) > 0 {
		var direct pluginConfigYAML
		if yaml.Unmarshal(raw, &direct) == nil {
			applyPluginConfigStruct(&cfg, direct)
		}
		var root struct {
			Plugins struct {
				Configs map[string]pluginConfigYAML `yaml:"configs"`
			} `yaml:"plugins"`
		}
		if yaml.Unmarshal(raw, &root) == nil {
			if pc, ok := root.Plugins.Configs[pluginName]; ok {
				applyPluginConfigStruct(&cfg, pc)
			}
		}
	}
	cfgMu.Lock()
	activeCfg = cfg
	cfgMu.Unlock()

	discoveredMu.Lock()
	lastDiscovery = time.Time{}
	discoveredMu.Unlock()
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
			ResponseInterceptor:              true,
			StreamChunkInterceptor:           true,
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

type responseInterceptReq struct {
	Body []byte `json:"Body"`
}

type responseInterceptResp struct {
	Body []byte `json:"Body,omitempty"`
}

type streamChunkInterceptReq struct {
	Body       []byte `json:"Body"`
	ChunkIndex int    `json:"ChunkIndex"`
}

type streamChunkInterceptResp struct {
	Body []byte `json:"Body,omitempty"`
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
			_ = json.Unmarshal(reqBytes, &lr)
			updateConfigFromYAML(lr.ConfigYAML)
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
		return okEnvelope(requestInterceptResp{})

	case "request.intercept_after":
		var req requestInterceptReq
		if json.Unmarshal(reqBytes, &req) != nil {
			return okEnvelope(requestInterceptResp{})
		}
		cfg := getConfig()
		if !cfg.Enabled {
			return okEnvelope(requestInterceptResp{})
		}
		toFmt := strings.ToLower(strings.TrimSpace(req.ToFormat))
		if toFmt != "" && toFmt != "openai" {
			return okEnvelope(requestInterceptResp{})
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
			return okEnvelope(requestInterceptResp{})
		}

		if srcFmt == "" || srcFmt == "openai" {
			if len(req.Body) == 0 {
				return okEnvelope(requestInterceptResp{})
			}
			model := req.Model
			if model == "" {
				model = req.RequestedModel
			}
			if out, changed := TransformMistralChatRequest(req.Body, model, cfg); changed {
				return okEnvelope(requestInterceptResp{Body: out})
			}
			return okEnvelope(requestInterceptResp{})
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
		return okEnvelope(requestInterceptResp{})

	case "request.normalize":
		var req requestTransformReq
		if err := json.Unmarshal(reqBytes, &req); err != nil {
			return okEnvelope(payloadResp{})
		}
		cfg := getConfig()
		if !cfg.Enabled || len(req.Body) == 0 {
			return okEnvelope(payloadResp{})
		}
		toFmt := strings.ToLower(strings.TrimSpace(req.ToFormat))
		if toFmt != "" && toFmt != "openai" {
			return okEnvelope(payloadResp{})
		}
		fromFmt := strings.ToLower(strings.TrimSpace(req.FromFormat))
		// Same-protocol OpenAI requests are already gated by selected_auth_id and normalized in request.intercept_after.
		if fromFmt == "" || fromFmt == "openai" {
			return okEnvelope(payloadResp{})
		}
		if !isCrossProtoMistralTarget(fromFmt, req.Model) {
			return okEnvelope(payloadResp{})
		}
		if out, changed := TransformMistralChatRequest(req.Body, req.Model, cfg); changed {
			return okEnvelope(payloadResp{Body: out})
		}
		return okEnvelope(payloadResp{})

	case "response.normalize_before":
		var req responseTransformReq
		if err := json.Unmarshal(reqBytes, &req); err != nil {
			return okEnvelope(payloadResp{})
		}
		cfg := getConfig()
		if !cfg.Enabled || len(req.Body) == 0 {
			return okEnvelope(payloadResp{})
		}
		if out, changed := TransformMistralResponse(req.Body, req.Stream); changed {
			return okEnvelope(payloadResp{Body: out})
		}
		return okEnvelope(payloadResp{})

	case "response.intercept_after":
		var req responseInterceptReq
		if err := json.Unmarshal(reqBytes, &req); err != nil {
			return okEnvelope(responseInterceptResp{})
		}
		cfg := getConfig()
		if !cfg.Enabled || len(req.Body) == 0 {
			return okEnvelope(responseInterceptResp{})
		}
		if out, changed := TransformMistralResponse(req.Body, false); changed {
			return okEnvelope(responseInterceptResp{Body: out})
		}
		return okEnvelope(responseInterceptResp{})

	case "response.intercept_stream_chunk":
		var req streamChunkInterceptReq
		if err := json.Unmarshal(reqBytes, &req); err != nil {
			return okEnvelope(streamChunkInterceptResp{})
		}
		cfg := getConfig()
		if !cfg.Enabled || req.ChunkIndex < 0 || len(req.Body) == 0 {
			return okEnvelope(streamChunkInterceptResp{})
		}
		if out, changed := TransformMistralResponse(req.Body, true); changed {
			return okEnvelope(streamChunkInterceptResp{Body: out})
		}
		return okEnvelope(streamChunkInterceptResp{})

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
		reqPtr = (*C.uint8_t)(C.CBytes(payload))
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

func refreshCredentials(callbackID, unseenID1, unseenID2 string, cfg Config) {
	if callbackID == "" {
		return
	}
	discoveredMu.RLock()
	fresh := time.Since(lastDiscovery) < 15*time.Second && len(knownCredIDs) > 0
	if fresh && unseenID1 != "" && !knownCredIDs[unseenID1] {
		fresh = false
	}
	if fresh && unseenID2 != "" && !knownCredIDs[unseenID2] {
		fresh = false
	}
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
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return json.Marshal(envelope{OK: true, Result: raw})
}

func errorEnvelope(code, message string) []byte {
	raw, _ := json.Marshal(envelope{OK: false, Error: &envelopeError{Code: code, Message: message}})
	return raw
}

func writeResponse(response *C.cliproxy_buffer, raw []byte) {
	if response == nil || len(raw) == 0 {
		return
	}
	ptr := C.CBytes(raw)
	if ptr == nil {
		return
	}
	response.ptr = ptr
	response.len = C.size_t(len(raw))
}
