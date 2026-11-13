package main

import (
	"bytes"
	"encoding/json"
	"path"
	"regexp"
	"strconv"
	"strings"
)

// Config holds runtime plugin settings.
type Config struct {
	Enabled                bool
	Models                 []string
	Providers              []string
	NormalizeToolCallIDs   bool
	ConvertThinkingHistory bool
}

// DefaultConfig returns out-of-the-box defaults for Mistral protocol wire-format compatibility.
func DefaultConfig() Config {
	return Config{
		Enabled: true,
		Models: []string{
			"mistral*",
			"*mistral*",
			"codestral*",
			"devstral*",
			"ministral*",
			"magistral*",
			"pixtral*",
			"voxtral*",
			"open-mistral*",
			"open-mixtral*",
			"labs-leanstral*",
		},
		Providers: []string{
			"mistral",
			"mistral-vibe",
			"la-plateforme",
		},
		NormalizeToolCallIDs:   true,
		ConvertThinkingHistory: true,
	}
}

var validMistralToolID = regexp.MustCompile(`^[A-Za-z0-9]{9}$`)

// HashMistralToolID deterministically hashes any tool_call_id into a 9-char [A-Za-z0-9] ID.
func HashMistralToolID(value string) string {
	fnv36 := func(seed uint32) string {
		res := seed
		for i := 0; i < len(value); i++ {
			res = (res ^ uint32(value[i])) * 16777619
		}
		s := strconv.FormatUint(uint64(res), 36)
		if len(s) < 7 {
			s = strings.Repeat("0", 7-len(s)) + s
		}
		return s
	}
	combined := fnv36(2166136261) + fnv36(2246822519)
	if len(combined) > 9 {
		return combined[len(combined)-9:]
	}
	return combined
}

// stripThinkingSuffix strips CPA's model thinking suffix such as "mistral-large-4(high)" for model matching.
func stripThinkingSuffix(model string) string {
	model = strings.TrimSpace(model)
	if idx := strings.LastIndexByte(model, '('); idx > 0 && strings.HasSuffix(model, ")") {
		return strings.TrimSpace(model[:idx])
	}
	return model
}

func matchPattern(pattern, value string) bool {
	pattern = strings.ToLower(strings.TrimSpace(pattern))
	value = strings.ToLower(strings.TrimSpace(value))
	if pattern == "" || value == "" {
		return false
	}
	if pattern == "*" || pattern == value {
		return true
	}
	if ok, err := path.Match(pattern, value); err == nil && ok {
		return true
	}
	if slash := strings.IndexByte(value, '/'); slash >= 0 {
		prefix := value[:slash]
		rest := value[slash+1:]
		if pattern == prefix || pattern == rest {
			return true
		}
		if ok, err := path.Match(pattern, rest); err == nil && ok {
			return true
		}
	}
	return false
}

// IsMistralTarget reports whether the request targets a Mistral model or provider.
func IsMistralTarget(model, requestedModel, provider string, body []byte, cfg Config) bool {
	candidates := []string{
		stripThinkingSuffix(model),
		stripThinkingSuffix(requestedModel),
	}
	if len(body) > 0 && len(candidates[0]) == 0 && len(candidates[1]) == 0 {
		var partial struct {
			Model string `json:"model"`
		}
		if json.Unmarshal(body, &partial) == nil && partial.Model != "" {
			candidates = append(candidates, stripThinkingSuffix(partial.Model))
		}
	}
	for _, p := range cfg.Providers {
		if provider != "" && strings.EqualFold(strings.TrimSpace(p), strings.TrimSpace(provider)) {
			return true
		}
		for _, c := range candidates {
			if strings.HasPrefix(strings.ToLower(c), strings.ToLower(strings.TrimSpace(p))+"/") {
				return true
			}
		}
	}
	for _, c := range candidates {
		if c == "" {
			continue
		}
		for _, pat := range cfg.Models {
			if matchPattern(pat, c) {
				return true
			}
		}
	}
	return false
}

// forbiddenTopLevelFields are non-Mistral OpenAI / 3rd-party fields that trigger Mistral 422 extra_forbidden.
var forbiddenTopLevelFields = []string{
	"store",
	"user",
	"verbosity",
	"reasoning",
	"reasoning_summary",
	"prompt_cache_retention",
	"prompt_cache_options",
	"safety_identifier",
	"include_reasoning",
	"reasoning_format",
	"thinking",
	"enable_thinking",
	"thinking_budget",
	"preserve_thinking",
	"clear_thinking",
	"tool_stream",
	"do_sample",
	"request_id",
	"user_id",
	"options",
	"web_search_options",
	"logit_bias",
	"truncation",
	"background",
	"previous_response_id",
	"max_output_tokens",
	"instructions",
	"include",
	"conversation",
	"prompt",
	"input",
	"text",
}

var forbiddenTopLevelFieldSet = func() map[string]struct{} {
	m := make(map[string]struct{}, len(forbiddenTopLevelFields))
	for _, f := range forbiddenTopLevelFields {
		m[f] = struct{}{}
	}
	return m
}()

var messageReasoningFields = []string{"reasoning_content", "reasoning", "reasoning_text", "reasoning_details"}

// restrictedReasoningEffortLevels lists only models with explicitly constrained wire-level values.
// Unknown models remain pass-through; no requested level is ever remapped to another level.
var restrictedReasoningEffortLevels = []struct {
	modelPattern string
	levels       []string
}{
	// DeepSeek: supports none, low, high, xhigh, max.
	{modelPattern: "*deepseek*", levels: []string{"none", "low", "high", "xhigh", "max"}},

	// GLM 5.2 (zai-glm-5-2, glm-5-2, glm-5.2): supports all standard levels except auto.
	// Must come before *zai-glm* so zai-glm-5-2 does not fall into the 5.3 subset.
	{modelPattern: "*glm-5?2*", levels: []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}},

	// GLM 5.3 and GLM 5.3 Flash (zai-glm-5-3, zai-glm-5-3-flash, glm-5.3, etc.): supports low, high, max.
	{modelPattern: "*glm-5?3*", levels: []string{"low", "high", "max"}},

	// Fallback for other zai-glm aliases (zai-glm-5, zai-glm-latest → GLM 5.3 levels).
	{modelPattern: "*zai-glm*", levels: []string{"low", "high", "max"}},

	// Mistral Large 4 (mistral-large-4, mistral-large-4-0, mistral-large-latest) supports low, medium, high.
	{modelPattern: "mistral-large-4*", levels: []string{"low", "medium", "high"}},
	{modelPattern: "mistral-large-latest", levels: []string{"low", "medium", "high"}},

	// Mistral Small 4 / Medium 3.x / Magistral / Vibe CLI models support none and high only.
	{modelPattern: "mistral-small-2603*", levels: []string{"none", "high"}},
	{modelPattern: "mistral-small-4*", levels: []string{"none", "high"}},
	{modelPattern: "mistral-small-latest", levels: []string{"none", "high"}},
	{modelPattern: "mistral-medium*", levels: []string{"none", "high"}},
	{modelPattern: "magistral-*", levels: []string{"none", "high"}},
	{modelPattern: "mistral-vibe-*", levels: []string{"none", "high"}},

	// Non-reasoning Mistral models (legacy Mistral Large, Codestral, Ministral, Mistral Code).
	{modelPattern: "mistral-large*", levels: []string{}},
	{modelPattern: "codestral*", levels: []string{}},
	{modelPattern: "ministral*", levels: []string{}},
	{modelPattern: "mistral-code*", levels: []string{}},
}

// resolveReasoningEffortSupport returns the supported levels and whether the model matched a restricted pattern.
func resolveReasoningEffortSupport(model string) ([]string, bool) {
	model = stripThinkingSuffix(model)
	for _, capability := range restrictedReasoningEffortLevels {
		if matchPattern(capability.modelPattern, model) {
			return capability.levels, true
		}
	}
	return nil, false
}

func isReasoningEffortSupported(model, effort string) bool {
	levels, matched := resolveReasoningEffortSupport(model)
	if !matched {
		return true
	}
	norm := strings.ToLower(strings.TrimSpace(effort))
	for _, level := range levels {
		if norm == level {
			return true
		}
	}
	return false
}

func resolveEffectiveEffortModel(primary, fallback string) string {
	if _, matched := resolveReasoningEffortSupport(primary); matched {
		return primary
	}
	if _, matched := resolveReasoningEffortSupport(fallback); matched {
		return fallback
	}
	if primary != "" {
		return primary
	}
	return fallback
}

func convertBudgetToLevel(budget int) (string, bool) {
	switch {
	case budget < -1:
		return "", false
	case budget == -1:
		return "auto", true
	case budget == 0:
		return "none", true
	case budget <= 512:
		return "minimal", true
	case budget <= 1024:
		return "low", true
	case budget <= 8192:
		return "medium", true
	case budget <= 24576:
		return "high", true
	default:
		return "xhigh", true
	}
}

func marshalNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

func jsonStringValue(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return strings.ToLower(strings.TrimSpace(s))
	}
	return ""
}

func jsonIntValue(raw json.RawMessage) (int, bool) {
	var i int
	if json.Unmarshal(raw, &i) == nil {
		return i, true
	}
	var f float64
	if json.Unmarshal(raw, &f) == nil {
		return int(f), true
	}
	return 0, false
}

// StripUnsupportedSourceThinking strips known-unsupported reasoning effort settings
// from the source request payload before CPA translates and applies thinking config.
// Supported efforts and unknown models remain untouched.
func stripGeminiThinkingInContainer(container map[string]json.RawMessage, targetModel string) bool {
	if container == nil {
		return false
	}
	changed := false
	for _, gcKey := range []string{"generationConfig", "generation_config"} {
		gcRaw, hasGC := container[gcKey]
		if !hasGC || string(bytes.TrimSpace(gcRaw)) == "null" {
			continue
		}
		var gc map[string]json.RawMessage
		if json.Unmarshal(gcRaw, &gc) != nil || gc == nil {
			continue
		}
		gcChanged := false
		for _, tcKey := range []string{"thinkingConfig", "thinking_config"} {
			tcRaw, hasTC := gc[tcKey]
			if !hasTC || string(bytes.TrimSpace(tcRaw)) == "null" {
				continue
			}
			var tc map[string]json.RawMessage
			if json.Unmarshal(tcRaw, &tc) != nil || tc == nil {
				continue
			}
			effectiveEffort := ""
			for _, lk := range []string{"thinkingLevel", "thinking_level"} {
				if v := jsonStringValue(tc[lk]); v != "" {
					effectiveEffort = v
					break
				}
			}
			if effectiveEffort == "" {
				for _, bk := range []string{"thinkingBudget", "thinking_budget"} {
					if bRaw, ok := tc[bk]; ok && string(bytes.TrimSpace(bRaw)) != "null" {
						if b, okInt := jsonIntValue(bRaw); okInt {
							if lvl, okLvl := convertBudgetToLevel(b); okLvl {
								effectiveEffort = lvl
							}
						}
						break
					}
				}
			}
			if effectiveEffort != "" && !isReasoningEffortSupported(targetModel, effectiveEffort) {
				tcChanged := false
				for _, k := range []string{"thinkingLevel", "thinking_level", "thinkingBudget", "thinking_budget"} {
					if _, exists := tc[k]; exists {
						delete(tc, k)
						tcChanged = true
					}
				}
				if tcChanged {
					if len(tc) == 0 {
						delete(gc, tcKey)
					} else {
						gc[tcKey], _ = marshalNoEscape(tc)
					}
					gcChanged = true
				}
			}
		}
		for _, directKey := range []string{"thinkingLevel", "thinking_level"} {
			if v := jsonStringValue(gc[directKey]); v != "" && !isReasoningEffortSupported(targetModel, v) {
				delete(gc, directKey)
				gcChanged = true
			}
		}
		for _, directBKey := range []string{"thinkingBudget", "thinking_budget"} {
			if bRaw, ok := gc[directBKey]; ok && string(bytes.TrimSpace(bRaw)) != "null" {
				if b, okInt := jsonIntValue(bRaw); okInt {
					if lvl, okLvl := convertBudgetToLevel(b); okLvl && !isReasoningEffortSupported(targetModel, lvl) {
						delete(gc, directBKey)
						gcChanged = true
					}
				}
			}
		}
		if gcChanged {
			container[gcKey], _ = marshalNoEscape(gc)
			changed = true
		}
	}
	return changed
}

func StripUnsupportedSourceThinking(body []byte, sourceFormat, model, requestedModel string) ([]byte, bool) {
	if len(body) == 0 {
		return body, false
	}
	targetModel := resolveEffectiveEffortModel(model, requestedModel)
	if targetModel != "" {
		if _, restricted := resolveReasoningEffortSupport(targetModel); !restricted {
			return body, false
		}
	}

	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil || root == nil {
		return body, false
	}
	if targetModel == "" {
		var bodyModel string
		_ = json.Unmarshal(root["model"], &bodyModel)
		targetModel = bodyModel
		if _, restricted := resolveReasoningEffortSupport(targetModel); !restricted {
			return body, false
		}
	}

	changed := false
	switch strings.ToLower(strings.TrimSpace(sourceFormat)) {
	case "", "openai":
		if reRaw, hasRE := root["reasoning_effort"]; hasRE && string(bytes.TrimSpace(reRaw)) != "null" {
			var effort string
			_ = json.Unmarshal(reRaw, &effort)
			if !isReasoningEffortSupported(targetModel, effort) {
				delete(root, "reasoning_effort")
				changed = true
			}
		}

	case "openai-response":
		if rRaw, hasR := root["reasoning"]; hasR && string(bytes.TrimSpace(rRaw)) != "null" {
			var rMap map[string]json.RawMessage
			if json.Unmarshal(rRaw, &rMap) == nil && rMap != nil {
				if effort := jsonStringValue(rMap["effort"]); effort != "" && !isReasoningEffortSupported(targetModel, effort) {
					delete(rMap, "effort")
					if len(rMap) == 0 {
						delete(root, "reasoning")
					} else {
						root["reasoning"], _ = marshalNoEscape(rMap)
					}
					changed = true
				}
			}
		}
		if reRaw, hasRE := root["reasoning_effort"]; hasRE && string(bytes.TrimSpace(reRaw)) != "null" {
			if effort := jsonStringValue(reRaw); effort != "" && !isReasoningEffortSupported(targetModel, effort) {
				delete(root, "reasoning_effort")
				changed = true
			}
		}

	case "claude":
		var outCfg map[string]json.RawMessage
		if ocRaw, hasOC := root["output_config"]; hasOC {
			_ = json.Unmarshal(ocRaw, &outCfg)
		}
		outputEffort := ""
		if outCfg != nil {
			outputEffort = jsonStringValue(outCfg["effort"])
		}

		var thMap map[string]json.RawMessage
		hasThinking := false
		if thRaw, ok := root["thinking"]; ok && string(bytes.TrimSpace(thRaw)) != "null" {
			if json.Unmarshal(thRaw, &thMap) == nil && thMap != nil {
				hasThinking = true
			}
		}

		effectiveEffort := ""
		if hasThinking {
			switch jsonStringValue(thMap["type"]) {
			case "disabled":
				effectiveEffort = "none"
			case "adaptive", "auto":
				if outputEffort != "" {
					effectiveEffort = outputEffort
				} else {
					effectiveEffort = "xhigh"
				}
			case "enabled":
				if btRaw, hasBT := thMap["budget_tokens"]; hasBT && string(bytes.TrimSpace(btRaw)) != "null" {
					if b, ok := jsonIntValue(btRaw); ok {
						if lvl, okLvl := convertBudgetToLevel(b); okLvl {
							effectiveEffort = lvl
						}
					}
				} else if outputEffort != "" {
					effectiveEffort = outputEffort
				} else {
					effectiveEffort = "auto"
				}
			}
		} else if outputEffort != "" {
			effectiveEffort = outputEffort
		}

		if effectiveEffort != "" && !isReasoningEffortSupported(targetModel, effectiveEffort) {
			if hasThinking {
				delete(root, "thinking")
				changed = true
			}
			if outCfg != nil {
				if _, hasEff := outCfg["effort"]; hasEff {
					delete(outCfg, "effort")
					if len(outCfg) == 0 {
						delete(root, "output_config")
					} else {
						root["output_config"], _ = marshalNoEscape(outCfg)
					}
					changed = true
				}
			}
		} else if outputEffort != "" && !isReasoningEffortSupported(targetModel, outputEffort) && outCfg != nil {
			delete(outCfg, "effort")
			if len(outCfg) == 0 {
				delete(root, "output_config")
			} else {
				root["output_config"], _ = marshalNoEscape(outCfg)
			}
			changed = true
		}

	case "gemini":
		if stripGeminiThinkingInContainer(root, targetModel) {
			changed = true
		}
		if reqRaw, hasReq := root["request"]; hasReq && string(bytes.TrimSpace(reqRaw)) != "null" {
			var reqObj map[string]json.RawMessage
			if json.Unmarshal(reqRaw, &reqObj) == nil && reqObj != nil {
				if stripGeminiThinkingInContainer(reqObj, targetModel) {
					root["request"], _ = marshalNoEscape(reqObj)
					changed = true
				}
			}
		}
	}

	if !changed {
		return body, false
	}
	out, err := marshalNoEscape(root)
	if err != nil {
		return body, false
	}
	return out, true
}

// TransformMistralChatRequest performs pure wire-format normalization from OpenAI Chat Completions to Mistral Chat Completions.
// It preserves supported reasoning_effort values and omits known-unsupported ones so the upstream model uses its default.
func TransformMistralChatRequest(body []byte, envelopeModel string, cfg Config) ([]byte, bool) {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil || root == nil {
		return body, false
	}

	changed := false

	// 1. Rename "seed" -> "random_seed"
	if seedRaw, hasSeed := root["seed"]; hasSeed {
		if _, hasRandomSeed := root["random_seed"]; !hasRandomSeed && string(bytes.TrimSpace(seedRaw)) != "null" {
			root["random_seed"] = seedRaw
		}
		delete(root, "seed")
		changed = true
	}

	// 2. Strip forbidden top-level fields (422 extra_forbidden on Mistral)
	for field := range root {
		if _, forbidden := forbiddenTopLevelFieldSet[field]; forbidden {
			delete(root, field)
			changed = true
		}
	}

	// 3. Sanitize stream_options (Mistral rejects null and unknown keys like include_obfuscation / continuous_usage_stats)
	if soRaw, ok := root["stream_options"]; ok {
		if string(bytes.TrimSpace(soRaw)) == "null" {
			delete(root, "stream_options")
			changed = true
		} else {
			var so map[string]json.RawMessage
			if json.Unmarshal(soRaw, &so) == nil && so != nil {
				soChanged := false
				for k := range so {
					if k != "include_usage" {
						delete(so, k)
						soChanged = true
					}
				}
				if len(so) == 0 {
					delete(root, "stream_options")
					changed = true
				} else if soChanged {
					root["stream_options"], _ = marshalNoEscape(so)
					changed = true
				}
			}
		}
	}

	// 3b. Drop null values on strict Mistral fields and resolve max_tokens/max_completion_tokens conflict
	for _, nonNullField := range []string{"response_format", "parallel_tool_calls", "tool_choice", "tools", "service_tier"} {
		if rawVal, exists := root[nonNullField]; exists && string(bytes.TrimSpace(rawVal)) == "null" {
			delete(root, nonNullField)
			changed = true
		}
	}
	if rfRaw, hasRF := root["response_format"]; hasRF {
		var rf map[string]json.RawMessage
		if json.Unmarshal(rfRaw, &rf) == nil && rf != nil {
			if jsRaw, hasJS := rf["json_schema"]; hasJS && string(bytes.TrimSpace(jsRaw)) != "null" {
				var js map[string]json.RawMessage
				if json.Unmarshal(jsRaw, &js) == nil && js != nil {
					jsChanged := false
					if strictRaw, hasStrict := js["strict"]; hasStrict && string(bytes.TrimSpace(strictRaw)) == "null" {
						delete(js, "strict")
						jsChanged = true
					}
					if schemaRaw, hasSchema := js["schema"]; hasSchema && string(bytes.TrimSpace(schemaRaw)) != "null" {
						if normSchema, sChg := normalizeToolParametersSchema(schemaRaw); sChg {
							js["schema"] = normSchema
							jsChanged = true
						}
					}
					if jsChanged {
						rf["json_schema"], _ = marshalNoEscape(js)
						root["response_format"], _ = marshalNoEscape(rf)
						changed = true
					}
				}
			}
		}
	}
	if stRaw, hasST := root["service_tier"]; hasST {
		if string(bytes.TrimSpace(stRaw)) != "null" {
			var st string
			if json.Unmarshal(stRaw, &st) == nil && st != "auto" && st != "standard_only" {
				delete(root, "service_tier")
				changed = true
			}
		}
	}
	if tcRaw, hasTC := root["tool_choice"]; hasTC {
		var tcStr string
		if json.Unmarshal(tcRaw, &tcStr) == nil {
			switch tcStr {
			case "auto", "none", "any", "required":
			default:
				delete(root, "tool_choice")
				changed = true
			}
		} else {
			var tcObj map[string]json.RawMessage
			if json.Unmarshal(tcRaw, &tcObj) != nil || jsonStringValue(tcObj["type"]) != "function" {
				delete(root, "tool_choice")
				changed = true
			}
		}
	}
	_, hasMT := root["max_tokens"]
	mtIsNull := hasMT && string(bytes.TrimSpace(root["max_tokens"])) == "null"
	mctRaw, hasMCT := root["max_completion_tokens"]
	mctIsNull := hasMCT && string(bytes.TrimSpace(mctRaw)) == "null"
	if mtIsNull {
		delete(root, "max_tokens")
		changed = true
	}
	if hasMCT {
		if mctIsNull {
			delete(root, "max_completion_tokens")
			changed = true
		} else if hasMT && !mtIsNull {
			delete(root, "max_completion_tokens")
			changed = true
		}
	}

	// 4. Drop only known-unsupported efforts (upstream default); otherwise preserve protocol mutual exclusion.
	var bodyModel string
	_ = json.Unmarshal(root["model"], &bodyModel)
	targetModel := resolveEffectiveEffortModel(envelopeModel, bodyModel)
	if reRaw, hasRE := root["reasoning_effort"]; hasRE && string(bytes.TrimSpace(reRaw)) != "null" {
		var effort string
		_ = json.Unmarshal(reRaw, &effort)
		if !isReasoningEffortSupported(targetModel, effort) {
			delete(root, "reasoning_effort")
			if _, hasPM := root["prompt_mode"]; hasPM {
				delete(root, "prompt_mode")
			}
			changed = true
		} else {
			if normEffort := strings.ToLower(strings.TrimSpace(effort)); normEffort != "" && normEffort != effort {
				if _, restricted := resolveReasoningEffortSupport(targetModel); restricted {
					root["reasoning_effort"], _ = marshalNoEscape(normEffort)
					changed = true
				}
			}
			if _, hasPM := root["prompt_mode"]; hasPM {
				delete(root, "prompt_mode")
				changed = true
			}
		}
	} else if _, hasPM := root["prompt_mode"]; hasPM {
		if levels, restricted := resolveReasoningEffortSupport(targetModel); restricted && len(levels) == 0 {
			delete(root, "prompt_mode")
			changed = true
		}
	}

	// 5. Sanitize tools & normalize Gemini uppercase JSON Schema types
	if toolsRaw, ok := root["tools"]; ok && string(bytes.TrimSpace(toolsRaw)) != "null" {
		var tools []map[string]json.RawMessage
		if json.Unmarshal(toolsRaw, &tools) == nil {
			var kept []map[string]json.RawMessage
			toolsChanged := false
			for _, t := range tools {
				var typ string
				_ = json.Unmarshal(t["type"], &typ)
				if typ == "" {
					typ = "function"
				}
				if typ != "function" && typ != "image_generation" && typ != "document_library" && typ != "connector" {
					toolsChanged = true
					continue
				}
				if typ == "function" {
					var fn map[string]json.RawMessage
					if json.Unmarshal(t["function"], &fn) != nil || fn == nil {
						fn = map[string]json.RawMessage{}
						for _, k := range []string{"name", "description", "parameters", "strict"} {
							if v, exists := t[k]; exists {
								fn[k] = v
							}
						}
						toolsChanged = true
					} else {
						for k := range fn {
							if k != "name" && k != "description" && k != "parameters" && k != "strict" {
								delete(fn, k)
								toolsChanged = true
							}
						}
					}
					for _, optNull := range []string{"description", "strict"} {
						if v, exists := fn[optNull]; exists && string(bytes.TrimSpace(v)) == "null" {
							delete(fn, optNull)
							toolsChanged = true
						}
					}
					if pRaw, hasP := fn["parameters"]; !hasP || string(bytes.TrimSpace(pRaw)) == "null" {
						fn["parameters"] = json.RawMessage(`{"type":"object","properties":{}}`)
						toolsChanged = true
					} else if normP, pChg := normalizeToolParametersSchema(pRaw); pChg {
						fn["parameters"] = normP
						toolsChanged = true
					}
					fnBytes, _ := marshalNoEscape(fn)
					cleanTool := map[string]json.RawMessage{
						"type":     json.RawMessage(`"function"`),
						"function": fnBytes,
					}
					if len(t) != 2 || t["cache_control"] != nil {
						toolsChanged = true
					}
					kept = append(kept, cleanTool)
				} else {
					kept = append(kept, t)
				}
			}
			if toolsChanged || len(kept) == 0 {
				if len(kept) == 0 {
					delete(root, "tools")
					delete(root, "tool_choice")
				} else {
					root["tools"], _ = marshalNoEscape(kept)
				}
				changed = true
			}
		}
	}

	// 6. Transform messages
	var rawMsgs []json.RawMessage
	if err := json.Unmarshal(root["messages"], &rawMsgs); err == nil && len(rawMsgs) > 0 {
		msgsChanged, newMsgs := transformMessages(rawMsgs, cfg)
		if msgsChanged {
			root["messages"], _ = marshalNoEscape(newMsgs)
			changed = true
		}
	}

	if !changed {
		return body, false
	}
	out, err := marshalNoEscape(root)
	if err != nil {
		return body, false
	}
	return out, true
}

func extractMessageReasoning(m map[string]json.RawMessage) string {
	for _, f := range []string{"reasoning_content", "reasoning", "reasoning_text"} {
		if raw, ok := m[f]; ok {
			var s string
			if json.Unmarshal(raw, &s) == nil && s != "" {
				return s
			}
		}
	}
	if raw, ok := m["reasoning_details"]; ok {
		var details []struct {
			Text    string `json:"text"`
			Summary string `json:"summary"`
		}
		if json.Unmarshal(raw, &details) == nil {
			var b strings.Builder
			for _, d := range details {
				if d.Text != "" {
					b.WriteString(d.Text)
				} else if d.Summary != "" {
					b.WriteString(d.Summary)
				}
			}
			return b.String()
		}
	}
	return ""
}

func inferMIMETypeFromFilename(filename, data string) string {
	name := strings.ToLower(strings.TrimSpace(filename))
	clean := strings.SplitN(strings.SplitN(name, "?", 2)[0], "#", 2)[0]
	switch path.Ext(clean) {
	case ".pdf":
		return "application/pdf"
	case ".txt", ".md", ".log":
		return "text/plain"
	case ".csv":
		return "text/csv"
	case ".json":
		return "application/json"
	case ".html", ".htm":
		return "text/html"
	case ".docx":
		return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
	case ".pptx":
		return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
	case ".xlsx":
		return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".webp":
		return "image/webp"
	case ".gif":
		return "image/gif"
	case ".bmp":
		return "image/bmp"
	case ".tiff", ".tif":
		return "image/tiff"
	}
	trimmed := strings.TrimSpace(data)
	switch {
	case strings.HasPrefix(trimmed, "JVBERi"):
		return "application/pdf"
	case strings.HasPrefix(trimmed, "iVBORw0KGgo"):
		return "image/png"
	case strings.HasPrefix(trimmed, "/9j/"):
		return "image/jpeg"
	case strings.HasPrefix(trimmed, "UklGR"):
		return "image/webp"
	case strings.HasPrefix(trimmed, "R0lGOD"):
		return "image/gif"
	}
	return "application/octet-stream"
}

func normalizeFilePartDataURL(val, filename string) string {
	trimmed := strings.TrimSpace(val)
	low := strings.ToLower(trimmed)
	if strings.HasPrefix(low, "http://") || strings.HasPrefix(low, "https://") || strings.HasPrefix(low, "data:") {
		return trimmed
	}
	mime := inferMIMETypeFromFilename(filename, trimmed)
	return "data:" + mime + ";base64," + trimmed
}

func isImageFilePart(urlOrData, filename string) bool {
	low := strings.ToLower(strings.TrimSpace(urlOrData))
	if strings.HasPrefix(low, "data:image/") {
		return true
	}
	for _, target := range []string{low, strings.ToLower(strings.TrimSpace(filename))} {
		clean := strings.SplitN(strings.SplitN(target, "?", 2)[0], "#", 2)[0]
		for _, ext := range []string{".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"} {
			if strings.HasSuffix(clean, ext) {
				return true
			}
		}
	}
	return false
}

func sanitizeContentParts(raw json.RawMessage) (json.RawMessage, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return raw, false
	}
	var parts []map[string]json.RawMessage
	if json.Unmarshal(trimmed, &parts) != nil {
		return raw, false
	}
	changed := false
	for _, p := range parts {
		if _, hasCC := p["cache_control"]; hasCC {
			delete(p, "cache_control")
			changed = true
		}
		switch jsonStringValue(p["type"]) {
		case "input_text", "output_text":
			p["type"] = json.RawMessage(`"text"`)
			changed = true
		case "file":
			var fileObj struct {
				Filename string `json:"filename"`
				FileData string `json:"file_data"`
				FileURL  string `json:"file_url"`
			}
			if fRaw, ok := p["file"]; ok {
				_ = json.Unmarshal(fRaw, &fileObj)
			}
			docURL := strings.TrimSpace(fileObj.FileData)
			if docURL == "" {
				docURL = strings.TrimSpace(fileObj.FileURL)
			}
			if docURL == "" {
				var topData, topURL string
				_ = json.Unmarshal(p["file_data"], &topData)
				_ = json.Unmarshal(p["file_url"], &topURL)
				docURL = strings.TrimSpace(topData)
				if docURL == "" {
					docURL = strings.TrimSpace(topURL)
				}
			}
			if docURL != "" {
				docURL = normalizeFilePartDataURL(docURL, fileObj.Filename)
				if isImageFilePart(docURL, fileObj.Filename) {
					p["type"] = json.RawMessage(`"image_url"`)
					p["image_url"], _ = marshalNoEscape(map[string]string{"url": docURL})
				} else {
					p["type"] = json.RawMessage(`"document_url"`)
					p["document_url"], _ = marshalNoEscape(docURL)
					if name := strings.TrimSpace(fileObj.Filename); name != "" {
						p["document_name"], _ = marshalNoEscape(name)
					}
				}
				delete(p, "file")
				delete(p, "file_data")
				delete(p, "file_url")
				changed = true
			}
		}
	}
	if !changed {
		return raw, false
	}
	out, err := marshalNoEscape(parts)
	if err != nil {
		return raw, false
	}
	return out, true
}

func contentPartsHaveThinkingBlock(parts []json.RawMessage) bool {
	for _, p := range parts {
		var meta struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(p, &meta) == nil && strings.EqualFold(strings.TrimSpace(meta.Type), "thinking") {
			return true
		}
	}
	return false
}

func isEmptyAssistantContent(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" || string(trimmed) == `""` || string(trimmed) == "[]" {
		return true
	}
	var s string
	if json.Unmarshal(trimmed, &s) == nil {
		return strings.TrimSpace(s) == ""
	}
	return false
}

func transformMessages(rawMsgs []json.RawMessage, cfg Config) (bool, []json.RawMessage) {
	changed := false
	decoded := make([]map[string]json.RawMessage, 0, len(rawMsgs))

	usedIDs := map[string]bool{}
	idMap := map[string]string{}
	for _, rm := range rawMsgs {
		var m map[string]json.RawMessage
		if json.Unmarshal(rm, &m) != nil || m == nil {
			continue
		}
		decoded = append(decoded, m)
		if !cfg.NormalizeToolCallIDs {
			continue
		}
		var tcList []map[string]json.RawMessage
		if json.Unmarshal(m["tool_calls"], &tcList) == nil {
			for _, tc := range tcList {
				var id string
				_ = json.Unmarshal(tc["id"], &id)
				if validMistralToolID.MatchString(id) {
					usedIDs[id] = true
				}
			}
		}
		var tcid string
		if json.Unmarshal(m["tool_call_id"], &tcid) == nil && validMistralToolID.MatchString(tcid) {
			usedIDs[tcid] = true
		}
	}

	normalizeID := func(id string, fallbackSeed string) string {
		if !cfg.NormalizeToolCallIDs {
			return id
		}
		if validMistralToolID.MatchString(id) {
			return id
		}
		key := id
		if key == "" {
			key = fallbackSeed
		}
		if mapped, ok := idMap[key]; ok {
			return mapped
		}
		attempt := 0
		candidate := HashMistralToolID(key)
		for usedIDs[candidate] {
			attempt++
			candidate = HashMistralToolID(key + ":" + strconv.Itoa(attempt))
		}
		usedIDs[candidate] = true
		idMap[key] = candidate
		return candidate
	}

	outMsgs := make([]map[string]json.RawMessage, 0, len(decoded))

	for idx, m := range decoded {
		var role string
		_ = json.Unmarshal(m["role"], &role)

		if role == "developer" {
			role = "system"
			m["role"] = json.RawMessage(`"system"`)
			changed = true
		}

		if cRaw, ok := m["content"]; ok {
			if cleanC, cChanged := sanitizeContentParts(cRaw); cChanged {
				m["content"] = cleanC
				changed = true
			}
		}

		switch role {
		case "system":
			if _, hasContent := m["content"]; !hasContent || string(bytes.TrimSpace(m["content"])) == "null" {
				m["content"] = json.RawMessage(`""`)
				changed = true
			}
			for k := range m {
				if k != "role" && k != "content" {
					delete(m, k)
					changed = true
				}
			}
			outMsgs = append(outMsgs, m)

		case "user":
			if _, hasContent := m["content"]; !hasContent || string(bytes.TrimSpace(m["content"])) == "null" {
				m["content"] = json.RawMessage(`""`)
				changed = true
			}
			for k := range m {
				if k != "role" && k != "content" {
					delete(m, k)
					changed = true
				}
			}
			outMsgs = append(outMsgs, m)

		case "assistant":
			hadReasoningField := false
			for _, rf := range messageReasoningFields {
				if _, ok := m[rf]; ok {
					hadReasoningField = true
				}
			}
			if hadReasoningField {
				think := extractMessageReasoning(m)
				for _, rf := range messageReasoningFields {
					delete(m, rf)
				}
				changed = true
				if cfg.ConvertThinkingHistory && strings.TrimSpace(think) != "" {
					cTrim := bytes.TrimSpace(m["content"])
					var existingList []json.RawMessage
					var existingStr string
					if len(cTrim) > 0 && cTrim[0] == '[' && json.Unmarshal(cTrim, &existingList) == nil {
						if !contentPartsHaveThinkingBlock(existingList) {
							thinkChunk, _ := marshalNoEscape(map[string]any{
								"type":   "thinking",
								"closed": true,
								"thinking": []map[string]string{
									{"type": "text", "text": think},
								},
							})
							parts := append([]json.RawMessage{thinkChunk}, existingList...)
							m["content"], _ = marshalNoEscape(parts)
						}
					} else {
						thinkChunk, _ := marshalNoEscape(map[string]any{
							"type":   "thinking",
							"closed": true,
							"thinking": []map[string]string{
								{"type": "text", "text": think},
							},
						})
						parts := []json.RawMessage{thinkChunk}
						if json.Unmarshal(cTrim, &existingStr) == nil && existingStr != "" {
							textChunk, _ := marshalNoEscape(map[string]string{"type": "text", "text": existingStr})
							parts = append(parts, textChunk)
						}
						m["content"], _ = marshalNoEscape(parts)
					}
				}
			}

			hasToolCalls := false
			if tcRaw, ok := m["tool_calls"]; ok && string(bytes.TrimSpace(tcRaw)) != "null" {
				var tcList []map[string]json.RawMessage
				if json.Unmarshal(tcRaw, &tcList) == nil {
					if len(tcList) == 0 {
						delete(m, "tool_calls")
						changed = true
					} else {
						hasToolCalls = true
						tcModified := false
						for tcIdx, tc := range tcList {
							var origID string
							_ = json.Unmarshal(tc["id"], &origID)
							normID := normalizeID(origID, "tc_"+strconv.Itoa(idx)+"_"+strconv.Itoa(tcIdx))
							if normID != origID {
								tc["id"], _ = marshalNoEscape(normID)
								tcModified = true
							}
							for k := range tc {
								if k != "id" && k != "type" && k != "function" && k != "index" {
									delete(tc, k)
									tcModified = true
								}
							}
						}
						if tcModified {
							m["tool_calls"], _ = marshalNoEscape(tcList)
							changed = true
						}
					}
				}
			}

			for k := range m {
				if k != "role" && k != "content" && k != "tool_calls" && k != "prefix" {
					delete(m, k)
					changed = true
				}
			}

			if _, hasContent := m["content"]; !hasContent {
				m["content"] = json.RawMessage(`""`)
				changed = true
			}

			if idx < len(decoded)-1 && !hasToolCalls && isEmptyAssistantContent(m["content"]) {
				changed = true
				continue
			}

			outMsgs = append(outMsgs, m)

		case "tool":
			var origID string
			_ = json.Unmarshal(m["tool_call_id"], &origID)
			normID := normalizeID(origID, "tool_"+strconv.Itoa(idx))
			if normID != origID {
				m["tool_call_id"], _ = marshalNoEscape(normID)
				changed = true
			}
			if _, hasContent := m["content"]; !hasContent || string(bytes.TrimSpace(m["content"])) == "null" {
				m["content"] = json.RawMessage(`""`)
				changed = true
			}
			for k := range m {
				if k != "role" && k != "content" && k != "tool_call_id" && k != "name" {
					delete(m, k)
					changed = true
				}
			}
			outMsgs = append(outMsgs, m)

		default:
			outMsgs = append(outMsgs, m)
		}
	}

	for i, m := range outMsgs {
		var role string
		_ = json.Unmarshal(m["role"], &role)
		if role != "assistant" {
			continue
		}
		isLast := i == len(outMsgs)-1
		if !isLast {
			if _, hasPrefix := m["prefix"]; hasPrefix {
				delete(m, "prefix")
				changed = true
			}
		} else {
			_, hasTC := m["tool_calls"]
			if !hasTC || string(bytes.TrimSpace(m["tool_calls"])) == "null" {
				var pfx bool
				if json.Unmarshal(m["prefix"], &pfx) != nil || !pfx {
					m["prefix"] = json.RawMessage(`true`)
					changed = true
				}
			} else if _, hasPrefix := m["prefix"]; hasPrefix {
				delete(m, "prefix")
				changed = true
			}
		}
	}

	if !changed {
		return false, rawMsgs
	}
	res := make([]json.RawMessage, 0, len(outMsgs))
	for _, m := range outMsgs {
		b, err := marshalNoEscape(m)
		if err == nil {
			res = append(res, b)
		}
	}
	return true, res
}

var geminiSchemaTypes = map[string]string{
	"OBJECT":  "object",
	"STRING":  "string",
	"NUMBER":  "number",
	"INTEGER": "integer",
	"BOOLEAN": "boolean",
	"ARRAY":   "array",
	"NULL":    "null",
}

func normalizeSchemaValue(v any) (any, bool) {
	switch node := v.(type) {
	case map[string]any:
		changed := false
		if rawType, ok := node["type"].(string); ok {
			if mapped, isUpper := geminiSchemaTypes[strings.ToUpper(strings.TrimSpace(rawType))]; isUpper && mapped != rawType {
				node["type"] = mapped
				changed = true
			}
		} else if typeArr, ok := node["type"].([]any); ok {
			for i, elem := range typeArr {
				if es, isStr := elem.(string); isStr {
					if mapped, isUpper := geminiSchemaTypes[strings.ToUpper(strings.TrimSpace(es))]; isUpper && mapped != es {
						typeArr[i] = mapped
						changed = true
					}
				}
			}
		}
		if nullable, ok := node["nullable"].(bool); ok {
			delete(node, "nullable")
			changed = true
			if nullable {
				if tStr, isStr := node["type"].(string); isStr && tStr != "null" {
					node["type"] = []any{tStr, "null"}
				}
			}
		}
		for k, child := range node {
			if k == "properties" {
				if props, isMap := child.(map[string]any); isMap {
					for pk, pv := range props {
						if nv, chg := normalizeSchemaValue(pv); chg {
							props[pk] = nv
							changed = true
						}
					}
					continue
				}
			}
			if nv, chg := normalizeSchemaValue(child); chg {
				node[k] = nv
				changed = true
			}
		}
		return node, changed
	case []any:
		changed := false
		for i, item := range node {
			if nv, chg := normalizeSchemaValue(item); chg {
				node[i] = nv
				changed = true
			}
		}
		return node, changed
	default:
		return v, false
	}
}

func normalizeToolParametersSchema(raw json.RawMessage) (json.RawMessage, bool) {
	if len(raw) == 0 {
		return raw, false
	}
	var parsed any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return raw, false
	}
	normalized, changed := normalizeSchemaValue(parsed)
	if !changed {
		return raw, false
	}
	out, err := marshalNoEscape(normalized)
	if err != nil {
		return raw, false
	}
	return out, true
}

// extractTypedContentParts extracts plain text and thinking text from Mistral's polymorphic content array.
func extractTypedContentParts(raw json.RawMessage) (text string, think string, hasThinking bool, isArray bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return "", "", false, false
	}
	var parts []struct {
		Type     string          `json:"type"`
		Text     string          `json:"text"`
		Thinking json.RawMessage `json:"thinking"`
	}
	if err := json.Unmarshal(trimmed, &parts); err != nil {
		return "", "", false, false
	}
	var tb, kb strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "text":
			tb.WriteString(p.Text)
		case "thinking":
			hasThinking = true
			var s string
			if json.Unmarshal(p.Thinking, &s) == nil {
				kb.WriteString(s)
				continue
			}
			var inner []struct {
				Text string `json:"text"`
			}
			if json.Unmarshal(p.Thinking, &inner) == nil {
				for _, in := range inner {
					kb.WriteString(in.Text)
				}
			}
		default:
			// Preserve multimodal/unknown blocks (e.g. image_url, audio, refusal) untouched.
			return "", "", false, false
		}
	}
	return tb.String(), kb.String(), hasThinking, true
}

func hasNonEmptyToolCalls(m map[string]json.RawMessage) bool {
	raw, ok := m["tool_calls"]
	if !ok {
		return false
	}
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null")) && !bytes.Equal(trimmed, []byte("[]"))
}

func skipJSONSpace(b []byte, i int) int {
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	return i
}

// mayNeedMistralResponseChange performs a zero-allocation byte scan so normal
// string-content chunks can skip JSON decoding altogether.
func mayNeedMistralResponseChange(body []byte) bool {
	if bytes.Contains(body, []byte(`"model_length"`)) || bytes.Contains(body, []byte(`"response_error"`)) {
		return true
	}
	key := []byte(`"content"`)
	rest := body
	for {
		i := bytes.Index(rest, key)
		if i < 0 {
			return false
		}
		rest = rest[i+len(key):]
		j := skipJSONSpace(rest, 0)
		if j < len(rest) && rest[j] == ':' {
			j = skipJSONSpace(rest, j+1)
			if j < len(rest) && rest[j] == '[' {
				return true
			}
		}
	}
}

func flattenMsgOrDelta(m map[string]json.RawMessage, isWholeMessage, allowTextOnlyFlatten bool) bool {
	if m == nil {
		return false
	}
	text, think, hasThinking, isArr := extractTypedContentParts(m["content"])
	if !isArr || (!hasThinking && !allowTextOnlyFlatten) {
		return false
	}
	switch {
	case text != "":
		m["content"], _ = marshalNoEscape(text)
	case !isWholeMessage:
		delete(m, "content")
	case hasNonEmptyToolCalls(m):
		m["content"] = json.RawMessage("null")
	default:
		m["content"] = json.RawMessage(`""`)
	}
	if think != "" {
		var existing string
		_ = json.Unmarshal(m["reasoning_content"], &existing)
		m["reasoning_content"], _ = marshalNoEscape(existing + think)
	}
	return true
}

func transformSingleJSONResponse(raw []byte, allowTextOnlyFlatten bool) ([]byte, bool) {
	if !mayNeedMistralResponseChange(raw) {
		return raw, false
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil || obj == nil {
		return raw, false
	}
	var choices []map[string]json.RawMessage
	if err := json.Unmarshal(obj["choices"], &choices); err != nil || len(choices) == 0 {
		return raw, false
	}
	changed := false
	for _, c := range choices {
		var fr string
		if json.Unmarshal(c["finish_reason"], &fr) == nil && fr == "model_length" {
			c["finish_reason"] = json.RawMessage(`"length"`)
			changed = true
		}
		if _, hasRespErr := c["response_error"]; hasRespErr {
			delete(c, "response_error")
			changed = true
		}
		if msgRaw, ok := c["message"]; ok && string(bytes.TrimSpace(msgRaw)) != "null" {
			var msg map[string]json.RawMessage
			if json.Unmarshal(msgRaw, &msg) == nil && flattenMsgOrDelta(msg, true, allowTextOnlyFlatten) {
				c["message"], _ = marshalNoEscape(msg)
				changed = true
			}
		}
		if deltaRaw, ok := c["delta"]; ok && string(bytes.TrimSpace(deltaRaw)) != "null" {
			var delta map[string]json.RawMessage
			if json.Unmarshal(deltaRaw, &delta) == nil && flattenMsgOrDelta(delta, false, allowTextOnlyFlatten) {
				c["delta"], _ = marshalNoEscape(delta)
				changed = true
			}
		}
	}
	if !changed {
		return raw, false
	}
	obj["choices"], _ = marshalNoEscape(choices)
	out, err := marshalNoEscape(obj)
	if err != nil {
		return raw, false
	}
	return out, true
}

// TransformMistralResponse converts Mistral streaming SSE chunks or non-streaming JSON responses
// with typed thinking content arrays into standard OpenAI reasoning_content + string content.
func TransformMistralResponse(body []byte, stream bool) ([]byte, bool) {
	return TransformMistralResponseScoped(body, stream, true)
}

// TransformMistralResponseScoped behaves like TransformMistralResponse, except text-only
// content chunk lists (without any thinking chunk) are flattened only when allowTextOnlyFlatten is true.
func TransformMistralResponseScoped(body []byte, stream, allowTextOnlyFlatten bool) ([]byte, bool) {
	if len(body) == 0 || !mayNeedMistralResponseChange(body) {
		return body, false
	}

	trimmed := bytes.TrimSpace(body)
	if len(trimmed) > 0 && trimmed[0] == '{' {
		return transformSingleJSONResponse(body, allowTextOnlyFlatten)
	}

	lines := bytes.Split(body, []byte("\n"))
	anyChanged := false
	for i, line := range lines {
		cr := bytes.HasSuffix(line, []byte("\r"))
		clean := bytes.TrimRight(line, "\r")
		trimmedLeft := bytes.TrimLeft(clean, " \t")
		if !bytes.HasPrefix(trimmedLeft, []byte("data:")) {
			continue
		}
		payload := bytes.TrimSpace(trimmedLeft[len("data:"):])
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) || payload[0] != '{' {
			continue
		}
		if newPayload, chg := transformSingleJSONResponse(payload, allowTextOnlyFlatten); chg {
			newLine := append([]byte("data: "), newPayload...)
			if cr {
				newLine = append(newLine, '\r')
			}
			lines[i] = newLine
			anyChanged = true
		}
	}
	if !anyChanged {
		return body, false
	}
	return bytes.Join(lines, []byte("\n")), true
}
