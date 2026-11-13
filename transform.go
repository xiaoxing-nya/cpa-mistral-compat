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
	"audio",
	"modalities",
	"logit_bias",
}

var messageReasoningFields = []string{"reasoning_content", "reasoning", "reasoning_text", "reasoning_details"}

// restrictedReasoningEffortLevels lists only models with explicitly constrained wire-level values.
// Unknown models remain pass-through; no requested level is ever remapped to another level.
var restrictedReasoningEffortLevels = []struct {
	modelPattern string
	levels       []string
}{
	// DeepSeek: supports none, low, high, xhigh, max.
	{modelPattern: "*deepseek*", levels: []string{"none", "low", "high", "xhigh", "max"}},

	// GLM 5.2 (zai-glm-5-2, glm-5-2, glm-5.2): supports none, high, max (low is NOT supported).
	// Must come before *zai-glm* so zai-glm-5-2 gets the correct 5.2 levels, not the 5.3 fallback.
	{modelPattern: "*glm-5?2*", levels: []string{"none", "high", "max"}},

	// GLM 5.3 and GLM 5.3 Flash (zai-glm-5-3, zai-glm-5-3-flash, glm-5.3, etc.): supports low, high, max (none is NOT supported).
	{modelPattern: "*glm-5?3*", levels: []string{"low", "high", "max"}},

	// Fallback for other zai-glm aliases (zai-glm-5, zai-glm-latest → GLM 5.3 levels).
	{modelPattern: "*zai-glm*", levels: []string{"low", "high", "max"}},

	// Mistral Large 4 supports low, medium, high.
	{modelPattern: "mistral-large-4*", levels: []string{"low", "medium", "high"}},

	// Mistral Small 4 and Mistral Medium 3.5 support none and high.
	{modelPattern: "mistral-small-2603*", levels: []string{"none", "high"}},
	{modelPattern: "mistral-small-4*", levels: []string{"none", "high"}},
	{modelPattern: "mistral-medium-3-5*", levels: []string{"none", "high"}},

	// Legacy Mistral Large models (mistral-large-latest, 2407, 2411) do not support reasoning_effort.
	// Dropping reasoning_effort prevents upstream "reasoning_effort is not enabled for this model" errors.
	{modelPattern: "mistral-large*", levels: []string{}},
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
	for _, level := range levels {
		if effort == level {
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
func StripUnsupportedSourceThinking(body []byte, sourceFormat, model, requestedModel string) ([]byte, bool) {
	if len(body) == 0 {
		return body, false
	}
	targetModel := resolveEffectiveEffortModel(model, requestedModel)
	if _, restricted := resolveReasoningEffortSupport(targetModel); !restricted {
		return body, false
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
						root["reasoning"], _ = json.Marshal(rMap)
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
						root["output_config"], _ = json.Marshal(outCfg)
					}
					changed = true
				}
			}
		}

	case "gemini":
		if gcRaw, hasGC := root["generationConfig"]; hasGC && string(bytes.TrimSpace(gcRaw)) != "null" {
			var gc map[string]json.RawMessage
			if json.Unmarshal(gcRaw, &gc) == nil && gc != nil {
				if tcRaw, hasTC := gc["thinkingConfig"]; hasTC && string(bytes.TrimSpace(tcRaw)) != "null" {
					var tc map[string]json.RawMessage
					if json.Unmarshal(tcRaw, &tc) == nil && tc != nil {
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
							for _, k := range []string{"thinkingLevel", "thinking_level", "thinkingBudget", "thinking_budget"} {
								if _, exists := tc[k]; exists {
									delete(tc, k)
									changed = true
								}
							}
							if changed {
								if len(tc) == 0 {
									delete(gc, "thinkingConfig")
								} else {
									gc["thinkingConfig"], _ = json.Marshal(tc)
								}
								root["generationConfig"], _ = json.Marshal(gc)
							}
						}
					}
				}
			}
		}
	}

	if !changed {
		return body, false
	}
	out, err := json.Marshal(root)
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
	for _, field := range forbiddenTopLevelFields {
		if _, ok := root[field]; ok {
			delete(root, field)
			changed = true
		}
	}

	// 3. Sanitize stream_options (Mistral only accepts include_usage; include_obfuscation is rejected)
	if soRaw, ok := root["stream_options"]; ok {
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
				root["stream_options"], _ = json.Marshal(so)
				changed = true
			}
		}
	}

	// 4. Drop only known-unsupported efforts (upstream default); otherwise preserve protocol mutual exclusion.
	if reRaw, hasRE := root["reasoning_effort"]; hasRE && string(bytes.TrimSpace(reRaw)) != "null" {
		var bodyModel, effort string
		_ = json.Unmarshal(root["model"], &bodyModel)
		_ = json.Unmarshal(reRaw, &effort)
		targetModel := resolveEffectiveEffortModel(envelopeModel, bodyModel)
		if !isReasoningEffortSupported(targetModel, effort) {
			delete(root, "reasoning_effort")
			changed = true
		} else if _, hasPM := root["prompt_mode"]; hasPM {
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
					if pRaw, hasP := fn["parameters"]; hasP {
						if normP, pChg := normalizeToolParametersSchema(pRaw); pChg {
							fn["parameters"] = normP
							toolsChanged = true
						}
					}
					fnBytes, _ := json.Marshal(fn)
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
			if toolsChanged {
				if len(kept) == 0 {
					delete(root, "tools")
					delete(root, "tool_choice")
				} else {
					root["tools"], _ = json.Marshal(kept)
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
			root["messages"], _ = json.Marshal(newMsgs)
			changed = true
		}
	}

	if !changed {
		return body, false
	}
	out, err := json.Marshal(root)
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
	}
	if !changed {
		return raw, false
	}
	out, err := json.Marshal(parts)
	if err != nil {
		return raw, false
	}
	return out, true
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
			for k := range m {
				if k != "role" && k != "content" {
					delete(m, k)
					changed = true
				}
			}
			outMsgs = append(outMsgs, m)

		case "user":
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
					thinkChunk, _ := json.Marshal(map[string]any{
						"type":   "thinking",
						"closed": true,
						"thinking": []map[string]string{
							{"type": "text", "text": think},
						},
					})
					parts := []json.RawMessage{thinkChunk}
					cTrim := bytes.TrimSpace(m["content"])
					var existingList []json.RawMessage
					var existingStr string
					if len(cTrim) > 0 && cTrim[0] == '[' && json.Unmarshal(cTrim, &existingList) == nil {
						parts = append(parts, existingList...)
					} else if json.Unmarshal(cTrim, &existingStr) == nil && existingStr != "" {
						textChunk, _ := json.Marshal(map[string]string{"type": "text", "text": existingStr})
						parts = append(parts, textChunk)
					}
					m["content"], _ = json.Marshal(parts)
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
								tc["id"], _ = json.Marshal(normID)
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
							m["tool_calls"], _ = json.Marshal(tcList)
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
				m["tool_call_id"], _ = json.Marshal(normID)
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
		b, err := json.Marshal(m)
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
	out, err := json.Marshal(normalized)
	if err != nil {
		return raw, false
	}
	return out, true
}

// extractTypedContentParts extracts plain text and thinking text from Mistral's polymorphic content array.
func extractTypedContentParts(raw json.RawMessage) (text string, think string, isArray bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return "", "", false
	}
	var parts []struct {
		Type     string          `json:"type"`
		Text     string          `json:"text"`
		Thinking json.RawMessage `json:"thinking"`
	}
	if err := json.Unmarshal(trimmed, &parts); err != nil {
		return "", "", false
	}
	var tb, kb strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "text":
			tb.WriteString(p.Text)
		case "thinking":
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
			return "", "", false
		}
	}
	return tb.String(), kb.String(), true
}

func flattenMsgOrDelta(m map[string]json.RawMessage, isWholeMessage bool) bool {
	if m == nil {
		return false
	}
	text, think, isArr := extractTypedContentParts(m["content"])
	if !isArr {
		return false
	}
	if text != "" || isWholeMessage {
		m["content"], _ = json.Marshal(text)
	} else {
		delete(m, "content")
	}
	if think != "" {
		var existing string
		_ = json.Unmarshal(m["reasoning_content"], &existing)
		m["reasoning_content"], _ = json.Marshal(existing + think)
	}
	return true
}

func transformSingleJSONResponse(raw []byte) ([]byte, bool) {
	if !bytes.Contains(raw, []byte(`"content":[`)) &&
		!bytes.Contains(raw, []byte(`"content": [`)) &&
		!bytes.Contains(raw, []byte(`"model_length"`)) {
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
		if msgRaw, ok := c["message"]; ok && string(bytes.TrimSpace(msgRaw)) != "null" {
			var msg map[string]json.RawMessage
			if json.Unmarshal(msgRaw, &msg) == nil && flattenMsgOrDelta(msg, true) {
				c["message"], _ = json.Marshal(msg)
				changed = true
			}
		}
		if deltaRaw, ok := c["delta"]; ok && string(bytes.TrimSpace(deltaRaw)) != "null" {
			var delta map[string]json.RawMessage
			if json.Unmarshal(deltaRaw, &delta) == nil && flattenMsgOrDelta(delta, false) {
				c["delta"], _ = json.Marshal(delta)
				changed = true
			}
		}
	}
	if !changed {
		return raw, false
	}
	obj["choices"], _ = json.Marshal(choices)
	out, err := json.Marshal(obj)
	if err != nil {
		return raw, false
	}
	return out, true
}

// TransformMistralResponse converts Mistral streaming SSE chunks or non-streaming JSON responses
// with typed thinking content arrays into standard OpenAI reasoning_content + string content.
func TransformMistralResponse(body []byte, stream bool) ([]byte, bool) {
	if len(body) == 0 {
		return body, false
	}
	if !bytes.Contains(body, []byte(`"content":[`)) &&
		!bytes.Contains(body, []byte(`"content": [`)) &&
		!bytes.Contains(body, []byte(`"model_length"`)) {
		return body, false
	}

	trimmed := bytes.TrimSpace(body)
	if len(trimmed) > 0 && trimmed[0] == '{' {
		return transformSingleJSONResponse(body)
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
		if newPayload, chg := transformSingleJSONResponse(payload); chg {
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
