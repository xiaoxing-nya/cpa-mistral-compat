PLUGIN_NAME := cpa-mistral-compat
HOST_ARCH   := $(shell uname -m)

ifndef CC_ARM64
ifeq ($(HOST_ARCH),aarch64)
CC_ARM64 := gcc
else ifneq ($(shell command -v aarch64-linux-gnu-gcc 2>/dev/null),)
CC_ARM64 := aarch64-linux-gnu-gcc
else ifneq ($(wildcard /tmp/zig/zig),)
CC_ARM64 := /tmp/zig/zig cc -target aarch64-linux-gnu.2.17
endif
endif

.PHONY: test build-amd64 build-arm64 clean

test:
	go vet ./...
	go test -race -v ./...

build-amd64:
	mkdir -p dist
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -buildmode=c-shared -o dist/$(PLUGIN_NAME)-linux-amd64.so .
	rm -f dist/*.h

build-arm64:
	@test -n "$(CC_ARM64)" || { echo "arm64 cross-compiler not found; set CC_ARM64 (e.g. aarch64-linux-gnu-gcc)" >&2; exit 1; }
	mkdir -p dist
	CC="$(CC_ARM64)" CGO_ENABLED=1 GOOS=linux GOARCH=arm64 go build -trimpath -ldflags="-s -w" -buildmode=c-shared -o dist/$(PLUGIN_NAME)-linux-arm64.so .
	rm -f dist/*.h

clean:
	rm -rf dist/
