PLUGIN_NAME := cpa-mistral-compat

.PHONY: test build-amd64 build-arm64 clean

test:
	go vet ./...
	go test -v ./...

build-amd64:
	mkdir -p dist
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -buildmode=c-shared -o dist/$(PLUGIN_NAME)-linux-amd64.so .
	rm -f dist/*.h

build-arm64:
	mkdir -p dist
	CGO_ENABLED=1 GOOS=linux GOARCH=arm64 go build -trimpath -ldflags="-s -w" -buildmode=c-shared -o dist/$(PLUGIN_NAME)-linux-arm64.so .
	rm -f dist/*.h

clean:
	rm -rf dist/
