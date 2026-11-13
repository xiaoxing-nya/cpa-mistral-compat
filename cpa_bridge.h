#ifndef CPA_BRIDGE_H
#define CPA_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    void* ptr;
    size_t len;
} cliproxy_buffer;

typedef int (*cliproxy_host_call_fn)(void*, const char*, const uint8_t*, size_t, cliproxy_buffer*);
typedef void (*cliproxy_host_free_fn)(void*, size_t);

typedef struct {
    uint32_t abi_version;
    void* host_ctx;
    cliproxy_host_call_fn call;
    cliproxy_host_free_fn free_buffer;
} cliproxy_host_api;

typedef int (*cliproxy_plugin_call_fn)(char*, uint8_t*, size_t, cliproxy_buffer*);
typedef void (*cliproxy_plugin_free_fn)(void*, size_t);
typedef void (*cliproxy_plugin_shutdown_fn)(void);

typedef struct {
    uint32_t abi_version;
    cliproxy_plugin_call_fn call;
    cliproxy_plugin_free_fn free_buffer;
    cliproxy_plugin_shutdown_fn shutdown;
} cliproxy_plugin_api;

#if defined(_WIN32)
#define CPA_INTERNAL
#else
#define CPA_INTERNAL __attribute__((visibility("hidden")))
#endif

CPA_INTERNAL void cpa_store_host_api(const cliproxy_host_api* host);
CPA_INTERNAL int cpa_call_host_api(const char* method, const uint8_t* request, size_t request_len, cliproxy_buffer* response);
CPA_INTERNAL void cpa_free_host_buffer(void* ptr, size_t len);
CPA_INTERNAL void* cpa_copy_out(const void* src, size_t len);

extern int cliproxyPluginCall(char*, uint8_t*, size_t, cliproxy_buffer*);
extern void cliproxyPluginFree(void*, size_t);
extern void cliproxyPluginShutdown(void);

#endif
