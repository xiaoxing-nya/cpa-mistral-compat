#include <string.h>

#include "cpa_bridge.h"

static const cliproxy_host_api* g_stored_host = NULL;

void cpa_store_host_api(const cliproxy_host_api* host) {
    __atomic_store_n(&g_stored_host, host, __ATOMIC_RELEASE);
}

int cpa_call_host_api(const char* method, const uint8_t* request, size_t request_len, cliproxy_buffer* response) {
    const cliproxy_host_api* host = __atomic_load_n(&g_stored_host, __ATOMIC_ACQUIRE);
    if (host == NULL || host->call == NULL) {
        return 1;
    }
    return host->call(host->host_ctx, method, request, request_len, response);
}

void cpa_free_host_buffer(void* ptr, size_t len) {
    const cliproxy_host_api* host = __atomic_load_n(&g_stored_host, __ATOMIC_ACQUIRE);
    if (host != NULL && host->free_buffer != NULL && ptr != NULL) {
        host->free_buffer(ptr, len);
    }
}

void* cpa_copy_out(const void* src, size_t len) {
    if (src == NULL || len == 0) {
        return NULL;
    }
    void* dst = malloc(len);
    if (dst != NULL) {
        memcpy(dst, src, len);
    }
    return dst;
}
