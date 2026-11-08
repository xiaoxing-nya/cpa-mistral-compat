#include "cpa_bridge.h"

static const cliproxy_host_api* g_stored_host = NULL;

void cpa_store_host_api(const cliproxy_host_api* host) {
    g_stored_host = host;
}

int cpa_call_host_api(const char* method, const uint8_t* request, size_t request_len, cliproxy_buffer* response) {
    if (g_stored_host == NULL || g_stored_host->call == NULL) {
        return 1;
    }
    return g_stored_host->call(g_stored_host->host_ctx, method, request, request_len, response);
}

void cpa_free_host_buffer(void* ptr, size_t len) {
    if (g_stored_host != NULL && g_stored_host->free_buffer != NULL && ptr != NULL) {
        g_stored_host->free_buffer(ptr, len);
    }
}
