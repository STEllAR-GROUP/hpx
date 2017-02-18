/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
 * Copyright (c) 2014-2016, Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2015 Los Alamos Nat. Security, LLC. All rights reserved.
 * Copyright (c) 2016 Cray Inc.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <assert.h>
#include <getopt.h>
#include <inttypes.h>
#include <netdb.h>
#include <poll.h>
#include <limits.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>

#ifdef HPX_HAVE_PARCELPORT_VERBS
# include "../libfabric/config-verbs.h"
#else
# include "../libfabric/config-gni.h"
#endif

#ifndef PP_FIVERSION
#define PP_FIVERSION FI_VERSION(1, 4)
#endif

#ifdef __APPLE__
#include "osx/osd.h"
#elif defined __FreeBSD__
#include "freebsd/osd.h"
#endif

enum precision {
    NANO = 1,
    MICRO = 1000,
    MILLI = 1000000,
};

enum {
    PP_OPT_ACTIVE = 1 << 0,
    PP_OPT_ITER = 1 << 1,
    PP_OPT_SIZE = 1 << 2,
    PP_OPT_VERIFY_DATA = 1 << 3,
};

struct pp_opts {
    uint16_t src_port;
    uint16_t dst_port;
    char *dst_addr;
    int iterations;
    int transfer_size;
    int sizes_enabled;
    int options;
};

#define PP_SIZE_MAX_POWER_TWO 22
#define PP_MAX_DATA_MSG                                                        \
    ((1 << PP_SIZE_MAX_POWER_TWO) + (1 << (PP_SIZE_MAX_POWER_TWO - 1)))

#define PP_STR_LEN 32
#define PP_MAX_CTRL_MSG 64
#define PP_CTRL_BUF_LEN 64
#define PP_MR_KEY 0xC0DE

#define INTEG_SEED 7
#define PP_ENABLE_ALL (~0)
#define PP_DEFAULT_SIZE (1 << 0)

#define PP_MSG_CHECK_PORT_OK "port ok"
#define PP_MSG_LEN_PORT 5
#define PP_MSG_CHECK_CNT_OK "cnt ok"
#define PP_MSG_LEN_CNT 10
#define PP_MSG_SYNC_Q "q"
#define PP_MSG_SYNC_A "a"

#define PP_PRINTERR(call, retv)                                                \
    fprintf(stderr, "%s(): %s:%-4d, ret=%d (%s)\n", call, __FILE__,        \
        __LINE__, (int)retv, fi_strerror((int) -retv))

#define PP_ERR(fmt, ...)                                                       \
    fprintf(stderr, "[%s] %s:%-4d: " fmt "\n", "error", __FILE__,          \
        __LINE__, ##__VA_ARGS__)

int pp_debug = 1;

#define PP_DEBUG(fmt, ...)                                                     \
    do {                                                                   \
        if (pp_debug) {                                                \
            fprintf(stderr, "[%s] %s:%-4d: " fmt, "debug",         \
                __FILE__, __LINE__, ##__VA_ARGS__);            \
        }                                                              \
    } while (0)

#define PP_CLOSE_FID(fd)                                                       \
    do {                                                                   \
        int ret;                                                       \
        if ((fd)) {                                                    \
            ret = fi_close(&(fd)->fid);                            \
            if (ret)                                               \
                PP_ERR("fi_close (%d) fid %d", ret,            \
                       (int)(fd)->fid.fclass);                 \
            fd = NULL;                                             \
        }                                                              \
    } while (0)

#ifndef MAX
#define MAX(a, b)                                                              \
    ({                                                                     \
        typeof(a) _a = (a);                                            \
        typeof(b) _b = (b);                                            \
        _a > _b ? _a : _b;                                             \
    })
#endif

#ifndef MIN
#define MIN(a, b)                                                              \
    ({                                                                     \
        typeof(a) _a = (a);                                            \
        typeof(b) _b = (b);                                            \
        _a < _b ? _a : _b;                                             \
    })
#endif

struct ct_pingpong {
    struct fi_info *fi_pep, *fi, *hints;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_pep *pep;
    struct fid_ep *ep;
    struct fid_cq *txcq, *rxcq;
    struct fid_mr *mr;
    struct fid_av *av;
    struct fid_eq *eq;

    struct fid_mr no_mr;
    struct fi_context tx_ctx, rx_ctx;
    uint64_t remote_cq_data;

    uint64_t tx_seq, rx_seq, tx_cq_cntr, rx_cq_cntr;

    fi_addr_t remote_fi_addr;
    void *buf, *tx_buf, *rx_buf;
    size_t buf_size, tx_size, rx_size;

    int timeout_sec;
    uint64_t start, end;

    struct fi_av_attr av_attr;
    struct fi_eq_attr eq_attr;
    struct fi_cq_attr cq_attr;
    struct pp_opts opts;

    long cnt_ack_msg;

    int ctrl_connfd;
    char ctrl_buf[PP_CTRL_BUF_LEN + 1];
    char rem_name[PP_MAX_CTRL_MSG];
};

static const char integ_alphabet[] =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
/* Size does not include trailing new line */
static const int integ_alphabet_length =
    (sizeof(integ_alphabet) / sizeof(*integ_alphabet)) - 1;

/*******************************************************************************
 *                                         Utils
 ******************************************************************************/

uint64_t pp_gettime_us(void)
{
    struct timeval now;

    gettimeofday(&now, NULL);
    return now.tv_sec * 1000000 + now.tv_usec;
}

long parse_ulong(char *str, long max)
{
    long ret;
    char *end;

    errno = 0;
    ret = strtol(str, &end, 10);
    if (*end != '\0' || errno != 0) {
        if (errno == 0)
            ret = -EINVAL;
        else
            ret = -errno;
        fprintf(stderr, "Error parsing \"%s\": %s\n", str,
            strerror(-ret));
        return ret;
    }

    if ((ret < 0) || (max > 0 && ret > max)) {
        ret = -ERANGE;
        fprintf(stderr, "Error parsing \"%s\": %s\n", str,
            strerror(-ret));
        return ret;
    }
    return ret;
}

int size_to_count(int size)
{
    if (size >= (1 << 20))
        return 100;
    else if (size >= (1 << 16))
        return 1000;
    else
        return 10000;
}

void pp_banner_fabric_info(struct ct_pingpong *ct)
{
    PP_DEBUG(
        "Running pingpong test with the %s endpoint through a %s provider\n",
        fi_tostr(&ct->fi->ep_attr->type, FI_TYPE_EP_TYPE),
        ct->fi->fabric_attr->prov_name);
    PP_DEBUG(" * Fabric Attributes:\n");
    PP_DEBUG("  - %-20s: %s\n", "name", ct->fi->fabric_attr->name);
    PP_DEBUG("  - %-20s: %s\n", "prov_name",
         ct->fi->fabric_attr->prov_name);
    PP_DEBUG("  - %-20s: %" PRIu32 "\n", "prov_version",
         ct->fi->fabric_attr->prov_version);
    PP_DEBUG(" * Domain Attributes:\n");
    PP_DEBUG("  - %-20s: %s\n", "name", ct->fi->domain_attr->name);
    PP_DEBUG("  - %-20s: %zu\n", "cq_cnt", ct->fi->domain_attr->cq_cnt);
    PP_DEBUG("  - %-20s: %zu\n", "cq_data_size",
         ct->fi->domain_attr->cq_data_size);
    PP_DEBUG("  - %-20s: %zu\n", "ep_cnt", ct->fi->domain_attr->ep_cnt);
    PP_DEBUG(" * Endpoint Attributes:\n");
    PP_DEBUG("  - %-20s: %s\n", "type",
         fi_tostr(&ct->fi->ep_attr->type, FI_TYPE_EP_TYPE));
    PP_DEBUG("  - %-20s: %" PRIu32 "\n", "protocol",
         ct->fi->ep_attr->protocol);
    PP_DEBUG("  - %-20s: %" PRIu32 "\n", "protocol_version",
         ct->fi->ep_attr->protocol_version);
    PP_DEBUG("  - %-20s: %zu\n", "max_msg_size",
         ct->fi->ep_attr->max_msg_size);
    PP_DEBUG("  - %-20s: %zu\n", "max_order_raw_size",
         ct->fi->ep_attr->max_order_raw_size);
}

void pp_banner_options(struct ct_pingpong *ct)
{
    char size_msg[50];
    char iter_msg[50];
    struct pp_opts opts = ct->opts;

    if ((opts.dst_addr == NULL) || (opts.dst_addr[0] == '\0'))
        opts.dst_addr = "None";

    if (opts.sizes_enabled == PP_ENABLE_ALL)
        snprintf(size_msg, 50, "%s", "All sizes");
    else if (opts.options & PP_OPT_SIZE)
        snprintf(size_msg, 50, "selected size = %d",
             opts.transfer_size);
    else
        snprintf(size_msg, 50, "nasty bug");

    if (opts.options & PP_OPT_ITER)
        snprintf(iter_msg, 50, "selected iterations: %d",
             opts.iterations);
    else {
        opts.iterations = size_to_count(opts.transfer_size);
        snprintf(iter_msg, 50, "default iterations: %d",
             opts.iterations);
    }

    PP_DEBUG(" * PingPong options:\n");
    PP_DEBUG("  - %-20s: [%" PRIu16 "]\n", "src_port", opts.src_port);
    PP_DEBUG("  - %-20s: [%s]\n", "dst_addr", opts.dst_addr);
    PP_DEBUG("  - %-20s: [%" PRIu16 "]\n", "dst_port", opts.dst_port);
    PP_DEBUG("  - %-20s: %s\n", "sizes_enabled", size_msg);
    PP_DEBUG("  - %-20s: %s\n", "iterations", iter_msg);
    if (ct->hints->fabric_attr->prov_name)
        PP_DEBUG("  - %-20s: %s\n", "provider",
              ct->hints->fabric_attr->prov_name);
    if (ct->hints->domain_attr->name)
        PP_DEBUG("  - %-20s: %s\n", "domain",
              ct->hints->domain_attr->name);
}

/*******************************************************************************
 *                                         Control Messaging
 ******************************************************************************/

int pp_getaddrinfo(char *name, uint16_t port, struct addrinfo **results)
{
    int ret;
    const char *err_msg;
    char port_s[6];

    struct addrinfo hints = {
        .ai_family = AF_INET,       /* IPv4 */
        .ai_socktype = SOCK_STREAM, /* TCP socket */
        .ai_protocol = IPPROTO_TCP, /* Any protocol */
        .ai_flags = AI_NUMERICSERV /* numeric port is used */
    };

    snprintf(port_s, 6, "%" PRIu16, port);

    ret = getaddrinfo(name, port_s, &hints, results);
    if (ret != 0) {
        err_msg = gai_strerror(ret);
        PP_ERR("getaddrinfo : %s", err_msg);
        ret = -EXIT_FAILURE;
        goto out;
    }
    ret = EXIT_SUCCESS;

out:
    return ret;
}

static int pp_ctrl_init_client(struct ct_pingpong *ct)
{
    struct sockaddr_in in_addr = {0};
    struct addrinfo *results;
    struct addrinfo *rp;
    int errno_save;
    int ret;

    ret = pp_getaddrinfo(ct->opts.dst_addr, ct->opts.dst_port, &results);
    if (ret)
        return ret;

    if (!results) {
        PP_ERR("getaddrinfo returned NULL list");
        return -EXIT_FAILURE;
    }

    for (rp = results; rp; rp = rp->ai_next) {
        ct->ctrl_connfd = socket(rp->ai_family, rp->ai_socktype,
                     rp->ai_protocol);
        if (ct->ctrl_connfd == -1) {
            errno_save = errno;
            continue;
        }

        if (ct->opts.src_port != 0) {
            in_addr.sin_family = AF_INET;
            in_addr.sin_port = htons(ct->opts.src_port);
            in_addr.sin_addr.s_addr = htonl(INADDR_ANY);

            ret =
                bind(ct->ctrl_connfd, (struct sockaddr *)&in_addr,
                 sizeof(in_addr));
            if (ret == -1) {
                errno_save = errno;
                close(ct->ctrl_connfd);
                continue;
            }
        }

        ret = connect(ct->ctrl_connfd, rp->ai_addr, rp->ai_addrlen);
        if (ret != -1)
            break;

        errno_save = errno;
        close(ct->ctrl_connfd);
    }

    if (!rp || ret == -1) {
        ret = -errno_save;
        ct->ctrl_connfd = -1;
        PP_ERR("failed to connect: %s", strerror(errno_save));
    } else {
        PP_DEBUG("CLIENT: connected\n");
    }

    freeaddrinfo(results);

    return ret;
}

static int pp_ctrl_init_server(struct ct_pingpong *ct)
{
    struct sockaddr_in ctrl_addr = {0};
    int optval = 1;
    int listenfd;
    int ret;

    listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenfd == -1) {
        ret = -errno;
        PP_PRINTERR("socket", ret);
        return ret;
    }

    ret = setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR,
            &optval, sizeof(optval));
    if (ret == -1) {
        ret = -errno;
        PP_PRINTERR("setsockopt(SO_REUSEADDR)", ret);
        goto fail_close_socket;
    }

    ctrl_addr.sin_family = AF_INET;
    ctrl_addr.sin_port = htons(ct->opts.src_port);
    ctrl_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    ret = bind(listenfd, (struct sockaddr *)&ctrl_addr,
           sizeof(ctrl_addr));
    if (ret == -1) {
        ret = -errno;
        PP_PRINTERR("bind", ret);
        goto fail_close_socket;
    }

    ret = listen(listenfd, 10);
    if (ret == -1) {
        ret = -errno;
        PP_PRINTERR("listen", ret);
        goto fail_close_socket;
    }

    PP_DEBUG("SERVER: waiting for connection\n");

    ct->ctrl_connfd = accept(listenfd, NULL, NULL);
    if (ct->ctrl_connfd == -1) {
        ret = -errno;
        PP_PRINTERR("accept", ret);
        goto fail_close_socket;
    }

    close(listenfd);

    PP_DEBUG("SERVER: connected\n");

    return ret;

fail_close_socket:
    if (ct->ctrl_connfd != -1) {
        close(ct->ctrl_connfd);
        ct->ctrl_connfd = -1;
    }

    if (listenfd != -1)
        close(listenfd);

    return ret;
}

int pp_ctrl_init(struct ct_pingpong *ct)
{
    const uint32_t default_ctrl = 47592;
    struct timeval tv = {
        .tv_sec = 5
    };
    int ret;

    PP_DEBUG("Initializing control messages\n");

    if (ct->opts.dst_addr) {
        if (ct->opts.dst_port == 0)
            ct->opts.dst_port = default_ctrl;
        usleep(1000000);
        ret = pp_ctrl_init_client(ct);
    } else {
        if (ct->opts.src_port == 0)
            ct->opts.src_port = default_ctrl;
        ret = pp_ctrl_init_server(ct);
    }

    if (ret)
        return ret;

    ret = setsockopt(ct->ctrl_connfd, SOL_SOCKET, SO_RCVTIMEO, &tv,
             sizeof(struct timeval));
    if (ret == -1) {
        ret = -errno;
        PP_PRINTERR("setsockopt(SO_RCVTIMEO)", ret);
        return ret;
    }

    PP_DEBUG("Control messages initialized\n");

    return ret;
}

int pp_ctrl_send(struct ct_pingpong *ct, char *buf, size_t size)
{
    int ret, err;

    ret = send(ct->ctrl_connfd, buf, size, 0);
    if (ret < 0) {
        err = -errno;
        PP_PRINTERR("ctrl/send", err);
        return err;
    }
    if (ret == 0) {
        err = -ECONNABORTED;
        PP_ERR("ctrl/read: no data or remote connection closed");
        return err;
    }

    return ret;
}

int pp_ctrl_recv(struct ct_pingpong *ct, char *buf, size_t size)
{
    int ret, err;

    do {
        PP_DEBUG("receiving\n");
        ret = recv(ct->ctrl_connfd, buf, size, 0);
    } while (ret == -1 && errno == EAGAIN);
    if (ret < 0) {
        err = -errno;
        PP_PRINTERR("ctrl/read", err);
        return err;
    }
    if (ret == 0) {
        err = -ECONNABORTED;
        PP_ERR("ctrl/read: no data or remote connection closed");
        return err;
    }

    return ret;
}

int pp_send_name(struct ct_pingpong *ct, struct fid *endpoint)
{
    char local_name[64];
    size_t addrlen;
    uint32_t len;
    int ret;

    PP_DEBUG("Fetching local address\n");

    addrlen = sizeof(local_name);
    ret = fi_getname(endpoint, local_name, &addrlen);
    if (ret) {
        PP_PRINTERR("fi_getname", ret);
        return ret;
    }

    if (addrlen > sizeof(local_name)) {
        PP_DEBUG("Address exceeds control buffer length\n");
        return -EMSGSIZE;
    }

    PP_DEBUG("Sending name length\n");
    len = htonl(addrlen);
    ret = pp_ctrl_send(ct, (char *) &len, sizeof(len));
    if (ret < 0)
        return ret;

    PP_DEBUG("Sending name\n");
    ret = pp_ctrl_send(ct, local_name, addrlen);
    PP_DEBUG("Sent name\n");

    return ret;
}

int pp_recv_name(struct ct_pingpong *ct)
{
    uint32_t len;
    int ret;

    PP_DEBUG("Receiving name length\n");
    ret = pp_ctrl_recv(ct, (char *) &len, sizeof(len));
    if (ret < 0)
        return ret;

    len = ntohl(len);
    PP_DEBUG("Got a length of %d ", len);

    if (len > sizeof(ct->rem_name)) {
        PP_DEBUG("Address length exceeds address storage\n");
        return -EMSGSIZE;
    }

    PP_DEBUG("Receiving name\n");
    ret = pp_ctrl_recv(ct, ct->rem_name, len);
    if (ret < 0)
        return ret;
    PP_DEBUG("Received name %s \n", ct->rem_name);
    PP_DEBUG("Received name %d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d\n",
    (int) ((uint8_t*) &ct->rem_name)[0],
    (int) ((uint8_t*) &ct->rem_name)[1],
    (int) ((uint8_t*) &ct->rem_name)[2],
    (int) ((uint8_t*) &ct->rem_name)[3],
    (int) ((uint8_t*) &ct->rem_name)[4],
    (int) ((uint8_t*) &ct->rem_name)[5],
    (int) ((uint8_t*) &ct->rem_name)[6],
    (int) ((uint8_t*) &ct->rem_name)[7],
    (int) ((uint8_t*) &ct->rem_name)[8],
    (int) ((uint8_t*) &ct->rem_name)[9],
    (int) ((uint8_t*) &ct->rem_name)[10],
    (int) ((uint8_t*) &ct->rem_name)[11],
    (int) ((uint8_t*) &ct->rem_name)[12],
    (int) ((uint8_t*) &ct->rem_name)[13],
    (int) ((uint8_t*) &ct->rem_name)[14],
    (int) ((uint8_t*) &ct->rem_name)[15]);

    ct->hints->dest_addr = malloc(len);
    if (!ct->hints->dest_addr) {
        PP_DEBUG("Failed to allocate memory for destination address\n");
        return -ENOMEM;
    }

    /* fi_freeinfo will free the dest_addr field. */
    memcpy(ct->hints->dest_addr, ct->rem_name, len);
    ct->hints->dest_addrlen = len;

    return 0;
}

int pp_ctrl_finish(struct ct_pingpong *ct)
{
    if (ct->ctrl_connfd != -1) {
        close(ct->ctrl_connfd);
        ct->ctrl_connfd = -1;
    }

    return 0;
}

int pp_ctrl_sync(struct ct_pingpong *ct)
{
    int ret;

    PP_DEBUG("Syncing nodes\n");

    if (ct->opts.dst_addr) {
        snprintf(ct->ctrl_buf, sizeof(PP_MSG_SYNC_Q), "%s",
             PP_MSG_SYNC_Q);

        PP_DEBUG("CLIENT: syncing\n");
        ret = pp_ctrl_send(ct, ct->ctrl_buf, sizeof(PP_MSG_SYNC_Q));
        PP_DEBUG("CLIENT: after send / ret=%d\n", ret);
        if (ret < 0)
            return ret;
        if (ret < sizeof(PP_MSG_SYNC_Q)) {
            PP_ERR("CLIENT: bad length of sent data (len=%d/%zu)",
                   ret, sizeof(PP_MSG_SYNC_Q));
            return -EBADMSG;
        }
        PP_DEBUG("CLIENT: syncing now\n");

        ret = pp_ctrl_recv(ct, ct->ctrl_buf, sizeof(PP_MSG_SYNC_A));
        PP_DEBUG("CLIENT: after recv / ret=%d\n", ret);
        if (ret < 0)
            return ret;
        if (strcmp(ct->ctrl_buf, PP_MSG_SYNC_A)) {
            ct->ctrl_buf[PP_CTRL_BUF_LEN] = '\0';
            PP_DEBUG("CLIENT: sync error while acking A: <%s> "
                 "(len=%zu)\n",
                 ct->ctrl_buf, strlen(ct->ctrl_buf));
            return -EBADMSG;
        }
        PP_DEBUG("CLIENT: synced\n");
    } else {
        PP_DEBUG("SERVER: syncing\n");
        ret = pp_ctrl_recv(ct, ct->ctrl_buf, sizeof(PP_MSG_SYNC_Q));
        PP_DEBUG("SERVER: after recv / ret=%d\n", ret);
        if (ret < 0)
            return ret;
        if (strcmp(ct->ctrl_buf, PP_MSG_SYNC_Q)) {
            ct->ctrl_buf[PP_CTRL_BUF_LEN] = '\0';
            PP_DEBUG("SERVER: sync error while acking Q: <%s> "
                 "(len=%zu)\n",
                 ct->ctrl_buf, strlen(ct->ctrl_buf));
            return -EBADMSG;
        }

        PP_DEBUG("SERVER: syncing now\n");
        snprintf(ct->ctrl_buf, sizeof(PP_MSG_SYNC_A), "%s",
             PP_MSG_SYNC_A);

        ret = pp_ctrl_send(ct, ct->ctrl_buf, sizeof(PP_MSG_SYNC_A));
        PP_DEBUG("SERVER: after send / ret=%d\n", ret);
        if (ret < 0)
            return ret;
        if (ret < sizeof(PP_MSG_SYNC_A)) {
            PP_ERR("SERVER: bad length of sent data (len=%d/%zu)",
                   ret, sizeof(PP_MSG_SYNC_A));
            return -EBADMSG;
        }
        PP_DEBUG("SERVER: synced\n");
    }

    PP_DEBUG("Nodes synced\n");

    return 0;
}

int pp_ctrl_txrx_msg_count(struct ct_pingpong *ct)
{
    int ret;

    PP_DEBUG("Exchanging ack count\n");

    if (ct->opts.dst_addr) {
        memset(&ct->ctrl_buf, '\0', PP_MSG_LEN_CNT + 1);
        snprintf(ct->ctrl_buf, PP_MSG_LEN_CNT + 1, "%ld",
             ct->cnt_ack_msg);

        PP_DEBUG("CLIENT: sending count = <%s> (len=%zu)\n",
             ct->ctrl_buf, strlen(ct->ctrl_buf));
        ret = pp_ctrl_send(ct, ct->ctrl_buf, PP_MSG_LEN_CNT);
        if (ret < 0)
            return ret;
        if (ret < PP_MSG_LEN_CNT) {
            PP_ERR("CLIENT: bad length of sent data (len=%d/%d)",
                   ret, PP_MSG_LEN_CNT);
            return -EBADMSG;
        }
        PP_DEBUG("CLIENT: sent count\n");

        ret =
            pp_ctrl_recv(ct, ct->ctrl_buf, sizeof(PP_MSG_CHECK_CNT_OK));
        if (ret < 0)
            return ret;
        if (ret < sizeof(PP_MSG_CHECK_CNT_OK)) {
            PP_ERR(
                "CLIENT: bad length of received data (len=%d/%zu)",
                ret, sizeof(PP_MSG_CHECK_CNT_OK));
            return -EBADMSG;
        }

        if (strcmp(ct->ctrl_buf, PP_MSG_CHECK_CNT_OK)) {
            PP_DEBUG("CLIENT: error while server acking the count: "
                 "<%s> (len=%zu)\n",
                 ct->ctrl_buf, strlen(ct->ctrl_buf));
            return ret;
        }
        PP_DEBUG("CLIENT: count acked by server\n");
    } else {
        memset(&ct->ctrl_buf, '\0', PP_MSG_LEN_CNT + 1);

        PP_DEBUG("SERVER: receiving count\n");
        ret = pp_ctrl_recv(ct, ct->ctrl_buf, PP_MSG_LEN_CNT);
        if (ret < 0)
            return ret;
        if (ret < PP_MSG_LEN_CNT) {
            PP_ERR(
                "SERVER: bad length of received data (len=%d/%d)",
                ret, PP_MSG_LEN_CNT);
            return -EBADMSG;
        }
        ct->cnt_ack_msg = parse_ulong(ct->ctrl_buf, -1);
        if (ct->cnt_ack_msg < 0)
            return ret;
        PP_DEBUG("SERVER: received count = <%ld> (len=%zu)\n",
             ct->cnt_ack_msg, strlen(ct->ctrl_buf));

        snprintf(ct->ctrl_buf, sizeof(PP_MSG_CHECK_CNT_OK), "%s",
             PP_MSG_CHECK_CNT_OK);
        ret =
            pp_ctrl_send(ct, ct->ctrl_buf, sizeof(PP_MSG_CHECK_CNT_OK));
        if (ret < 0)
            return ret;
        if (ret < sizeof(PP_MSG_CHECK_CNT_OK)) {
            PP_ERR(
                "CLIENT: bad length of received data (len=%d/%zu)",
                ret, sizeof(PP_MSG_CHECK_CNT_OK));
            return -EBADMSG;
        }
        PP_DEBUG("SERVER: acked count to client\n");
    }

    PP_DEBUG("Ack count exchanged\n");

    return 0;
}

/*******************************************************************************
 *                                         Options
 ******************************************************************************/

static inline void pp_start(struct ct_pingpong *ct)
{
    PP_DEBUG("Starting test chrono\n");
    ct->opts.options |= PP_OPT_ACTIVE;
    ct->start = pp_gettime_us();
}

static inline void pp_stop(struct ct_pingpong *ct)
{
    ct->end = pp_gettime_us();
    ct->opts.options &= ~PP_OPT_ACTIVE;
    PP_DEBUG("Stopped test chrono\n");
}

static inline int pp_check_opts(struct ct_pingpong *ct, uint64_t flags)
{
    return (ct->opts.options & flags) == flags;
}

/*******************************************************************************
 *                                         Data Verification
 ******************************************************************************/

void pp_fill_buf(void *buf, int size)
{
    char *msg_buf;
    int msg_index;
    static unsigned int iter;
    int i;

    msg_index = ((iter++) * INTEG_SEED) % integ_alphabet_length;
    msg_buf = (char *)buf;
    for (i = 0; i < size; i++) {
        PP_DEBUG("index=%d msg_index=%d\n", i, msg_index);
        msg_buf[i] = integ_alphabet[msg_index++];
        if (msg_index >= integ_alphabet_length)
            msg_index = 0;
    }
}

int pp_check_buf(void *buf, int size)
{
    char *recv_data;
    char c;
    static unsigned int iter;
    int msg_index;
    int i;

    PP_DEBUG("Verifying buffer content\n");

    msg_index = ((iter++) * INTEG_SEED) % integ_alphabet_length;
    recv_data = (char *)buf;

    for (i = 0; i < size; i++) {
        c = integ_alphabet[msg_index++];
        if (msg_index >= integ_alphabet_length)
            msg_index = 0;
        if (c != recv_data[i]) {
            PP_DEBUG("index=%d msg_index=%d expected=%d got=%d\n",
                 i, msg_index, c, recv_data[i]);
            break;
        }
    }
    if (i != size) {
        PP_DEBUG("Finished veryfing buffer: content is corrupted\n");
        printf("Error at iteration=%d size=%d byte=%d\n", iter, size,
               i);
        return 1;
    }

    PP_DEBUG("Buffer verified\n");

    return 0;
}


/*******************************************************************************
 *                                         Error handling
 ******************************************************************************/

void eq_readerr(struct fid_eq *eq)
{
    struct fi_eq_err_entry eq_err;
    int rd;

    rd = fi_eq_readerr(eq, &eq_err, 0);
    if (rd != sizeof(eq_err)) {
        PP_PRINTERR("fi_eq_readerr", rd);
    } else {
        PP_ERR("eq_readerr: %s",
               fi_eq_strerror(eq, eq_err.prov_errno, eq_err.err_data,
                      NULL, 0));
    }
}

void pp_process_eq_err(ssize_t rd, struct fid_eq *eq, const char *fn)
{
    if (rd == -FI_EAVAIL)
        eq_readerr(eq);
    else
        PP_PRINTERR(fn, rd);
}

/*******************************************************************************
 *                                         Test sizes
 ******************************************************************************/

int generate_test_sizes(struct pp_opts *opts, size_t tx_size, int **sizes_)
{
    int defaults[6] = {64, 256, 1024, 4096, 65536, 1048576};
    int power_of_two;
    int half_up;
    int n = 0;
    int i;
    int *sizes = NULL;

    PP_DEBUG("Generating test sizes\n");

    sizes = calloc(64, sizeof(*sizes));
    if (sizes == NULL)
        return 0;
    *sizes_ = sizes;

    if (opts->options & PP_OPT_SIZE) {
        if (opts->transfer_size > tx_size)
            return 0;

        sizes[0] = opts->transfer_size;
        n = 1;
    } else if (opts->sizes_enabled != PP_ENABLE_ALL) {
        for (i = 0; i < (sizeof(defaults) / sizeof(defaults[0])); i++) {
            if (defaults[i] > tx_size)
                break;

            sizes[i] = defaults[i];
            n++;
        }
    } else {
        for (i = 0;; i++) {
            power_of_two = (i == 0) ? 0 : (1 << i);
            half_up =
                (i == 0) ? 1 : power_of_two + (power_of_two / 2);

            if (power_of_two > tx_size)
                break;

            sizes[i * 2] = power_of_two;
            n++;

            if (half_up > tx_size)
                break;

            sizes[(i * 2) + 1] = half_up;
            n++;
        }
    }

    PP_DEBUG("Generated %d test sizes\n", n);

    return n;
}

/*******************************************************************************
 *                                    Performance output
 ******************************************************************************/

/* str must be an allocated buffer of PP_STR_LEN bytes */
char *size_str(char *str, uint64_t size)
{
    uint64_t base, fraction = 0;
    char mag;

    memset(str, '\0', PP_STR_LEN);

    if (size >= (1 << 30)) {
        base = 1 << 30;
        mag = 'g';
    } else if (size >= (1 << 20)) {
        base = 1 << 20;
        mag = 'm';
    } else if (size >= (1 << 10)) {
        base = 1 << 10;
        mag = 'k';
    } else {
        base = 1;
        mag = '\0';
    }

    if (size / base < 10)
        fraction = (size % base) * 10 / base;

    if (fraction)
        snprintf(str, PP_STR_LEN, "%" PRIu64 ".%" PRIu64 "%c",
             size / base, fraction, mag);
    else
        snprintf(str, PP_STR_LEN, "%" PRIu64 "%c", size / base, mag);

    return str;
}

/* str must be an allocated buffer of PP_STR_LEN bytes */
char *cnt_str(char *str, size_t size, uint64_t cnt)
{
    if (cnt >= 1000000000)
        snprintf(str, size, "%" PRIu64 "b", cnt / 1000000000);
    else if (cnt >= 1000000)
        snprintf(str, size, "%" PRIu64 "m", cnt / 1000000);
    else if (cnt >= 1000)
        snprintf(str, size, "%" PRIu64 "k", cnt / 1000);
    else
        snprintf(str, size, "%" PRIu64, cnt);

    return str;
}

void show_perf(char *name, int tsize, int sent, int acked,
           uint64_t start, uint64_t end, int xfers_per_iter)
{
    static int header = 1;
    char str[PP_STR_LEN];
    int64_t elapsed = end - start;
    uint64_t bytes = (uint64_t)sent * tsize * xfers_per_iter;
    float usec_per_xfer;

    if (sent == 0)
        return;

    if (name) {
        if (header) {
            printf("%-50s%-8s%-8s%-9s%-8s%8s %10s%13s%13s\n",
                   "name", "bytes", "#sent", "#ack", "total",
                   "time", "MB/sec", "usec/xfer", "Mxfers/sec");
            header = 0;
        }

        printf("%-50s", name);
    } else {
        if (header) {
            printf("%-8s%-8s%-9s%-8s%8s %10s%13s%13s\n", "bytes",
                   "#sent", "#ack", "total", "time", "MB/sec",
                   "usec/xfer", "Mxfers/sec");
            header = 0;
        }
    }

    printf("%-8s", size_str(str, tsize));
    printf("%-8s", cnt_str(str, sizeof(str), sent));

    if (sent == acked)
        printf("=%-8s", cnt_str(str, sizeof(str), acked));
    else if (sent < acked)
        printf("-%-8s", cnt_str(str, sizeof(str), acked - sent));
    else
        printf("+%-8s", cnt_str(str, sizeof(str), sent - acked));

    printf("%-8s", size_str(str, bytes));

    usec_per_xfer = ((float)elapsed / sent / xfers_per_iter);
    printf("%8.2fs%10.2f%11.2f%11.2f\n", elapsed / 1000000.0,
           bytes / (1.0 * elapsed), usec_per_xfer, 1.0 / usec_per_xfer);
}

/*******************************************************************************
 *                                      Data Messaging
 ******************************************************************************/

int pp_cq_readerr(struct fid_cq *cq)
{
    struct fi_cq_err_entry cq_err;
    int ret;

    ret = fi_cq_readerr(cq, &cq_err, 0);
    if (ret < 0) {
        PP_PRINTERR("fi_cq_readerr", ret);
    } else {
        PP_ERR("cq_readerr: %s",
               fi_cq_strerror(cq, cq_err.prov_errno, cq_err.err_data,
                      NULL, 0));
        ret = -cq_err.err;
    }
    return ret;
}

static int pp_get_cq_comp(struct fid_cq *cq, uint64_t *cur, uint64_t total,
              int timeout_sec)
{
    struct fi_cq_err_entry comp;
    uint64_t a = 0, b = 0;
    int ret = 0;

    if (timeout_sec >= 0)
        a = pp_gettime_us();

    while (total - *cur > 0) {
        ret = fi_cq_read(cq, &comp, 1);
        if (ret > 0) {
            if (timeout_sec >= 0)
                a = pp_gettime_us();

            (*cur)++;
        } else if (ret < 0 && ret != -FI_EAGAIN) {
            if (ret == -FI_EAVAIL) {
                ret = pp_cq_readerr(cq);
                (*cur)++;
            } else {
                PP_PRINTERR("pp_get_cq_comp", ret);
            }

            return ret;
        } else if (timeout_sec >= 0) {
            b = pp_gettime_us();
            if ((b - a) / 1000000 > timeout_sec) {
                fprintf(stderr, "%ds timeout expired\n",
                    timeout_sec);
                return -FI_ENODATA;
            }
        }
    }

    return 0;
}

int pp_get_rx_comp(struct ct_pingpong *ct, uint64_t total)
{
    int ret = FI_SUCCESS;

    if (ct->rxcq) {
        ret = pp_get_cq_comp(ct->rxcq, &(ct->rx_cq_cntr), total,
                     ct->timeout_sec);
    } else {
        PP_ERR(
            "Trying to get a RX completion when no RX CQ was opened");
        ret = -FI_EOTHER;
    }
    return ret;
}

int pp_get_tx_comp(struct ct_pingpong *ct, uint64_t total)
{
    int ret;

    if (ct->txcq) {
        ret = pp_get_cq_comp(ct->txcq, &(ct->tx_cq_cntr), total, -1);
    } else {
        PP_ERR(
            "Trying to get a TX completion when no TX CQ was opened");
        ret = -FI_EOTHER;
    }
    return ret;
}

#define PP_POST(post_fn, comp_fn, seq, op_str, ...)                            \
    do {                                                                   \
        int timeout_sec_save;                                          \
        int ret, rc;                                                   \
                                           \
        while (1) {                                                    \
            ret = post_fn(__VA_ARGS__);                            \
            if (!ret)                                              \
                break;                                         \
                                           \
            if (ret != -FI_EAGAIN) {                               \
                PP_PRINTERR(op_str, ret);                      \
                return ret;                                    \
            }                                                      \
                                           \
            timeout_sec_save = ct->timeout_sec;                    \
            ct->timeout_sec = 0;                                   \
            rc = comp_fn(ct, seq);                                 \
            ct->timeout_sec = timeout_sec_save;                    \
            if (rc && rc != -FI_EAGAIN) {                          \
                PP_ERR("Failed to get " op_str " completion"); \
                return rc;                                     \
            }                                                      \
        }                                                              \
        seq++;                                                         \
    } while (0)

ssize_t pp_post_tx(struct ct_pingpong *ct, struct fid_ep *ep, size_t size,
           struct fi_context *ctx)
{
    PP_POST(fi_send, pp_get_tx_comp, ct->tx_seq, "transmit", ep, ct->tx_buf,
        size, fi_mr_desc(ct->mr), ct->remote_fi_addr, ctx);
    return 0;
}

ssize_t pp_tx(struct ct_pingpong *ct, struct fid_ep *ep, size_t size)
{
    ssize_t ret;

    if (pp_check_opts(ct, PP_OPT_VERIFY_DATA | PP_OPT_ACTIVE))
        pp_fill_buf((char *)ct->tx_buf, size);

    ret = pp_post_tx(ct, ep, size, &(ct->tx_ctx));
    if (ret)
        return ret;

    ret = pp_get_tx_comp(ct, ct->tx_seq);

    return ret;
}

ssize_t pp_post_inject(struct ct_pingpong *ct, struct fid_ep *ep, size_t size)
{
    PP_POST(fi_inject, pp_get_tx_comp, ct->tx_seq, "inject", ep, ct->tx_buf,
        size, ct->remote_fi_addr);
    ct->tx_cq_cntr++;
    return 0;
}

ssize_t pp_inject(struct ct_pingpong *ct, struct fid_ep *ep, size_t size)
{
    ssize_t ret;

    if (pp_check_opts(ct, PP_OPT_VERIFY_DATA | PP_OPT_ACTIVE))
        pp_fill_buf((char *)ct->tx_buf, size);

    ret = pp_post_inject(ct, ep, size);
    if (ret)
        return ret;

    return ret;
}

ssize_t pp_post_rx(struct ct_pingpong *ct, struct fid_ep *ep, size_t size,
           struct fi_context *ctx)
{
    PP_POST(fi_recv, pp_get_rx_comp, ct->rx_seq, "receive", ep, ct->rx_buf,
        MAX(size, PP_MAX_CTRL_MSG), fi_mr_desc(ct->mr), 0, ctx);
    return 0;
}

ssize_t pp_rx(struct ct_pingpong *ct, struct fid_ep *ep, size_t size)
{
    ssize_t ret;

    ret = pp_get_rx_comp(ct, ct->rx_seq);
    if (ret)
        return ret;

    if (pp_check_opts(ct, PP_OPT_VERIFY_DATA | PP_OPT_ACTIVE)) {
        ret = pp_check_buf((char *)ct->rx_buf, size);
        if (ret)
            return ret;
    }
    /* TODO: verify CQ data, if available */

    /* Ignore the size arg. Post a buffer large enough to handle all message
     * sizes. pp_sync() makes use of pp_rx() and gets called in tests just
     * before message size is updated. The recvs posted are always for the
     * next incoming message.
     */
    ret = pp_post_rx(ct, ct->ep, ct->rx_size, &(ct->rx_ctx));
    if (!ret)
        ct->cnt_ack_msg++;

    return ret;
}

/*******************************************************************************
 *                                Initialization and allocations
 ******************************************************************************/

void init_test(struct ct_pingpong *ct, struct pp_opts *opts)
{
    char sstr[PP_STR_LEN];

    size_str(sstr, opts->transfer_size);
    if (!(opts->options & PP_OPT_ITER))
        opts->iterations = size_to_count(opts->transfer_size);

    ct->cnt_ack_msg = 0;
}

uint64_t pp_init_cq_data(struct fi_info *info)
{
    if (info->domain_attr->cq_data_size >= sizeof(uint64_t)) {
        return 0x0123456789abcdefULL;
    } else {
        return 0x0123456789abcdefULL &
               ((0x1ULL << (info->domain_attr->cq_data_size * 8)) - 1);
    }
}

int pp_alloc_msgs(struct ct_pingpong *ct)
{
    int ret;
    long alignment = 1;

    ct->tx_size = ct->opts.options & PP_OPT_SIZE ? ct->opts.transfer_size
                             : PP_MAX_DATA_MSG;
    if (ct->tx_size > ct->fi->ep_attr->max_msg_size)
        ct->tx_size = ct->fi->ep_attr->max_msg_size;
    ct->rx_size = ct->tx_size;
    ct->buf_size = MAX(ct->tx_size, PP_MAX_CTRL_MSG) +
               MAX(ct->rx_size, PP_MAX_CTRL_MSG);

    alignment = sysconf(_SC_PAGESIZE);
    if (alignment < 0) {
        ret = -errno;
        PP_PRINTERR("sysconf", ret);
        return ret;
    }
    /* Extra alignment for the second part of the buffer */
    ct->buf_size += alignment;

    ret = posix_memalign(&(ct->buf), (size_t)alignment, ct->buf_size);
    if (ret) {
        PP_PRINTERR("posix_memalign", ret);
        return ret;
    }
    memset(ct->buf, 0, ct->buf_size);
    ct->rx_buf = ct->buf;
    ct->tx_buf = (char *)ct->buf + MAX(ct->rx_size, PP_MAX_CTRL_MSG);
    ct->tx_buf = (void *)(((uintptr_t)ct->tx_buf + alignment - 1) &
                  ~(alignment - 1));

    ct->remote_cq_data = pp_init_cq_data(ct->fi);

    if (ct->fi->mode & FI_LOCAL_MR) {
        ret = fi_mr_reg(ct->domain, ct->buf, ct->buf_size,
                FI_SEND | FI_RECV, 0, PP_MR_KEY, 0, &(ct->mr),
                NULL);
        if (ret) {
            PP_PRINTERR("fi_mr_reg", ret);
            return ret;
        }
    } else {
        ct->mr = &(ct->no_mr);
    }

    return 0;
}

int pp_open_fabric_res(struct ct_pingpong *ct)
{
    int ret;

    PP_DEBUG("Opening fabric resources: fabric, eq & domain\n");

    ret = fi_fabric(ct->fi->fabric_attr, &(ct->fabric), NULL);
    if (ret) {
        PP_PRINTERR("fi_fabric", ret);
        return ret;
    }

    ret = fi_eq_open(ct->fabric, &(ct->eq_attr), &(ct->eq), NULL);
    if (ret) {
        PP_PRINTERR("fi_eq_open", ret);
        return ret;
    }

    PP_DEBUG("Creating domain");
    ret = fi_domain(ct->fabric, ct->fi, &(ct->domain), NULL);
    if (ret) {
        PP_PRINTERR("fi_domain", ret);
        return ret;
    }

    PP_DEBUG("Fabric resources opened\n");

    return 0;
}

int pp_alloc_active_res(struct ct_pingpong *ct, struct fi_info *fi)
{
    int ret;

    ret = pp_alloc_msgs(ct);
    if (ret)
        return ret;

    if (ct->cq_attr.format == FI_CQ_FORMAT_UNSPEC)
        ct->cq_attr.format = FI_CQ_FORMAT_CONTEXT;

    ct->cq_attr.wait_obj = FI_WAIT_NONE;

    ct->cq_attr.size = fi->tx_attr->size;
    ret = fi_cq_open(ct->domain, &(ct->cq_attr), &(ct->txcq), &(ct->txcq));
    if (ret) {
        PP_PRINTERR("fi_cq_open", ret);
        return ret;
    }

    ct->cq_attr.size = fi->rx_attr->size;
    ret = fi_cq_open(ct->domain, &(ct->cq_attr), &(ct->rxcq), &(ct->rxcq));
    if (ret) {
        PP_PRINTERR("fi_cq_open", ret);
        return ret;
    }

    if (fi->ep_attr->type == FI_EP_RDM ||
        fi->ep_attr->type == FI_EP_DGRAM) {
        if (fi->domain_attr->av_type != FI_AV_UNSPEC)
            ct->av_attr.type = fi->domain_attr->av_type;

        ret = fi_av_open(ct->domain, &(ct->av_attr), &(ct->av), NULL);
        if (ret) {
            PP_PRINTERR("fi_av_open", ret);
            return ret;
        }
    }

    ret = fi_endpoint(ct->domain, fi, &(ct->ep), NULL);
    if (ret) {
        PP_PRINTERR("fi_endpoint", ret);
        return ret;
    }

    return 0;
}

int pp_getinfo(struct ct_pingpong *ct, struct fi_info *hints,
           struct fi_info **info)
{
    uint64_t flags = 0;
    int ret;

    if (!hints->ep_attr->type)
        hints->ep_attr->type = FI_EP_DGRAM;
    PP_DEBUG("Calling getinfo\n");
    ret = fi_getinfo(PP_FIVERSION, NULL, NULL, flags, hints, info);
    if (ret) {
        PP_PRINTERR("fi_getinfo", ret);
        return ret;
    }
    return 0;
}

#define PP_EP_BIND(ep, fd, flags)                                              \
    do {                                                                   \
        int ret;                                                       \
        if ((fd)) {                                                    \
            ret = fi_ep_bind((ep), &(fd)->fid, (flags));           \
            if (ret) {                                             \
                PP_PRINTERR("fi_ep_bind", ret);                \
                return ret;                                    \
            }                                                      \
        }                                                              \
    } while (0)

int pp_init_ep(struct ct_pingpong *ct)
{
    int ret;

    PP_DEBUG("Initializing endpoint\n");

    if (ct->fi->ep_attr->type == FI_EP_MSG)
        PP_EP_BIND(ct->ep, ct->eq, 0);
    PP_EP_BIND(ct->ep, ct->av, 0);
    PP_EP_BIND(ct->ep, ct->txcq, FI_TRANSMIT);
    PP_EP_BIND(ct->ep, ct->rxcq, FI_RECV);

    ret = fi_enable(ct->ep);
    if (ret) {
        PP_PRINTERR("fi_enable", ret);
        return ret;
    }

    ret = pp_post_rx(ct, ct->ep, MAX(ct->rx_size, PP_MAX_CTRL_MSG),
             &(ct->rx_ctx));
    if (ret)
        return ret;

    PP_DEBUG("Endpoint initialized\n");

    return 0;
}

int pp_av_insert(struct fid_av *av, void *addr, size_t count,
         fi_addr_t *fi_addr, uint64_t flags, void *context)
{
    int ret;

    PP_DEBUG("Connection-less endpoint: inserting new address in vector\n");

    ret = fi_av_insert(av, addr, count, fi_addr, flags, context);
    if (ret < 0) {
        PP_PRINTERR("fi_av_insert", ret);
        return ret;
    } else if (ret != count) {
        PP_ERR("fi_av_insert: number of addresses inserted = %d;"
               " number of addresses given = %zd\n",
               ret, count);
        return -EXIT_FAILURE;
    }

    PP_DEBUG("Connection-less endpoint: new address inserted in vector\n");

    return 0;
}

int pp_exchange_names_connected(struct ct_pingpong *ct)
{
    int ret;

    PP_DEBUG("Connection-based endpoint: setting up connection\n");

    ret = pp_ctrl_sync(ct);
    if (ret)
        return ret;

    if (ct->opts.dst_addr) {
        ret = pp_recv_name(ct);
        if (ret < 0)
            return ret;

        ret = pp_getinfo(ct, ct->hints, &(ct->fi));
        if (ret)
            return ret;
    } else {
        ret = pp_send_name(ct, &ct->pep->fid);
        if (ret < 0)
            return ret;
    }

    return 0;
}

int pp_start_server(struct ct_pingpong *ct)
{
    int ret;

    PP_DEBUG("Connected endpoint: starting server\n");

    ret = pp_getinfo(ct, ct->hints, &(ct->fi_pep));
    if (ret)
        return ret;

    ret = fi_fabric(ct->fi_pep->fabric_attr, &(ct->fabric), NULL);
    if (ret) {
        PP_PRINTERR("fi_fabric", ret);
        return ret;
    }

    ret = fi_eq_open(ct->fabric, &(ct->eq_attr), &(ct->eq), NULL);
    if (ret) {
        PP_PRINTERR("fi_eq_open", ret);
        return ret;
    }

    ret = fi_passive_ep(ct->fabric, ct->fi_pep, &(ct->pep), NULL);
    if (ret) {
        PP_PRINTERR("fi_passive_ep", ret);
        return ret;
    }

    ret = fi_pep_bind(ct->pep, &(ct->eq->fid), 0);
    if (ret) {
        PP_PRINTERR("fi_pep_bind", ret);
        return ret;
    }

    ret = fi_listen(ct->pep);
    if (ret) {
        PP_PRINTERR("fi_listen", ret);
        return ret;
    }

    PP_DEBUG("Connected endpoint: server started\n");

    return 0;
}

int pp_server_connect(struct ct_pingpong *ct)
{
    struct fi_eq_cm_entry entry;
    uint32_t event;
    ssize_t rd;
    int ret;

    PP_DEBUG("Connected endpoint: connecting server\n");

    ret = pp_exchange_names_connected(ct);
    if (ret)
        goto err;

    ret = pp_ctrl_sync(ct);
    if (ret)
        goto err;

    /* Listen */
    rd = fi_eq_sread(ct->eq, &event, &entry, sizeof(entry), -1, 0);
    if (rd != sizeof(entry)) {
        pp_process_eq_err(rd, ct->eq, "fi_eq_sread");
        return (int)rd;
    }

    ct->fi = entry.info;
    if (event != FI_CONNREQ) {
        fprintf(stderr, "Unexpected CM event %d\n", event);
        ret = -FI_EOTHER;
        goto err;
    }

    PP_DEBUG("Creating domain\n");
    ret = fi_domain(ct->fabric, ct->fi, &(ct->domain), NULL);
    if (ret) {
        PP_PRINTERR("fi_domain", ret);
        goto err;
    }

    ret = pp_alloc_active_res(ct, ct->fi);
    if (ret)
        goto err;

    ret = pp_init_ep(ct);
    if (ret)
        goto err;

    PP_DEBUG("accepting\n");

    ret = fi_accept(ct->ep, NULL, 0);
    if (ret) {
        PP_PRINTERR("fi_accept", ret);
        goto err;
    }

    ret = pp_ctrl_sync(ct);
    if (ret)
        goto err;

    /* Accept */
    rd = fi_eq_sread(ct->eq, &event, &entry, sizeof(entry), -1, 0);
    if (rd != sizeof(entry)) {
        pp_process_eq_err(rd, ct->eq, "fi_eq_sread");
        ret = (int)rd;
        goto err;
    }

    if (event != FI_CONNECTED || entry.fid != &(ct->ep->fid)) {
        fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
            event, entry.fid, ct->ep);
        ret = -FI_EOTHER;
        goto err;
    }

    PP_DEBUG("Connected endpoint: server connected\n");

    return 0;
err:
    fi_reject(ct->pep, ct->fi->handle, NULL, 0);
    return ret;
}

int pp_client_connect(struct ct_pingpong *ct)
{
    struct fi_eq_cm_entry entry;
    uint32_t event;
    ssize_t rd;
    int ret;

    ret = pp_exchange_names_connected(ct);
    if (ret)
        return ret;

    /* Check that the remote is still up */
    ret = pp_ctrl_sync(ct);
    if (ret)
        return ret;

    ret = pp_open_fabric_res(ct);
    if (ret)
        return ret;

    ret = pp_alloc_active_res(ct, ct->fi);
    if (ret)
        return ret;

    ret = pp_init_ep(ct);
    if (ret)
        return ret;

    PP_DEBUG("Calling fi_connect \n");
    PP_DEBUG("using name %d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d\n",
    (int) ((uint8_t*) &ct->rem_name)[0],
    (int) ((uint8_t*) &ct->rem_name)[1],
    (int) ((uint8_t*) &ct->rem_name)[2],
    (int) ((uint8_t*) &ct->rem_name)[3],
    (int) ((uint8_t*) &ct->rem_name)[4],
    (int) ((uint8_t*) &ct->rem_name)[5],
    (int) ((uint8_t*) &ct->rem_name)[6],
    (int) ((uint8_t*) &ct->rem_name)[7],
    (int) ((uint8_t*) &ct->rem_name)[8],
    (int) ((uint8_t*) &ct->rem_name)[9],
    (int) ((uint8_t*) &ct->rem_name)[10],
    (int) ((uint8_t*) &ct->rem_name)[11],
    (int) ((uint8_t*) &ct->rem_name)[12],
    (int) ((uint8_t*) &ct->rem_name)[13],
    (int) ((uint8_t*) &ct->rem_name)[14],
    (int) ((uint8_t*) &ct->rem_name)[15]);

    ret = fi_connect(ct->ep, ct->rem_name, NULL, 0);
    if (ret) {
        PP_PRINTERR("fi_connect", ret);
        return ret;
    }

    ret = pp_ctrl_sync(ct);
    if (ret)
        return ret;

    /* Connect */
    rd = fi_eq_sread(ct->eq, &event, &entry, sizeof(entry), -1, 0);
    if (rd != sizeof(entry)) {
        pp_process_eq_err(rd, ct->eq, "fi_eq_sread");
        ret = (int)rd;
        return ret;
    }

    if (event != FI_CONNECTED || entry.fid != &(ct->ep->fid)) {
        fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
            event, entry.fid, ct->ep);
        ret = -FI_EOTHER;
        return ret;
    }

    return 0;
}

int pp_init_fabric(struct ct_pingpong *ct)
{
    int ret;

    ret = pp_ctrl_init(ct);
    if (ret)
        return ret;

    PP_DEBUG("Initializing fabric\n");

    PP_DEBUG("Connection-less endpoint: initializing address vector\n");

    if (ct->opts.dst_addr) {
        ret = pp_recv_name(ct);
        if (ret < 0)
            return ret;

        ret = pp_getinfo(ct, ct->hints, &(ct->fi));
        if (ret)
            return ret;

        ret = pp_open_fabric_res(ct);
        if (ret)
            return ret;

        ret = pp_alloc_active_res(ct, ct->fi);
        if (ret)
            return ret;

        ret = pp_init_ep(ct);
        if (ret)
            return ret;

        ret = pp_send_name(ct, &ct->ep->fid);
    } else {
        PP_DEBUG("SERVER: getinfo\n");
        ret = pp_getinfo(ct, ct->hints, &(ct->fi));
        if (ret)
            return ret;

        PP_DEBUG("SERVER: open fabric resources\n");
        ret = pp_open_fabric_res(ct);
        if (ret)
            return ret;

        PP_DEBUG("SERVER: allocate active resource\n");
        ret = pp_alloc_active_res(ct, ct->fi);
        if (ret)
            return ret;

        PP_DEBUG("SERVER: initialize endpoint\n");
        ret = pp_init_ep(ct);
        if (ret)
            return ret;

        ret = pp_send_name(ct, &ct->ep->fid);
        if (ret < 0)
            return ret;

        ret = pp_recv_name(ct);
    }

    if (ret < 0)
        return ret;

    ret = pp_av_insert(ct->av, ct->rem_name, 1, &(ct->remote_fi_addr), 0,
               NULL);
    if (ret)
        return ret;
    PP_DEBUG("Connection-less endpoint: address vector initialized\n");

    PP_DEBUG("Fabric Initialized\n");

    return 0;
}

/*******************************************************************************
 *                                Deallocations and Final
 ******************************************************************************/

void pp_free_res(struct ct_pingpong *ct)
{
    PP_DEBUG("Freeing resources of test suite\n");

    if (ct->mr != &(ct->no_mr))
        PP_CLOSE_FID(ct->mr);
    PP_CLOSE_FID(ct->ep);
    PP_CLOSE_FID(ct->pep);
    PP_CLOSE_FID(ct->rxcq);
    PP_CLOSE_FID(ct->txcq);
    PP_CLOSE_FID(ct->av);
    PP_CLOSE_FID(ct->eq);
    PP_CLOSE_FID(ct->domain);
    PP_CLOSE_FID(ct->fabric);

    if (ct->buf) {
        free(ct->buf);
        ct->buf = ct->rx_buf = ct->tx_buf = NULL;
        ct->buf_size = ct->rx_size = ct->tx_size = 0;
    }
    if (ct->fi_pep) {
        fi_freeinfo(ct->fi_pep);
        ct->fi_pep = NULL;
    }
    if (ct->fi) {
        fi_freeinfo(ct->fi);
        ct->fi = NULL;
    }
    if (ct->hints) {
        fi_freeinfo(ct->hints);
        ct->hints = NULL;
    }

    PP_DEBUG("Resources of test suite freed\n");
}

int pp_finalize(struct ct_pingpong *ct)
{
    struct iovec iov;
    int ret;
    struct fi_context ctx;
    struct fi_msg msg;

    PP_DEBUG("Terminating test\n");

    strcpy(ct->tx_buf, "fin");
    iov.iov_base = ct->tx_buf;
    iov.iov_len = 4;

    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.iov_count = 1;
    msg.addr = ct->remote_fi_addr;
    msg.context = &ctx;

    ret = fi_sendmsg(ct->ep, &msg, FI_INJECT | FI_TRANSMIT_COMPLETE);
    if (ret) {
        PP_PRINTERR("transmit", ret);
        return ret;
    }

    ret = pp_get_tx_comp(ct, ++ct->tx_seq);
    if (ret)
        return ret;

    ret = pp_get_rx_comp(ct, ct->rx_seq);
    if (ret)
        return ret;

    ret = pp_ctrl_finish(ct);
    if (ret)
        return ret;

    PP_DEBUG("Test terminated\n");

    return 0;
}

/*******************************************************************************
 *                                CLI: Usage and Options parsing
 ******************************************************************************/

void pp_pingpong_usage(char *name, char *desc)
{
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
    fprintf(stderr, "  %s [OPTIONS] <srv_addr>\tconnect to server\n", name);

    if (desc)
        fprintf(stderr, "\n%s\n", desc);

    fprintf(stderr, "\nOptions:\n");

    fprintf(stderr, " %-20s %s\n", "-B <src_port>",
        "source control port number (server: 47592, client: auto)");
    fprintf(stderr, " %-20s %s\n", "-P <dst_port>",
        "destination control port number (client: 47592)");

    fprintf(stderr, " %-20s %s\n", "-d <domain>", "domain name");
    fprintf(stderr, " %-20s %s\n", "-p <provider>",
        "specific provider name eg sockets, verbs");
    fprintf(stderr, " %-20s %s\n", "-e <ep_type>",
        "endpoint type: msg|rdm|dgram (dgram)");

    fprintf(stderr, " %-20s %s\n", "-I <number>",
        "number of iterations (1000)");
    fprintf(stderr, " %-20s %s\n", "-S <size>",
        "specific transfer size or 'all' (all)");

    fprintf(stderr, " %-20s %s\n", "-c", "enables data_integrity checks");

    fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
    fprintf(stderr, " %-20s %s\n", "-v", "enable debugging output");
}

void pp_parse_opts(struct ct_pingpong *ct, int op, char *optarg)
{
    switch (op) {

    /* Domain */
    case 'd':
        ct->hints->domain_attr->name = strdup(optarg);
        break;

    /* Provider */
    case 'p':
        /* The provider name will be checked during the fabric
         * initialization.
         */
        ct->hints->fabric_attr->prov_name = strdup(optarg);
        break;

    /* Endpoint */
    case 'e':
        if (!strncasecmp("msg", optarg, 3) && (strlen(optarg) == 3)) {
            ct->hints->ep_attr->type = FI_EP_MSG;
        } else if (!strncasecmp("rdm", optarg, 3) &&
               (strlen(optarg) == 3)) {
            ct->hints->ep_attr->type = FI_EP_RDM;
        } else if (!strncasecmp("dgram", optarg, 5) &&
               (strlen(optarg) == 5)) {
            ct->hints->ep_attr->type = FI_EP_DGRAM;
        } else {
            fprintf(stderr, "Unknown endpoint : %s\n", optarg);
            exit(EXIT_FAILURE);
        }
        break;

    /* Iterations */
    case 'I':
        ct->opts.options |= PP_OPT_ITER;
        ct->opts.iterations = (int)parse_ulong(optarg, INT_MAX);
        if (ct->opts.iterations < 0)
            ct->opts.iterations = 0;
        break;

    /* Message Size */
    case 'S':
        if (!strncasecmp("all", optarg, 3) && (strlen(optarg) == 3)) {
            ct->opts.sizes_enabled = PP_ENABLE_ALL;
        } else {
            ct->opts.options |= PP_OPT_SIZE;
            ct->opts.transfer_size =
                (int)parse_ulong(optarg, INT_MAX);
        }
        break;

    /* Check data */
    case 'c':
        ct->opts.options |= PP_OPT_VERIFY_DATA;
        break;

    /* Source Port */
    case 'B':
        ct->opts.src_port = parse_ulong(optarg, UINT16_MAX);
        break;

    /* Destination Port */
    case 'P':
        ct->opts.dst_port = parse_ulong(optarg, UINT16_MAX);
        break;

    /* Debug */
    case 'v':
        pp_debug = 1;
        break;
    default:
        /* let getopt handle unknown opts*/
        break;
    }
}

/*******************************************************************************
 *      PingPong core and implemenations for endpoints
 ******************************************************************************/

int pingpong(struct ct_pingpong *ct)
{
    int ret, i;

    ret = pp_ctrl_sync(ct);
    if (ret)
        return ret;

    pp_start(ct);
    if (ct->opts.dst_addr) {
        for (i = 0; i < ct->opts.iterations; i++) {

            if (ct->opts.transfer_size <
                ct->fi->tx_attr->inject_size)
                ret = pp_inject(ct, ct->ep,
                        ct->opts.transfer_size);
            else
                ret = pp_tx(ct, ct->ep, ct->opts.transfer_size);
            if (ret)
                return ret;

            ret = pp_rx(ct, ct->ep, ct->opts.transfer_size);
            if (ret)
                return ret;
        }
    } else {
        for (i = 0; i < ct->opts.iterations; i++) {

            ret = pp_rx(ct, ct->ep, ct->opts.transfer_size);
            if (ret)
                return ret;

            if (ct->opts.transfer_size <
                ct->fi->tx_attr->inject_size)
                ret = pp_inject(ct, ct->ep,
                        ct->opts.transfer_size);
            else
                ret = pp_tx(ct, ct->ep, ct->opts.transfer_size);
            if (ret)
                return ret;
        }
    }
    pp_stop(ct);

    ret = pp_ctrl_txrx_msg_count(ct);
    if (ret)
        return ret;

    PP_DEBUG("Results:\n");
    show_perf(NULL, ct->opts.transfer_size, ct->opts.iterations,
          ct->cnt_ack_msg, ct->start, ct->end, 2);

    return 0;
}

int run_suite_pingpong(struct ct_pingpong *ct)
{
    int i, sizes_cnt;
    int ret = 0;
    int *sizes = NULL;

    pp_banner_fabric_info(ct);

    sizes_cnt = generate_test_sizes(&ct->opts, ct->tx_size, &sizes);

    PP_DEBUG("Count of sizes to test: %d\n", sizes_cnt);

    for (i = 0; i < sizes_cnt; i++) {
        ct->opts.transfer_size = sizes[i];
        init_test(ct, &(ct->opts));
        ret = pingpong(ct);
        if (ret)
            goto out;
    }

out:
    free(sizes);
    return ret;
}

static int run_pingpong_dgram(struct ct_pingpong *ct)
{
    int ret;

    PP_DEBUG("Selected endpoint: DGRAM\n");

    ret = pp_init_fabric(ct);
    if (ret)
        return ret;

    /* Post an extra receive to avoid lacking a posted receive in the
     * finalize.
     */
    ret = fi_recv(ct->ep, ct->rx_buf, ct->rx_size, fi_mr_desc(ct->mr), 0,
              &ct->rx_ctx);

    ret = run_suite_pingpong(ct);
    if (ret)
        return ret;

    return pp_finalize(ct);
}

static int run_pingpong_rdm(struct ct_pingpong *ct)
{
    int ret;

    PP_DEBUG("Selected endpoint: RDM\n");

    ret = pp_init_fabric(ct);
    if (ret)
        return ret;

    ret = run_suite_pingpong(ct);
    if (ret)
        return ret;

    return pp_finalize(ct);
}

static int run_pingpong_msg(struct ct_pingpong *ct)
{
    int ret;

    PP_DEBUG("Selected endpoint: MSG\n");

    ret = pp_ctrl_init(ct);
    if (ret)
        return ret;

    if (!ct->opts.dst_addr) {
        ret = pp_start_server(ct);
        if (ret)
            return ret;
    }

    if (ct->opts.dst_addr) {
        ret = pp_client_connect(ct);
        PP_DEBUG("CLIENT: client_connect=%s\n", ret ? "KO" : "OK");
    } else {
        ret = pp_server_connect(ct);
        PP_DEBUG("SERVER: server_connect=%s\n", ret ? "KO" : "OK");
    }

    if (ret)
        return ret;
printf("\nRunning MSG pinpong\n\n");
    ret = run_suite_pingpong(ct);
    if (ret)
        goto out;

    ret = pp_finalize(ct);
out:
    fi_shutdown(ct->ep, 0);
    return ret;
}

int main(int argc, char **argv)
{
    int ret, op;

    ret = EXIT_SUCCESS;

    struct ct_pingpong ct = {
        .timeout_sec = -1,
        .ctrl_connfd = -1,
        .opts = {
            .iterations = 1000,
            .transfer_size = 1024,
            .sizes_enabled = PP_DEFAULT_SIZE
        },
        .eq_attr.wait_obj = FI_WAIT_UNSPEC,
    };

    ct.hints = fi_allocinfo();
    if (!ct.hints)
        return EXIT_FAILURE;
    ct.hints->ep_attr->type = FI_EP_DGRAM;
    ct.hints->caps = FI_MSG;
    ct.hints->mode = FI_CONTEXT | FI_LOCAL_MR;

    while ((op = getopt(argc, argv, "hvd:p:e:I:S:B:P:c")) != -1) {
        switch (op) {
        default:
            pp_parse_opts(&ct, op, optarg);
            break;
        case '?':
        case 'h':
            pp_pingpong_usage(argv[0],
                      "Ping pong client and server");
            return EXIT_FAILURE;
        }
    }

    if (optind < argc)
        ct.opts.dst_addr = argv[optind];

    pp_banner_options(&ct);

    switch (ct.hints->ep_attr->type) {
    case FI_EP_DGRAM:
        if (ct.opts.options & PP_OPT_SIZE)
            ct.hints->ep_attr->max_msg_size = ct.opts.transfer_size;
        ret = run_pingpong_dgram(&ct);
        break;
    case FI_EP_RDM:
        ret = run_pingpong_rdm(&ct);
        break;
    case FI_EP_MSG:
        ret = run_pingpong_msg(&ct);
        break;
    default:
        fprintf(stderr, "Endpoint unsupported: %d\n",
            ct.hints->ep_attr->type);
        ret = EXIT_FAILURE;
    }

    pp_free_res(&ct);
    return -ret;
}
