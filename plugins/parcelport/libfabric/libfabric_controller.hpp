//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_CONTROLLER_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_CONTROLLER_HPP

// config
#include <hpx/config/defines.hpp>
//
#include <hpx/lcos/local/shared_mutex.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/lcos/future.hpp>
//
#include <plugins/parcelport/parcelport_logging.hpp>
#include <plugins/parcelport/libfabric/rdma_locks.hpp>
#include <plugins/parcelport/libfabric/libfabric_endpoint.hpp>
//
#include <plugins/parcelport/libfabric/unordered_map.hpp>
//
#include <memory>
#include <deque>
#include <chrono>
#include <iostream>
#include <functional>
#include <map>
#include <atomic>
#include <string>
#include <utility>
#include <cstdint>
//
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include "fabric_error.hpp"

#ifndef PP_FIVERSION
#define PP_FIVERSION FI_VERSION(1, 4)
#endif

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

enum {
    PP_OPT_ACTIVE = 1 << 0,
    PP_OPT_ITER = 1 << 1,
    PP_OPT_SIZE = 1 << 2,
    PP_OPT_VERIFY_DATA = 1 << 3,
};

struct fabric_interface {
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

    struct fi_av_attr av_attr;
    struct fi_eq_attr eq_attr;
    struct fi_cq_attr cq_attr;

    ~fabric_interface() {
        if (fi_pep) {
            fi_freeinfo(fi_pep);
            fi_pep = NULL;
        }
        if (fi) {
            fi_freeinfo(fi);
            fi = NULL;
        }
        if (hints) {
            fi_freeinfo(hints);
            hints = NULL;
        }
    }
};

int pp_getinfo(struct fabric_interface *ct, struct fi_info *hints,
           struct fi_info **info)
{
    uint64_t flags = 0;
    int ret;

    if (!hints->ep_attr->type)
        hints->ep_attr->type = FI_EP_DGRAM;

    ret = fi_getinfo(PP_FIVERSION, NULL, NULL, flags, hints, info);
    if (ret) {
        PP_PRINTERR("fi_getinfo", ret);
        return ret;
    }
    return 0;
}

int pp_start_server(struct fabric_interface *ct)
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

int size_to_count(int size)
{
    if (size >= (1 << 20))
        return 100;
    else if (size >= (1 << 16))
        return 1000;
    else
        return 10000;
}

void pp_banner_options(struct fabric_interface *ct)
{
    if (ct->hints->fabric_attr->prov_name)
        PP_DEBUG("  - %-20s: %s\n", "provider",
              ct->hints->fabric_attr->prov_name);
    if (ct->hints->domain_attr->name)
        PP_DEBUG("  - %-20s: %s\n", "domain",
              ct->hints->domain_attr->name);
}

void pp_banner_fabric_info(struct fabric_interface *ct)
{
    PP_DEBUG(
        "Running pingpong test with the %s endpoint through a %s provider\n",
        fi_tostr(&ct->fi->ep_attr->type, FI_TYPE_EP_TYPE),
        ct->fi->fabric_attr->prov_name);
    PP_DEBUG(" * Fabric Attributes:\n");
    PP_DEBUG("  - %-20s: %s\n", "name", ct->fi->fabric_attr->name);
    PP_DEBUG("  - %-20s: %s\n", "prov_name",
         ct->fi->fabric_attr->prov_name);
    PP_DEBUG("  - %-20s: %u\n", "prov_version",
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
    PP_DEBUG("  - %-20s: %u\n", "protocol",
         ct->fi->ep_attr->protocol);
    PP_DEBUG("  - %-20s: %u\n", "protocol_version",
         ct->fi->ep_attr->protocol_version);
    PP_DEBUG("  - %-20s: %zu\n", "max_msg_size",
         ct->fi->ep_attr->max_msg_size);
    PP_DEBUG("  - %-20s: %zu\n", "max_order_raw_size",
         ct->fi->ep_attr->max_order_raw_size);
}


// @TODO : Remove the client map pair as we have a copy in the libfabric_parcelport class
// that does almost the same job.
// @TODO : Most of this code could be moved into the main parcelport, or the endpoint
// classes
namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{


    class libfabric_controller
    {
    public:
        typedef hpx::lcos::local::spinlock mutex_type;
        typedef hpx::parcelset::policies::libfabric::unique_lock<mutex_type> unique_lock;
        typedef hpx::parcelset::policies::libfabric::scoped_lock<mutex_type> scoped_lock;

        // constructor gets infor from device and sets up all necessary
        // maps, queues and server endpoint etc
        libfabric_controller(std::string provider, std::string domain,
        	std::string endpoint, int port)
        {
            LOG_DEBUG_MSG("Entering controller constructor");
            //
            struct fabric_interface ct = {};
            ct.hints = fi_allocinfo();
            if (!ct.hints) {
                throw fabric_error(0, "Failed to init fabric hints");
            }
            ct.hints->caps = FI_MSG | FI_RMA;
            ct.hints->mode = ~0;
            ct.hints->fabric_attr->prov_name = strdup("verbs");
            ct.hints->domain_attr->name = strdup("mlx5_0");

            std::string endpoint_type = endpoint;
            if (endpoint_type=="msg") {
                ct.hints->ep_attr->type = FI_EP_MSG;
            } else if (endpoint_type=="rdm") {
                ct.hints->ep_attr->type = FI_EP_RDM;
            } else if (endpoint_type=="dgram") {
                ct.hints->ep_attr->type = FI_EP_DGRAM;
            }

//            ct.opts.dst_addr = "greina6";

            pp_banner_options(&ct);
            pp_start_server(&ct);

            LOG_DEBUG_MSG("Fetching local address");
            struct fid *endp = &ct.pep->fid;

            char local_name[64];
            size_t addrlen;
            uint32_t len;
            int ret;

            addrlen = sizeof(local_name);
            ret = fi_getname(endp, local_name, &addrlen);
            if (ret) {
                fabric_error(ret, "fi_getname");
            }

            if (addrlen > sizeof(local_name)) {
                fabric_error(0, "Address exceeds control buffer length");
            }

            LOG_DEBUG_MSG("Name length " << decnumber(addrlen));
            LOG_DEBUG_MSG("address info is "
                    << ipaddress(local_name[0]) << " "
                    << ipaddress(local_name[4]) << " "
                    << ipaddress(local_name[8]) << " "
                    << ipaddress(local_name[12]) << " ");

            pp_banner_options(&ct);

            // Create a fabric domain object.
            protection_domain_ = std::make_shared<libfabric_domain>(ct.fabric, ct.fi_pep);
            //LOG_DEVEL_MSG("created protection domain " << hexpointer(protection_domain_));
            if (ct.fi->mode & FI_LOCAL_MR) {

            }

            // Create a memory pool for pinned buffers
            memory_pool_ = std::make_shared<rdma_memory_pool> (protection_domain_);


        }

        // clean up all resources
        ~libfabric_controller() {};

        // initiate a listener for connections
        int startup() { return 0; }

        // returns true when all connections have been disconnected and none are active
        bool isTerminated() {
            return (qp_endpoint_map_.size() == 0);
        }

        // types we need for connection and disconnection callback functions
        // into the main parcelport code.
        typedef std::function<void(libfabric_endpoint_ptr)>       ConnectionFunction;
        typedef std::function<int(libfabric_endpoint_ptr client)> DisconnectionFunction;

        // Set a callback which will be called immediately after
        // RDMA_CM_EVENT_ESTABLISHED has been received.
        // This should be used to initialize all structures for handling a new connection
        void setConnectionFunction(ConnectionFunction f) {
            this->connection_function_ = f;
        }

        // currently not used.
        void setDisconnectionFunction(DisconnectionFunction f) {
            this->disconnection_function_ = f;
        }

        // This is the main polling function that checks for work completions
        // and connection manager events, if stopped is true, then completions
        // are thrown away, otherwise the completion callback is triggered
        int poll_endpoints(bool stopped=false) { return 0; }

        int poll_for_work_completions(bool stopped=false) { return 0; }
/*
        inline libfabric_completion_queue *get_completion_queue() const {
            return completion_queue_.get();
        }

        inline libfabric_endpoint *get_client_from_completion(struct fi_cq_msg_entry &completion)
        {
            return qp_endpoint_map_.at(completion.qp_num).get();
        }
*/
        inline libfabric_domain_ptr get_protection_domain() {
            return this->protection_domain_;
        }

        libfabric_endpoint* get_endpoint(uint32_t remote_ip) {
            return (std::get<0>(connections_started_.find(remote_ip)->second)).get();

        }

        inline rdma_memory_pool_ptr get_memory_pool() {
            return memory_pool_;
        }

        typedef std::function<int(struct fi_cq_msg_entry completion, libfabric_endpoint *client)>
            CompletionFunction;

        void setCompletionFunction(CompletionFunction f) {
            this->completion_function_ = f;
        }

        void refill_client_receives(bool force=true) {}

        hpx::shared_future<libfabric_endpoint_ptr> connect_to_server(uint32_t remote_ip) {
            return hpx::make_ready_future(
                std::make_shared<libfabric_endpoint>()
            );
        }

        void disconnect_from_server(libfabric_endpoint_ptr client) {}

        void disconnect_all() {}

        bool active() { return false; }

    private:
        void debug_connections() {}

        int handle_event(struct rdma_cm_event *cm_event, libfabric_endpoint *client) { return 0; }

        int handle_connect_request(
            struct rdma_cm_event *cm_event, std::uint32_t remote_ip) { return 0; }

        int start_server_connection(uint32_t remote_ip) { return 0; }

        // store info about local device
        std::string           device_;
        std::string           interface_;
        sockaddr_in           local_addr_;

        // callback functions used for connection event handling
        ConnectionFunction    connection_function_;
        DisconnectionFunction disconnection_function_;

        // callback function for handling a completion event
        CompletionFunction    completion_function_;

        // Protection domain for all resources.
        libfabric_domain_ptr protection_domain_;
        // Pinned memory pool used for allocating buffers
        rdma_memory_pool_ptr        memory_pool_;
        // Server/Listener for RDMA connections.
        libfabric_endpoint_ptr          server_endpoint_;
        // Shared completion queue for all endoints
//        libfabric_completion_queue_ptr  completion_queue_;
        // Count outstanding receives posted to SRQ + Completion queue
        std::atomic<uint16_t>       preposted_receives_;

        // only allow one thread to handle connect/disconnect events etc
        mutex_type            controller_mutex_;

        typedef std::tuple<
            libfabric_endpoint_ptr,
            hpx::promise<libfabric_endpoint_ptr>,
            hpx::shared_future<libfabric_endpoint_ptr>
        > promise_tuple_type;
        //
        typedef std::pair<const uint32_t, promise_tuple_type> ClientMapPair;
        typedef std::pair<const uint32_t, libfabric_endpoint_ptr> QPMapPair;

        // Map of all active clients indexed by queue pair number.
//        hpx::concurrent::unordered_map<uint32_t, promise_tuple_type> clients_;

        typedef hpx::concurrent::unordered_map<uint32_t, promise_tuple_type>
            ::map_read_lock_type map_read_lock_type;
        typedef hpx::concurrent::unordered_map<uint32_t, promise_tuple_type>
            ::map_write_lock_type map_write_lock_type;

        // Map of connections started, needed during address resolution until
        // qp is created, and then to hold a future to an endpoint that the parcelport
        // can get and wait on
        hpx::concurrent::unordered_map<uint32_t, promise_tuple_type> connections_started_;

        typedef std::pair<uint32_t, libfabric_endpoint_ptr> qp_map_type;
        hpx::concurrent::unordered_map<uint32_t, libfabric_endpoint_ptr> qp_endpoint_map_;

        // used to skip polling event channel too frequently
        typedef std::chrono::time_point<std::chrono::system_clock> time_type;
        time_type event_check_time_;
        uint32_t  event_pause_;

    };

    // Smart pointer for libfabric_controller object.
    typedef std::shared_ptr<libfabric_controller> libfabric_controller_ptr;

}}}}

#endif
