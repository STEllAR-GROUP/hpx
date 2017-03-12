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
#include <hpx/runtime/parcelset/parcelport_impl.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
//
#include <plugins/parcelport/parcelport_logging.hpp>
#include <plugins/parcelport/libfabric/rdma_locks.hpp>
#include <plugins/parcelport/libfabric/locality.hpp>
//
#include <plugins/parcelport/unordered_map.hpp>
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
#include <rdma/fi_rma.h>
#include "fabric_error.hpp"

#define HPX_PARCELPORT_LIBFABRIC_MAX_PREPOSTS 16

#ifdef HPX_PARCELPORT_LIBFABRIC_GNI
# define HPX_PARCELPORT_LIBFABRIC_ENDPOINT_RDM
#else
# define HPX_PARCELPORT_LIBFABRIC_ENDPOINT_MSG
#endif

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

        // when a new connection is requested, it will be completed asynchronously
        // we need a promise/future for each endpoint so that we can set the new
        // endpoint when the connection completes and is ready
        // Note - only used during connection, then deleted
        typedef std::tuple<
            hpx::promise<fid_ep *>,
            hpx::shared_future<fid_ep *>
        > promise_tuple_type;

        // used in map of ipaddress->promise_tuple_type
        typedef std::pair<const uint32_t, promise_tuple_type> ClientMapPair;

        // lock types for maps
        typedef hpx::concurrent::unordered_map<uint32_t, promise_tuple_type>
            ::map_read_lock_type map_read_lock_type;
        typedef hpx::concurrent::unordered_map<uint32_t, promise_tuple_type>
            ::map_write_lock_type map_write_lock_type;

        // Map of connections started, needed until connection is completed
        hpx::concurrent::unordered_map<uint32_t, promise_tuple_type> endpoint_tmp_;
        std::unordered_map<uint32_t, fi_addr_t> endpoint_av_;

        locality here_;

        struct fi_info    *fabric_info_;
        struct fid_fabric *fabric_;
        struct fid_domain *fabric_domain_;
        // Server/Listener for RDMA connections.
        struct fid_pep    *ep_passive_;
        struct fid_ep     *ep_active_;
        struct fid_ep     *ep_shared_rx_cxt_;


        // we will use just one event queue for all connections
        struct fid_eq     *event_queue_;
        struct fid_cq     *txcq, *rxcq;
        struct fid_av     *av;

        struct fi_context tx_ctx, rx_ctx;
        uint64_t remote_cq_data;

        uint64_t tx_seq, rx_seq, tx_cq_cntr, rx_cq_cntr;

        struct fi_av_attr av_attr;
        struct fi_eq_attr eq_attr;
        struct fi_cq_attr cq_attr;

        bool immediate_;

        // --------------------------------------------------------------------
        // constructor gets info from device and sets up all necessary
        // maps, queues and server endpoint etc
        libfabric_controller(
            std::string provider,
            std::string domain,
            std::string endpoint, int port=7910)
        {
            FUNC_START_DEBUG_MSG;
            fabric_info_      = nullptr;
            fabric_           = nullptr;
            ep_passive_       = nullptr;
            ep_active_        = nullptr;
            event_queue_      = nullptr;
            fabric_domain_    = nullptr;
            ep_shared_rx_cxt_ = nullptr;
            //
            txcq  = nullptr;
            rxcq  = nullptr;
            av    = nullptr;
            //
            tx_ctx = {};
            rx_ctx = {};
            remote_cq_data = 0;
            tx_seq = 0;
            rx_seq = 0;
            tx_cq_cntr = 0;
            rx_cq_cntr = 0;

            av_attr = {};
            eq_attr = {};
            cq_attr = {};
            //
            immediate_          = false;
            preposted_receives_ = 0;
            //
            open_fabric(provider, domain, endpoint);

            // Create a memory pool for pinned buffers
            memory_pool_ = std::make_shared<rdma_memory_pool> (fabric_domain_);

            // setup a passive listener, or an active RDM endpoint
            here_ = create_local_endpoint();
            create_event_queue();

            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        // clean up all resources
        ~libfabric_controller()
        {
            fi_close(&fabric_->fid);
            fi_close(&ep_passive_->fid);
            fi_close(&ep_active_->fid);
            fi_close(&event_queue_->fid);
            fi_close(&fabric_domain_->fid);
            fi_close(&ep_shared_rx_cxt_->fid);
            // clean up
            fi_freeinfo(fabric_info_);
        }

        // --------------------------------------------------------------------
        // initialize the basic fabric/domain/name
        void open_fabric(
            std::string provider, std::string domain, std::string endpoint_type)
        {
            FUNC_START_DEBUG_MSG;
            struct fi_info *fabric_hints_ = fi_allocinfo();
            if (!fabric_hints_) {
                throw fabric_error(-1, "Failed to allocate fabric hints");
            }
            // we require message and RMA support, so ask for them
            // we also want receives to carry source address info
            fabric_hints_->caps                   = FI_MSG | FI_RMA | FI_SOURCE | FI_WRITE | FI_READ | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_RMA_EVENT;
            fabric_hints_->mode                   = ~0; // FI_CONTEXT | FI_LOCAL_MR;
            fabric_hints_->fabric_attr->prov_name = strdup(provider.c_str());
            LOG_DEBUG_MSG("fabric provider " << fabric_hints_->fabric_attr->prov_name);
            if (domain.size()>0) {
                fabric_hints_->domain_attr->name  = strdup(domain.c_str());
                LOG_DEBUG_MSG("fabric domain "   << fabric_hints_->domain_attr->name);
            }

            // use infiniband type basic registration for now
            fabric_hints_->domain_attr->mr_mode = FI_MR_BASIC;

#ifdef HPX_PARCELPORT_LIBFABRIC_ENDPOINT_RDM
            LOG_DEVEL_MSG("Selecting endpoint type RDM");
            fabric_hints_->ep_attr->type = FI_EP_RDM;
#else
            // we will use a shared receive context for active endpoints
            fabric_hints_->ep_attr->rx_ctx_cnt = FI_SHARED_CONTEXT;

            if (endpoint_type=="msg") {
                fabric_hints_->ep_attr->type = FI_EP_MSG;
            } else if (endpoint_type=="rdm") {
                fabric_hints_->ep_attr->type = FI_EP_RDM;
            } else if (endpoint_type=="dgram") {
                fabric_hints_->ep_attr->type = FI_EP_DGRAM;
            }
            else {
                LOG_DEBUG_MSG("endpoint type not set, using RDM");
                fabric_hints_->ep_attr->type = FI_EP_RDM;
            }
#endif

            // by default, we will always want completions on both tx/rx events
            fabric_hints_->tx_attr->op_flags = FI_COMPLETION;
            fabric_hints_->rx_attr->op_flags = FI_COMPLETION;

            uint64_t flags = 0;
            LOG_DEBUG_MSG("Getting initial info about fabric");
            int ret = fi_getinfo(FI_VERSION(1,4), NULL, NULL,
                flags, fabric_hints_, &fabric_info_);
            if (ret) {
                throw fabric_error(ret, "Failed to get fabric info");
            }
            LOG_DEVEL_MSG("Fabric info " << fi_tostr(fabric_info_, FI_TYPE_INFO));

            immediate_ = (fabric_info_->rx_attr->mode & FI_RX_CQ_DATA)!=0;
            LOG_DEVEL_MSG("Fabric supports immediate data " << immediate_);
            bool context = (fabric_hints_->mode & FI_CONTEXT)!=0;
            LOG_DEVEL_MSG("Fabric requires FI_CONTEXT " << context);

            LOG_DEBUG_MSG("Creating fabric object");
            ret = fi_fabric(fabric_info_->fabric_attr, &fabric_, NULL);
            if (ret) {
                throw fabric_error(ret, "Failed to get fi_fabric");
            }

            // Allocate a domain.
            LOG_DEBUG_MSG("Allocating domain ");
            ret = fi_domain(fabric_, fabric_info_, &fabric_domain_, NULL);
            if (ret) throw fabric_error(ret, "fi_domain");

            fi_freeinfo(fabric_hints_);
            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        // create endpoint and get ready for possible communications
        void startup()
        {
            FUNC_START_DEBUG_MSG;
            //
#ifdef HPX_PARCELPORT_LIBFABRIC_ENDPOINT_RDM
            bind_endpoint_to_queues(ep_active_);
#else
            bind_endpoint_to_queues(ep_passive_);
#endif
            refill_client_receives(HPX_PARCELPORT_LIBFABRIC_MAX_PREPOSTS, true);
        }

        // --------------------------------------------------------------------
        void create_event_queue()
        {
            LOG_DEVEL_MSG("Creating event queue");
            eq_attr.wait_obj = FI_WAIT_UNSPEC;
            int ret = fi_eq_open(fabric_, &eq_attr, &event_queue_, NULL);
            if (ret) throw fabric_error(ret, "fi_eq_open");

            if (fabric_info_->ep_attr->type == FI_EP_MSG) {
                LOG_DEBUG_MSG("Binding event queue to passive endpoint");
                ret = fi_pep_bind(ep_passive_, &event_queue_->fid, 0);
                if (ret) throw fabric_error(ret, "fi_pep_bind");

                LOG_DEBUG_MSG("Passive endpoint : listen");
                ret = fi_listen(ep_passive_);
                if (ret) throw fabric_error(ret, "fi_listen");

                LOG_DEBUG_MSG("Allocating shared receive context");
                ret = fi_srx_context(fabric_domain_, fabric_info_->rx_attr,
                    &ep_shared_rx_cxt_, NULL);
                if (ret) throw fabric_error(ret, "fi_srx_context");
            }
            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        locality create_local_endpoint()
        {
            struct fid *id;
#ifdef HPX_PARCELPORT_LIBFABRIC_ENDPOINT_RDM
            LOG_DEBUG_MSG("Creating active endpoint");
            new_endpoint_active(fabric_info_, &ep_active_);
            LOG_DEVEL_MSG("active endpoint " << hexpointer(ep_active_));
            id = &ep_active_->fid;
#else
            LOG_DEBUG_MSG("Creating passive endpoint");
            ret = fi_passive_ep(fabric_, fabric_info_, &ep_passive_, NULL);
            if (ret) {
                throw fabric_error(ret, "Failed to create fi_passive_ep");
            }
            LOG_DEVEL_MSG("passive endpoint " << hexpointer(ep_passive_));
            id = &ep_passive_->fid;
#endif
            locality::locality_data local_addr;
            std::size_t addrlen = locality::array_size;
            LOG_DEVEL_MSG("Fetching local address using size " << decnumber(addrlen));
            int ret = fi_getname(id, local_addr.data(), &addrlen);
            if (ret || (addrlen>locality::array_size)) {
                fabric_error(ret, "fi_getname - size error or other problem");
            }

            //LOG_EXCLUSIVE(
                std::stringstream temp1, temp2;
                for (std::size_t i=0; i<locality::array_length; ++i) {
                    temp1 << ipaddress(local_addr[i]);
                }
                LOG_DEVEL_MSG("address info is " << temp1.str().c_str());
                for (std::size_t i=0; i<locality::array_length; ++i) {
                    temp2 << hexuint32(local_addr[i]);
                }
                LOG_DEVEL_MSG("address info is " << temp2.str().c_str());
            //);
            FUNC_END_DEBUG_MSG;
            return locality(local_addr);
        }

        // --------------------------------------------------------------------
        void new_endpoint_active(struct fi_info *info, struct fid_ep **new_endpoint)
        {
            FUNC_START_DEBUG_MSG;
            // create an 'active' endpoint that can be used for sending/receiving
            LOG_DEBUG_MSG("Creating active endpoint");
            int ret = fi_endpoint(fabric_domain_, info, new_endpoint, NULL);
            if (ret) throw fabric_error(ret, "fi_endpoint");

            if (info->ep_attr->type == FI_EP_MSG) {
                if (event_queue_) {
                    LOG_DEBUG_MSG("Binding endpoint to EQ");
                    ret = fi_ep_bind(*new_endpoint, &event_queue_->fid, 0);
                    if (ret) throw fabric_error(ret, "bind event_queue_");
                }
            }
        }

        // --------------------------------------------------------------------
        void bind_endpoint_to_queues(struct fid_ep *endpoint)
        {
            int ret;
            if (av) {
                LOG_DEBUG_MSG("Binding endpoint to AV");
                ret = fi_ep_bind(endpoint, &av->fid, 0);
                if (ret) throw fabric_error(ret, "bind event_queue_");
            }

            if (txcq) {
                LOG_DEBUG_MSG("Binding endpoint to TX CQ");
                ret = fi_ep_bind(endpoint, &txcq->fid, FI_TRANSMIT);
                if (ret) throw fabric_error(ret, "bind txcq");
            }

            if (rxcq) {
                LOG_DEBUG_MSG("Binding endpoint to RX CQ");
                ret = fi_ep_bind(endpoint, &rxcq->fid, FI_RECV);
                if (ret) throw fabric_error(ret, "rxcq");
            }

            if (ep_shared_rx_cxt_) {
                LOG_DEBUG_MSG("Binding endpoint to shared receive context");
                ret = fi_ep_bind(endpoint, &ep_shared_rx_cxt_->fid, 0);
                if (ret) throw fabric_error(ret, "ep_shared_rx_cxt_");
            }

            LOG_DEVEL_MSG("Enabling endpoint " << hexpointer(endpoint));
            ret = fi_enable(endpoint);
            if (ret) throw fabric_error(ret, "fi_enable");

            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        void initialize_localities(hpx::agas::addressing_service &as)
        {
            FUNC_START_DEBUG_MSG;
            std::uint32_t N = hpx::get_config().get_num_localities();
            LOG_DEVEL_MSG("Parcelport initialize_localities with " << N << " localities");

            // make sure address vector is created
            create_completion_queues(fabric_info_, N);

            for (std::uint32_t i=0; i<N; ++i) {
                hpx::naming::gid_type l = hpx::naming::get_gid_from_locality_id(i);
                LOG_DEVEL_MSG("Resolving locality" << l);
                // each locality may be reachable by mutiplte parcelports
                const parcelset::endpoints_type &res = as.resolve_locality(l);
                // get the fabric related data
                auto it = res.find("libfabric");
                LOG_DEVEL_MSG("locality resolution " << it->first << " => " << it->second);
                const hpx::parcelset::locality &fabric_locality = it->second;
                const locality &loc = fabric_locality.get<locality>();
                // put the provide specific data into the address vector
                // so that we can look it  up later
                fi_addr_t dummy = insert_address(loc);
            }
            LOG_DEVEL_MSG("Done getting localities ");
            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        fi_addr_t get_fabric_address(const locality &dest_fabric)
        {
            return endpoint_av_.find(dest_fabric.ip_address())->second;
        }

        // --------------------------------------------------------------------
        const locality & here() const { return here_; }

        // --------------------------------------------------------------------
        const bool & immedate_data_supported() const { return immediate_; }

        // --------------------------------------------------------------------
        // returns true when all connections have been disconnected and none are active
        bool isTerminated() {
            return false;
            //return (qp_endpoint_map_.size() == 0);
        }

        // types we need for connection and disconnection callback functions
        // into the main parcelport code.
        typedef std::function<void(fid_ep *endpoint, uint32_t ipaddr)> ConnectionFunction;
        typedef std::function<void(fid_ep *endpoint, uint32_t ipaddr)> DisconnectionFunction;

        // --------------------------------------------------------------------
        // Set a callback which will be called immediately after
        // RDMA_CM_EVENT_ESTABLISHED has been received.
        // This should be used to initialize all structures for handling a new connection
        void setConnectionFunction(ConnectionFunction f) {
            connection_function_ = f;
        }

        // --------------------------------------------------------------------
        // currently not used.
        void setDisconnectionFunction(DisconnectionFunction f) {
            disconnection_function_ = f;
        }

        // --------------------------------------------------------------------
        // This is the main polling function that checks for work completions
        // and connection manager events, if stopped is true, then completions
        // are thrown away, otherwise the completion callback is triggered
        int poll_endpoints(bool stopped=false)
        {
            poll_for_work_completions(stopped);
            poll_event_queue(stopped);

            return 0;
        }

        // --------------------------------------------------------------------
        int poll_for_work_completions(bool stopped=false)
        {
            // @TODO, disable polling until queues are initialized to avoid this check
            // if queues are not setup, don't poll
            if (HPX_UNLIKELY(!rxcq)) return 0;

            LOG_TIMED_INIT(poll);
            LOG_TIMED_BLOCK(poll, DEVEL, 5.0,
                {
                    LOG_DEVEL_MSG("Polling work completion channel");
                }
            )
/*
struct fi_cq_msg_entry {
    void     *op_context; // operation context
    uint64_t flags;       // completion flags
    size_t   len;         // size of received data
};
*/
            //std::array<char, 256> buffer;
            fi_cq_msg_entry entry;
            int ret = fi_cq_read(txcq, &entry, 1);
            if (ret>0) {
                //struct fi_cq_msg_entry *entry = (struct fi_cq_msg_entry *)(buffer.data());
                LOG_DEBUG_MSG("Completion txcq wr_id "
                    << fi_tostr(&entry.flags, FI_TYPE_OP_FLAGS) << " "
                    << hexpointer(entry.op_context) << "length " << hexuint32(entry.len));
                if (entry.flags & FI_RMA) {
                    LOG_DEVEL_MSG("@@@@@ Received a txcq RMA completion %%%%% ");
                    rdma_completion_function_(entry.op_context, ep_active_, entry.len);
                }
                else if (entry.flags == (FI_MSG | FI_SEND)) {
                    send_completion_function_(entry.op_context, ep_active_, entry.len);
                }
                else {
                    LOG_DEVEL_MSG("$$$$$ Received an unknown txcq completion ***** "
                        << decnumber(entry.flags));
                    std::terminate();
                }
                return 1;
            }
            else if (ret==0 || ret==-EAGAIN) {
                // do nothing, we will try again on the next check
                LOG_TIMED_MSG(poll, DEVEL, 5, "txcq EAGAIN");
            }
            else if (ret<0) {
                struct fi_cq_err_entry e = {};
                int err_sz = 0;
                err_sz = fi_cq_readerr(txcq, &e ,0);
                LOG_ERROR_MSG("txcq Error with flags " << e.flags << " len " << e.len);
                throw fabric_error(ret, "completion txcq read");
            }

            // receives will use fi_cq_readfrom as we want the source address
            fi_addr_t src_addr;
            ret = fi_cq_readfrom(rxcq, &entry, 1, &src_addr);
            if (ret>0) {
                if (--preposted_receives_<HPX_PARCELPORT_LIBFABRIC_MAX_PREPOSTS/4) {
                    refill_client_receives(HPX_PARCELPORT_LIBFABRIC_MAX_PREPOSTS, false);
                }
                LOG_DEBUG_MSG("Completion rxcq wr_id "
                    << fi_tostr(&entry.flags, FI_TYPE_OP_FLAGS)
                    << " source " << hexpointer(src_addr)
                    << "context " << hexpointer(entry.op_context));
//                void *client = reinterpret_cast<libfabric_memory_region*>
//                    (entry.op_context)->get_user_data();
                if ((entry.flags & FI_RMA) == FI_RMA) {
                    LOG_DEVEL_MSG("Received an rxcq RMA completion");
                    rdma_completion_function_(entry.op_context,
                        ep_active_, src_addr);
                }
                else if (entry.flags == (FI_MSG | FI_RECV)) {
//                    LOG_DEVEL_MSG("FI_ADDR_T FI_RECV " << hexpointer(src_addr));
                    recv_completion_function_(entry.op_context,
                        ep_active_, entry.len, src_addr);
                }
                else {
                    LOG_DEVEL_MSG("Received an unknown rxcq completion "
                        << decnumber(entry.flags));
                }
                return 1;
            }
            else if (ret==0 || ret==-EAGAIN) {
                // do nothing, we will try again on the next check
                LOG_TIMED_MSG(poll, DEVEL, 5, "rxcq EAGAIN");
            }
            else if (ret<0) {
                struct fi_cq_err_entry e = {};
                int err_sz = 0;
                err_sz = fi_cq_readerr(rxcq, &e ,0);
                LOG_ERROR_MSG("rxcq Error with flags " << e.flags << " len " << e.len);
                throw fabric_error(ret, "completion rxcq read");
            }
            return 0;
        }

        // --------------------------------------------------------------------
        int poll_event_queue(bool stopped=false)
        {
            LOG_TIMED_INIT(poll);
            LOG_TIMED_BLOCK(poll, DEVEL, 5.0,
                {
                    LOG_DEVEL_MSG("Polling event completion channel");
                }
            )
            struct fi_eq_cm_entry *cm_entry;
            struct fi_eq_entry    *entry;
            struct fid_ep         *new_ep;
            uint32_t *addr;
            uint32_t event;
            std::array<char, 256> buffer;
            ssize_t rd = fi_eq_read(event_queue_, &event, buffer.data(), sizeof(buffer), 0);
            if (rd > 0) {
                LOG_DEBUG_MSG("fi_eq_cm_entry " << decnumber(sizeof(fi_eq_cm_entry)) << " fi_eq_entry " << decnumber(sizeof(fi_eq_entry)));
                LOG_DEBUG_MSG("got event " << event << " with bytes = " << decnumber(rd));
                switch (event) {
                case FI_CONNREQ:
                {
                    cm_entry = reinterpret_cast<struct fi_eq_cm_entry*>(buffer.data());
                    locality::locality_data addressinfo;
                    std::memcpy(addressinfo.data(), cm_entry->info->dest_addr, locality::array_size);
                    locality loc(addressinfo);
                    LOG_DEVEL_MSG("FI_CONNREQ                 from "
                        << ipaddress(loc.ip_address()) << "-> "
                        << ipaddress(here_.ip_address())
                        << "( " << ipaddress(here_.ip_address()) << " )");
                    {
                        auto result = insert_new_future(loc.ip_address());
                        // if the insert fails, it means we have a connection
                        // already in progress, reject if we are a lower ip address
                        if (!result.first && loc.ip_address()>here_.ip_address()) {
                            LOG_DEVEL_MSG("FI_CONNREQ priority fi_reject   "
                                << ipaddress(loc.ip_address()) << "-> "
                                << ipaddress(here_.ip_address())
                                << "( " << ipaddress(here_.ip_address()) << " )");
//                            int ret = fi_reject(ep_passive_, cm_entry->info->handle,  nullptr, 0);
//                            if (ret) throw fabric_error(ret, "new_ep fi_reject failed");
                            fi_freeinfo(cm_entry->info);
                            return 0;
                        }
                        // create a new endpoint for this request and accept it
                        new_endpoint_active(cm_entry->info, &new_ep);
                        LOG_DEVEL_MSG("Calling fi_accept               "
                            << ipaddress(loc.ip_address()) << "-> "
                            << ipaddress(here_.ip_address())
                            << "( " << ipaddress(here_.ip_address()) << " )");
                        int ret = fi_accept(new_ep, &here_.ip_address(), sizeof(uint32_t));
                        if (ret) throw fabric_error(ret, "new_ep fi_accept failed");
                    }
                    fi_freeinfo(cm_entry->info);
                    break;
                }
                case FI_CONNECTED:
                {
                    cm_entry = reinterpret_cast<struct fi_eq_cm_entry*>(buffer.data());
                    new_ep = container_of(cm_entry->fid, struct fid_ep, fid);
                    locality::locality_data address;
                    std::size_t len = sizeof(locality::locality_data);
                    fi_getpeer(new_ep, address.data(), &len);
                    //
                    auto present1 = endpoint_tmp_.is_in_map(address[1]);
                    if (!present1.second) {
                        throw fabric_error(0, "FI_CONNECTED, endpoint map error");
                    }
                    LOG_DEVEL_MSG("FI_CONNECTED " << hexpointer(new_ep)
                        << ipaddress(locality::ip_address(address)) << "<> "
                        << ipaddress(here_.ip_address())
                        << "( " << ipaddress(here_.ip_address()) << " )");

                    // call parcelport connection function before setting future
                    connection_function_(new_ep, address[1]);

                    // if there is an entry for a locally started connection on this IP
                    // then set the future ready with the verbs endpoint
                    LOG_DEVEL_MSG("FI_CONNECTED setting future     "
                            << ipaddress(locality::ip_address(address)) << "<> "
                            << ipaddress(here_.ip_address())
                        << "( " << ipaddress(here_.ip_address()) << " )");

                    std::get<0>(endpoint_tmp_.find(address[1])->second).
                        set_value(new_ep);

                    // once the future is set, the entry can be removed?
//                    endpoint_tmp_.erase(present1.first);
                }
                break;
                case FI_NOTIFY:
                    LOG_DEBUG_MSG("Got FI_NOTIFY");
                    break;
                case FI_SHUTDOWN:
                    LOG_DEBUG_MSG("Got FI_SHUTDOWN");
                    break;
                case FI_MR_COMPLETE:
                    LOG_DEBUG_MSG("Got FI_MR_COMPLETE");
                    break;
                case FI_AV_COMPLETE:
                    LOG_DEBUG_MSG("Got FI_AV_COMPLETE");
                    break;
                case FI_JOIN_COMPLETE:
                    LOG_DEBUG_MSG("Got FI_JOIN_COMPLETE");
                    break;
                }
                //                   HPX_ASSERT(rd == sizeof(struct fi_eq_cm_entry));
                //                   HPX_ASSERT(cm_entry->fid == event_queue_->fid);
            }
            else {
                LOG_TIMED_MSG(poll, DEVEL, 5, "We did not get an event completion")
            }
            return 0;
        }

        // --------------------------------------------------------------------
        inline struct fid_domain * get_domain() {
            return fabric_domain_;
        }

        // --------------------------------------------------------------------
        inline rdma_memory_pool_ptr get_memory_pool() {
            return memory_pool_;
        }

        // --------------------------------------------------------------------
        typedef std::function<void(void* region, struct fid_ep *client, uint32_t len)>
            CompletionFunctionSend;
        typedef std::function<void(void* region, struct fid_ep *client, uint32_t len,
            fi_addr_t src)> CompletionFunctionRecv;
        typedef std::function<void(void* region, struct fid_ep *client,
            fi_addr_t src)> CompletionFunctionRdma;

        void setSendCompletionFunction(CompletionFunctionSend f) {
            send_completion_function_ = f;
        }
        void setRecvCompletionFunction(CompletionFunctionRecv f) {
            recv_completion_function_ = f;
        }
        void setRDMACompletionFunction(CompletionFunctionRdma f) {
            rdma_completion_function_ = f;
        }

        // ---------------------------------------------------------------------------
        // The number of outstanding work requests
        inline uint32_t get_receive_count() const { return preposted_receives_; }

        // ---------------------------------------------------------------------------
        void refill_client_receives(unsigned int preposts, bool force=true)
        {
            FUNC_START_DEBUG_MSG;
            LOG_DEBUG_MSG("Entering refill " << hexpointer(memory_pool_.get())
                << "size of waiting receives is "
                << decnumber(get_receive_count()));

#ifdef HPX_PARCELPORT_LIBFABRIC_ENDPOINT_RDM
            struct fid_ep *endpoint = ep_active_;
#else
            struct fid_ep *endpoint = ep_shared_rx_cxt_;
#endif
            while (get_receive_count() < preposts) {
                // if the pool has spare small blocks (just use 0 size) then
                // refill the queues, but don't wait, just abort if none are available
                if (force || memory_pool_->can_allocate_unsafe(
                    memory_pool_->small_.chunk_size()))
                {
                    auto region = memory_pool_->allocate_region(memory_pool_->small_.chunk_size());
                    void *desc = region->get_desc();
                    LOG_TRACE_MSG("Pre-Posting a receive to client size "
                        << hexnumber(memory_pool_->small_.chunk_size())
                        << " descriptor " << hexpointer(desc));
                    region->set_user_data(endpoint);
                    int ret = fi_recv(endpoint, region->get_address(), region->get_size(), desc, 0, region);
                    if (ret!=0 && ret != -FI_EAGAIN) {
                        throw fabric_error(ret, "pp_post_rx");
                    }
                    ++preposted_receives_;
                }
                else {
                    LOG_DEBUG_MSG("aborting refill can_allocate_unsafe false");
                    break; // don't block, if there are no free memory blocks
                }
            }
            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        void create_completion_queues(struct fi_info *info, int N)
        {
            FUNC_START_DEBUG_MSG;

            // only one thread must be allowed to create queues,
            // and it is only required once
            scoped_lock lock(initialization_mutex_);
            if (txcq!=nullptr || rxcq!=nullptr || av!=nullptr) {
                return;
            }

            int ret;
            // @TODO - why do we check this
            if (cq_attr.format == FI_CQ_FORMAT_UNSPEC) {
                LOG_DEBUG_MSG("Setting CQ attribute to FI_CQ_FORMAT_MSG");
                cq_attr.format = FI_CQ_FORMAT_MSG;
            }

            // open a completion queue on our fabric domain and set context ptr to tx queue
            cq_attr.wait_obj = FI_WAIT_NONE;
            cq_attr.size = info->tx_attr->size;
            info->tx_attr->op_flags |= FI_COMPLETION;
            cq_attr.flags |= FI_COMPLETION;
            LOG_DEBUG_MSG("Creating CQ with tx size " << decnumber(info->tx_attr->size));
            ret = fi_cq_open(fabric_domain_, &cq_attr, &txcq, &txcq);
            if (ret) throw fabric_error(ret, "fi_cq_open");

            // open a completion queue on our fabric domain and set context ptr to rx queue
            cq_attr.size = info->rx_attr->size;
            LOG_DEBUG_MSG("Creating CQ with rx size " << decnumber(info->rx_attr->size));
            ret = fi_cq_open(fabric_domain_, &cq_attr, &rxcq, &rxcq);
            if (ret) throw fabric_error(ret, "fi_cq_open");

            if (info->ep_attr->type == FI_EP_RDM || info->ep_attr->type == FI_EP_DGRAM) {
                if (info->domain_attr->av_type != FI_AV_UNSPEC)
                    av_attr.type = info->domain_attr->av_type;
                else {
                    LOG_DEVEL_MSG("Setting map type to FI_AV_MAP");
                    av_attr.type  = FI_AV_MAP;
                    av_attr.count = N;
                }

                LOG_DEVEL_MSG("Creating address vector ");
                ret = fi_av_open(fabric_domain_, &av_attr, &av, NULL);
                if (ret) throw fabric_error(ret, "fi_av_open");
            }
            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        std::pair<bool, hpx::shared_future<struct fid_ep*>> insert_new_future(
            uint32_t remote_ip)
        {

            LOG_ERROR_MSG("insert_new_future : Obsolete in RDM mode");
            std::terminate();

            LOG_DEVEL_MSG("Inserting future in map         "
                << ipaddress(here_.ip_address()) << "-> "
                << ipaddress(remote_ip)
                << "( " << ipaddress(here_.ip_address()) << " )");

            //
            hpx::promise<struct fid_ep*> new_endpoint_promise;
            hpx::future<struct fid_ep*>  new_endpoint_future =
                new_endpoint_promise.get_future();
            //
            auto fp_pair = std::make_pair(
                    remote_ip,
                    std::make_tuple(
                        std::move(new_endpoint_promise),
                        std::move(new_endpoint_future)));
             //
            auto it = endpoint_tmp_.insert(std::move(fp_pair));
            // if the insert failed, we must safely delete the future/promise
            if (!it.second) {
                LOG_DEVEL_MSG("Must safely delete promise");
            }

            // get the future that was inserted or already present
            // the future will become ready when the remote end accepts/rejects our connection
            // or we accept a connection from a remote
            hpx::shared_future<struct fid_ep*> result = std::get<1>(it.first->second);

            // if the insert fails due to a duplicate value, return the duplicate
            if (!it.second) {
                return std::make_pair(false, result);
            }
            return std::make_pair(true, result);
        }

        // --------------------------------------------------------------------
        fi_addr_t insert_address(const locality &remote)
        {
            FUNC_START_DEBUG_MSG;
            LOG_DEVEL_MSG("inserting address in vector "
                << ipaddress(remote.ip_address()));
            fi_addr_t result = 0xffffffff;
            int ret = fi_av_insert(av, remote.fabric_data(), 1, &result, 0, nullptr);
            if (ret < 0) {
                fabric_error(ret, "fi_av_insert");
            }
            else if (ret != 1) {
                fabric_error(ret, "fi_av_insert did not return 1");
            }
            endpoint_av_.insert(std::make_pair(remote.ip_address(), result));
            LOG_DEVEL_MSG("Address inserted in vector "
                << ipaddress(remote.ip_address()) << hexuint64(result));
            FUNC_END_DEBUG_MSG;
            return result;
        }

        // --------------------------------------------------------------------
        hpx::shared_future<struct fid_ep*> connect_to_server(const locality &remote)
        {

            LOG_ERROR_MSG("connect_to_server : Obsolete in RDM mode");
            std::terminate();

            const uint32_t &remote_ip = remote.ip_address();

            // Has a connection been started from here already?
            // Note: The future must be created before we call fi_connect
            // otherwise a connection may complete before the future is setup
            auto connection = insert_new_future(remote_ip);

            // if a connection is already underway, just return the future
            if (!connection.first) {
                LOG_DEVEL_MSG("connect to server : returning existing future");
                // the future will become ready when the remote end accepts/rejects
                // our connection - or we accept a connection from a remote
                return connection.second;
            }

            // for thread safety, make a copy of the fi_info before setting
            // the address in it. fi_freeinfo will free the dest_addr field.
            struct fi_info *new_info = fi_dupinfo(fabric_info_);
            new_info->dest_addrlen = locality::array_size;
            new_info->dest_addr = malloc(locality::array_size);
            std::memcpy(new_info->dest_addr, remote.fabric_data(), locality::array_size);

            uint64_t flags = 0;
            struct fi_info *fabric_info_active_;
            int ret = fi_getinfo(FI_VERSION(1,4), nullptr, nullptr,
                flags, new_info, &fabric_info_active_);
            if (ret) throw fabric_error(ret, "fi_getinfo");

            LOG_DEBUG_MSG("New connection for IP address "
                << ipaddress(remote.ip_address())
                << "Fabric info " << fi_tostr(fabric_info_active_, FI_TYPE_INFO));
            create_completion_queues(fabric_info_active_, 0);

            fid_ep *new_endpoint;
            new_endpoint_active(fabric_info_active_, &new_endpoint);

            // now it is safe to call connect
            LOG_DEVEL_MSG("Calling fi_connect         from "
                << ipaddress(here_.ip_address()) << "-> "
                << ipaddress(remote.ip_address())
                << "( " << ipaddress(here_.ip_address()) << " )");

            ret = fi_connect(new_endpoint, remote.fabric_data(), nullptr, 0);
            if (ret) throw fabric_error(ret, "fi_connect");

            LOG_DEBUG_MSG("Deleting new endpoint info structure");
            fi_freeinfo(fabric_info_active_);
            fi_freeinfo(new_info);

            return connection.second;
        }

//        void disconnect_from_server(struct fid_ep *client) {}

        void disconnect_all() {}

        bool active() { return false; }

    private:
        // store info about local device
        std::string           device_;
        std::string           interface_;
        sockaddr_in           local_addr_;

        // callback functions used for connection event handling
        ConnectionFunction    connection_function_;
        DisconnectionFunction disconnection_function_;

        // callback function for handling a completion event
        CompletionFunctionSend send_completion_function_;
        CompletionFunctionRecv recv_completion_function_;
        CompletionFunctionRdma rdma_completion_function_;

        // Pinned memory pool used for allocating buffers
        rdma_memory_pool_ptr  memory_pool_;

        // Shared completion queue for all endoints
        // Count outstanding receives posted to SRQ + Completion queue
        std::atomic<uint16_t> preposted_receives_;

        // only allow one thread to handle connect/disconnect events etc
        mutex_type            initialization_mutex_;
        mutex_type            endpoint_map_mutex_;

        // used to skip polling event channel too frequently
        typedef std::chrono::time_point<std::chrono::system_clock> time_type;
        time_type event_check_time_;
        uint32_t  event_pause_;

    };

    // Smart pointer for libfabric_controller obje
    typedef std::shared_ptr<libfabric_controller> libfabric_controller_ptr;

}}}}

#endif
