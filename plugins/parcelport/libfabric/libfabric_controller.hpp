//  Copyright (c) 2016 John Biddiscombe
//  Copyright (c) 2017 Thomas Heller
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
#include <plugins/parcelport/libfabric/receiver.hpp>
#include <plugins/parcelport/libfabric/sender.hpp>
#include <plugins/parcelport/libfabric/rma_receiver.hpp>
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
#include <array>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <cstdint>
#include <cstddef>
#include <cstring>
//
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include "fabric_error.hpp"

#if defined(HPX_PARCELPORT_LIBFABRIC_GNI) || \
    defined(HPX_PARCELPORT_LIBFABRIC_SOCKETS) || \
    defined(HPX_PARCELPORT_LIBFABRIC_PSM2)
# define HPX_PARCELPORT_LIBFABRIC_ENDPOINT_RDM
#else
# define HPX_PARCELPORT_LIBFABRIC_ENDPOINT_MSG
#endif

#ifdef HPX_PARCELPORT_LIBFABRIC_GNI
# include "rdma/fi_ext_gni.h"
#endif

#ifdef HPX_PARCELPORT_LIBFABRIC_HAVE_PMI
//
# include <pmi2.h>
//
# include <boost/archive/iterators/base64_from_binary.hpp>
# include <boost/archive/iterators/binary_from_base64.hpp>
# include <boost/archive/iterators/transform_width.hpp>

    using namespace boost::archive::iterators;

    typedef
        base64_from_binary<
            transform_width<std::string::const_iterator, 6, 8>
            > base64_t;

    typedef
        transform_width<
            binary_from_base64<std::string::const_iterator>, 8, 6
    > binary_t;
#endif

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

        // NOTE: Connection maps are not used for endpoint type RDM
        // when a new connection is requested, it will be completed asynchronously
        // we need a promise/future for each endpoint so that we can set the new
        // endpoint when the connection completes and is ready
        // Note - only used during connection, then deleted
        typedef std::tuple<
            hpx::promise<fid_ep *>,
            hpx::shared_future<fid_ep *>
        > promise_tuple_type;

        // lock types for maps
        typedef hpx::concurrent::unordered_map<uint32_t, promise_tuple_type>
            ::map_read_lock_type map_read_lock_type;
        typedef hpx::concurrent::unordered_map<uint32_t, promise_tuple_type>
            ::map_write_lock_type map_write_lock_type;

        // Map of connections started, needed until connection is completed
        hpx::concurrent::unordered_map<uint32_t, promise_tuple_type> endpoint_tmp_;
        std::unordered_map<uint32_t, fi_addr_t> endpoint_av_;

        locality here_;
        locality agas_;

        struct fi_info    *fabric_info_;
        struct fid_fabric *fabric_;
        struct fid_domain *fabric_domain_;
        // Server/Listener for RDMA connections.
        struct fid_pep    *ep_passive_;
        struct fid_ep     *ep_active_;
        struct fid_ep     *ep_shared_rx_cxt_;

        // we will use just one event queue for all connections
        struct fid_eq     *event_queue_;
        struct fid_cq     *txcq_, *rxcq_;
        struct fid_av     *av_;

        bool immediate_;

        // --------------------------------------------------------------------
        // constructor gets info from device and sets up all necessary
        // maps, queues and server endpoint etc
        libfabric_controller(
            std::string provider,
            std::string domain,
            std::string endpoint, int port=7910)
          : fabric_info_(nullptr)
          , fabric_(nullptr)
          , fabric_domain_(nullptr)
          , ep_passive_(nullptr)
          , ep_active_(nullptr)
          , ep_shared_rx_cxt_(nullptr)
          , event_queue_(nullptr)
            //
          , txcq_(nullptr)
          , rxcq_(nullptr)
          , av_(nullptr)
            //
          , immediate_(false)
        {
            FUNC_START_DEBUG_MSG;
            open_fabric(provider, domain, endpoint);

            // Create a memory pool for pinned buffers
            memory_pool_.reset(
                new rma_memory_pool<libfabric_region_provider>(fabric_domain_));

            // setup a passive listener, or an active RDM endpoint
            here_ = create_local_endpoint();
#ifndef HPX_PARCELPORT_LIBFABRIC_ENDPOINT_RDM
            create_event_queue();
#endif

            LOG_DEBUG_MSG("Calling boot PMI");
            boot_PMI();
            FUNC_END_DEBUG_MSG;
        }

        void boot_PMI()
        {
#ifdef HPX_PARCELPORT_LIBFABRIC_HAVE_PMI
            int spawned;
            int size;
            int rank;
            int appnum;

            LOG_DEBUG_MSG("Calling PMI init");
            PMI2_Init(&spawned, &size, &rank, &appnum);
            LOG_DEBUG_MSG("Called PMI init on rank" << decnumber(rank));

            // create address vector and queues we need if bootstrapping
            create_completion_queues(fabric_info_, size);

            // we must pass out libfabric data to other nodes
            // encode it as a string to put into the PMI KV store
            std::string encoded_locality(
                base64_t((const char*)(here_.fabric_data())),
                base64_t((const char*)(here_.fabric_data()) + locality::array_size));
            int encoded_length = encoded_locality.size();
            LOG_DEBUG_MSG("Encoded locality as " << encoded_locality
                << " with length " << decnumber(encoded_length));

            // Key name for PMI
            std::string pmi_key = "hpx_libfabric_" + std::to_string(rank);
            // insert out data in the KV store
            LOG_DEBUG_MSG("Calling PMI2_KVS_Put on rank " << decnumber(rank));
            PMI2_KVS_Put(pmi_key.data(), encoded_locality.data());

            // Wait for all to do the same
            LOG_DEBUG_MSG("Calling PMI2_KVS_Fence on rank " << decnumber(rank));
            PMI2_KVS_Fence();

            // read libfabric data for all nodes and insert into our Address vector
            for (int i = 0; i < size; ++i)
            {
                // read one locality key
                std::string pmi_key = "hpx_libfabric_" + std::to_string(i);
                char encoded_data[locality::array_size*2];
                int length = 0;
                PMI2_KVS_Get(0, i, pmi_key.data(), encoded_data,
                    encoded_length + 1, &length);
                if (length != encoded_length)
                {
                    LOG_ERROR_MSG("PMI value length mismatch, expected "
                        << decnumber(encoded_length) << "got " << decnumber(length));
                }

                // decode the string back to raw locality data
                LOG_DEBUG_MSG("Calling decode for " << decnumber(i)
                    << " locality data on rank " << decnumber(rank));
                locality new_locality;
                std::copy(binary_t(encoded_data),
                          binary_t(encoded_data + encoded_length),
                          (new_locality.fabric_data_writable()));

                // insert locality into address vector
                LOG_DEBUG_MSG("Calling insert_address for " << decnumber(i)
                    << "on rank " << decnumber(rank));
                insert_address(new_locality);
                LOG_DEBUG_MSG("rank " << decnumber(i)
                    << "added to address vector");
                if (i == 0)
                {
                    agas_ = new_locality;
                }
            }

            PMI2_Finalize();
#endif
        }

        // --------------------------------------------------------------------
        // clean up all resources
        ~libfabric_controller()
        {
            unsigned int messages_handled = 0;
            unsigned int acks_received    = 0;
            unsigned int msg_plain        = 0;
            unsigned int msg_rma          = 0;
            unsigned int sent_ack         = 0;
            unsigned int rma_reads        = 0;
            unsigned int recv_deletes     = 0;
            //
            for (auto &r : receivers_) {
                r.cleanup();
                // from receiver
                messages_handled += r.messages_handled_;
                acks_received    += r.acks_received_;
                // from rma_receivers
                msg_plain        += r.msg_plain_;
                msg_rma          += r.msg_rma_;
                sent_ack         += r.sent_ack_;
                rma_reads        += r.rma_reads_;
                recv_deletes     += r.recv_deletes_;
            }

            LOG_DEBUG_MSG(
                   "Received messages " << decnumber(messages_handled)
                << "Received acks "     << decnumber(acks_received)
                << "Sent acks "     << decnumber(sent_ack)
                << "Total reads "       << decnumber(rma_reads)
                << "Total deletes "     << decnumber(recv_deletes)
                << "deletes error " << decnumber(messages_handled - recv_deletes));

            // Cleaning up receivers to avoid memory leak errors.
            receivers_.clear();

            LOG_DEBUG_MSG("closing fabric_->fid");
            if (fabric_)
                fi_close(&fabric_->fid);
#ifdef HPX_PARCELPORT_LIBFABRIC_ENDPOINT_RDM
            LOG_DEBUG_MSG("closing ep_active_->fid");
            if (ep_active_)
                fi_close(&ep_active_->fid);
#else
            LOG_DEBUG_MSG("closing ep_passive_->fid");
            if (ep_passive_)
                fi_close(&ep_passive_->fid);
#endif
            LOG_DEBUG_MSG("closing event_queue_->fid");
            if (event_queue_)
                fi_close(&event_queue_->fid);
            LOG_DEBUG_MSG("closing fabric_domain_->fid");
            if (fabric_domain_)
                fi_close(&fabric_domain_->fid);
            LOG_DEBUG_MSG("closing ep_shared_rx_cxt_->fid");
            if (ep_shared_rx_cxt_)
                fi_close(&ep_shared_rx_cxt_->fid);
            // clean up
            LOG_DEBUG_MSG("freeing fabric_info");
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
            fabric_hints_->caps                   = FI_MSG | FI_RMA | FI_SOURCE |
                FI_WRITE | FI_READ | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_RMA_EVENT;
            fabric_hints_->mode                   = FI_CONTEXT | FI_LOCAL_MR;
            fabric_hints_->fabric_attr->prov_name = strdup(provider.c_str());
            LOG_DEBUG_MSG("fabric provider " << fabric_hints_->fabric_attr->prov_name);
            if (domain.size()>0) {
                fabric_hints_->domain_attr->name  = strdup(domain.c_str());
                LOG_DEBUG_MSG("fabric domain "   << fabric_hints_->domain_attr->name);
            }

            // use infiniband type basic registration for now
            fabric_hints_->domain_attr->mr_mode = FI_MR_BASIC;

            // Disable the use of progress threads
            fabric_hints_->domain_attr->control_progress = FI_PROGRESS_MANUAL;
            fabric_hints_->domain_attr->data_progress = FI_PROGRESS_MANUAL;

            // Enable thread safe mode Does not work with psm2 provider
            fabric_hints_->domain_attr->threading = FI_THREAD_SAFE;

            // Enable resource management
            fabric_hints_->domain_attr->resource_mgmt = FI_RM_ENABLED;

#ifdef HPX_PARCELPORT_LIBFABRIC_ENDPOINT_RDM
            LOG_DEBUG_MSG("Selecting endpoint type RDM");
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
            int ret = fi_getinfo(FI_VERSION(1,4), nullptr, nullptr,
                flags, fabric_hints_, &fabric_info_);
            if (ret) {
                throw fabric_error(ret, "Failed to get fabric info");
            }
            LOG_DEBUG_MSG("Fabric info " << fi_tostr(fabric_info_, FI_TYPE_INFO));

            immediate_ = (fabric_info_->rx_attr->mode & FI_RX_CQ_DATA)!=0;
            LOG_DEBUG_MSG("Fabric supports immediate data " << immediate_);
            LOG_EXCLUSIVE(bool context = (fabric_hints_->mode & FI_CONTEXT)!=0);
            LOG_DEBUG_MSG("Fabric requires FI_CONTEXT " << context);

            LOG_DEBUG_MSG("Creating fabric object");
            ret = fi_fabric(fabric_info_->fabric_attr, &fabric_, nullptr);
            if (ret) {
                throw fabric_error(ret, "Failed to get fi_fabric");
            }

            // Allocate a domain.
            LOG_DEBUG_MSG("Allocating domain ");
            ret = fi_domain(fabric_, fabric_info_, &fabric_domain_, nullptr);
            if (ret) throw fabric_error(ret, "fi_domain");

            // Cray specific. Disable memory registration cache
            _set_disable_registration();

            fi_freeinfo(fabric_hints_);
            FUNC_END_DEBUG_MSG;
        }

        // -------------------------------------------------------------------
        // create endpoint and get ready for possible communications
        void startup(parcelport *pp)
        {
            FUNC_START_DEBUG_MSG;
            //
#ifdef HPX_PARCELPORT_LIBFABRIC_ENDPOINT_RDM
            bind_endpoint_to_queues(ep_active_);
#else
            bind_endpoint_to_queues(ep_passive_);
            fabric_info_->handle = &(ep_passive->fid);

            LOG_DEBUG_MSG("Creating active endpoint");
            new_endpoint_active(fabric_info_, &ep_active_);
            LOG_DEBUG_MSG("active endpoint " << hexpointer(ep_active_);

            bind_endpoint_to_queues(ep_active_);
#endif

            // filling our vector of receivers...
            std::size_t num_receivers = HPX_PARCELPORT_LIBFABRIC_MAX_PREPOSTS;
            receivers_.reserve(num_receivers);
            for(std::size_t i = 0; i != num_receivers; ++i)
            {
                receivers_.emplace_back(pp, ep_active_, *memory_pool_);
            }
        }

        // --------------------------------------------------------------------
        // Special GNI extensions to disable memory registration cache

        // this helper function only works for string ops
        void _set_check_domain_op_value(int op, const char *value)
        {
#ifdef HPX_PARCELPORT_LIBFABRIC_GNI
            int ret;
            struct fi_gni_ops_domain *gni_domain_ops;
            char *get_val;

            ret = fi_open_ops(&fabric_domain_->fid, FI_GNI_DOMAIN_OPS_1,
                      0, (void **) &gni_domain_ops, nullptr);
            if (ret) throw fabric_error(ret, "fi_open_ops");
            LOG_DEBUG_MSG("domain ops returned " << hexpointer(gni_domain_ops));

            ret = gni_domain_ops->set_val(&fabric_domain_->fid,
                    (dom_ops_val_t)(op), &value);
            if (ret) throw fabric_error(ret, "set val (ops)");

            ret = gni_domain_ops->get_val(&fabric_domain_->fid,
                    (dom_ops_val_t)(op), &get_val);
            LOG_DEBUG_MSG("Cache mode set to " << get_val);
            if (std::string(value) != std::string(get_val))
                throw fabric_error(ret, "get val");
#endif
        }

        void _set_disable_registration()
        {
#ifdef HPX_PARCELPORT_LIBFABRIC_GNI
            _set_check_domain_op_value(GNI_MR_CACHE, "none");
#endif
        }

        // -------------------------------------------------------------------
        void create_event_queue()
        {
            LOG_DEBUG_MSG("Creating event queue");
            fi_eq_attr eq_attr = {};
            eq_attr.wait_obj = FI_WAIT_NONE;
            int ret = fi_eq_open(fabric_, &eq_attr, &event_queue_, nullptr);
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
                    &ep_shared_rx_cxt_, nullptr);
                if (ret) throw fabric_error(ret, "fi_srx_context");
            }
            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        locality create_local_endpoint()
        {
            struct fid *id;
            int ret;
#ifdef HPX_PARCELPORT_LIBFABRIC_ENDPOINT_RDM
            LOG_DEBUG_MSG("Creating active endpoint");
            new_endpoint_active(fabric_info_, &ep_active_);
            LOG_DEBUG_MSG("active endpoint " << hexpointer(ep_active_));
            id = &ep_active_->fid;
#else
            LOG_DEBUG_MSG("Creating passive endpoint");
            ret = fi_passive_ep(fabric_, fabric_info_, &ep_passive_, nullptr);
            if (ret) {
                throw fabric_error(ret, "Failed to create fi_passive_ep");
            }
            LOG_DEBUG_MSG("passive endpoint " << hexpointer(ep_passive_));
            id = &ep_passive_->fid;
#endif
            locality::locality_data local_addr;
            std::size_t addrlen = locality::array_size;
            LOG_DEBUG_MSG("Fetching local address using size " << decnumber(addrlen));
            ret = fi_getname(id, local_addr.data(), &addrlen);
            if (ret || (addrlen>locality::array_size)) {
                fabric_error(ret, "fi_getname - size error or other problem");
            }

            LOG_EXCLUSIVE(
            {
                std::stringstream temp1;
                for (std::size_t i=0; i<locality::array_length; ++i) {
                    temp1 << ipaddress(local_addr[i]);
                }
                LOG_DEBUG_MSG("address info is " << temp1.str().c_str());
                std::stringstream temp2;
                for (std::size_t i=0; i<locality::array_length; ++i) {
                    temp2 << hexuint32(local_addr[i]);
                }
                LOG_DEBUG_MSG("address info is " << temp2.str().c_str());
            });
            FUNC_END_DEBUG_MSG;
            return locality(local_addr);
        }

        // --------------------------------------------------------------------
        void new_endpoint_active(struct fi_info *info, struct fid_ep **new_endpoint)
        {
            FUNC_START_DEBUG_MSG;
            // create an 'active' endpoint that can be used for sending/receiving
            LOG_DEBUG_MSG("Creating active endpoint");
            LOG_DEBUG_MSG("Got info mode " << (info->mode & FI_NOTIFY_FLAGS_ONLY));
            int ret = fi_endpoint(fabric_domain_, info, new_endpoint, nullptr);
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
            if (av_) {
                LOG_DEBUG_MSG("Binding endpoint to AV");
                ret = fi_ep_bind(endpoint, &av_->fid, 0);
                if (ret) throw fabric_error(ret, "bind event_queue_");
            }

            if (txcq_) {
                LOG_DEBUG_MSG("Binding endpoint to TX CQ");
                ret = fi_ep_bind(endpoint, &txcq_->fid, FI_TRANSMIT);
                if (ret) throw fabric_error(ret, "bind txcq");
            }

            if (rxcq_) {
                LOG_DEBUG_MSG("Binding endpoint to RX CQ");
                ret = fi_ep_bind(endpoint, &rxcq_->fid, FI_RECV);
                if (ret) throw fabric_error(ret, "rxcq");
            }

            if (ep_shared_rx_cxt_) {
                LOG_DEBUG_MSG("Binding endpoint to shared receive context");
                ret = fi_ep_bind(endpoint, &ep_shared_rx_cxt_->fid, 0);
                if (ret) throw fabric_error(ret, "ep_shared_rx_cxt_");
            }

            LOG_DEBUG_MSG("Enabling endpoint " << hexpointer(endpoint));
            ret = fi_enable(endpoint);
            if (ret) throw fabric_error(ret, "fi_enable");

            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        void initialize_localities(hpx::agas::addressing_service &as)
        {
            FUNC_START_DEBUG_MSG;
#ifndef HPX_PARCELPORT_LIBFABRIC_HAVE_PMI
            std::uint32_t N = hpx::get_config().get_num_localities();
            LOG_DEBUG_MSG("Parcelport initialize_localities with " << N << " localities");

            // make sure address vector is created
            create_completion_queues(fabric_info_, N);

            for (std::uint32_t i=0; i<N; ++i) {
                hpx::naming::gid_type l = hpx::naming::get_gid_from_locality_id(i);
                LOG_DEBUG_MSG("Resolving locality" << l);
                // each locality may be reachable by mutiplte parcelports
                const parcelset::endpoints_type &res = as.resolve_locality(l);
                // get the fabric related data
                auto it = res.find("libfabric");
                LOG_DEBUG_MSG("locality resolution " << it->first << " => " <<it->second);
                const hpx::parcelset::locality &fabric_locality = it->second;
                const locality &loc = fabric_locality.get<locality>();
                // put the provide specific data into the address vector
                // so that we can look it  up later
                /*fi_addr_t dummy =*/ insert_address(loc);
            }
#endif
            LOG_DEBUG_MSG("Done getting localities ");
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
        typedef std::function<void(fid_ep *endpoint, uint32_t ipaddr)>
            ConnectionFunction;
        typedef std::function<void(fid_ep *endpoint, uint32_t ipaddr)>
            DisconnectionFunction;

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
            int work = poll_for_work_completions();

#ifdef HPX_PARCELPORT_LIBFABRIC_ENDPOINT_MSG
            work += poll_event_queue(stopped);
#endif
            return work;
        }

        // --------------------------------------------------------------------
        int poll_for_work_completions()
        {
            // @TODO, disable polling until queues are initialized to avoid this check
            // if queues are not setup, don't poll
            if (HPX_UNLIKELY(!rxcq_)) return 0;
            //
            return poll_send_queue() + poll_recv_queue();
        }

        // --------------------------------------------------------------------
        int poll_send_queue()
        {
            LOG_TIMED_INIT(poll);
            LOG_TIMED_BLOCK(poll, DEVEL, 5.0, { LOG_DEBUG_MSG("poll_send_queue"); });

            fi_cq_msg_entry entry;
            int ret = 0;
            {
                std::unique_lock<mutex_type> l(polling_mutex_, std::try_to_lock);
                if (l)
                    ret = fi_cq_read(txcq_, &entry, 1);
            }
            if (ret>0) {
                LOG_DEBUG_MSG("Completion txcq wr_id "
                    << fi_tostr(&entry.flags, FI_TYPE_OP_FLAGS)
                    << " (" << decnumber(entry.flags) << ") "
                    << "context " << hexpointer(entry.op_context)
                    << "length " << hexuint32(entry.len));
                if (entry.flags & FI_RMA) {
                    LOG_DEBUG_MSG("Received a txcq RMA completion "
                        << "Context " << hexpointer(entry.op_context));
                    rma_receiver* rcv = reinterpret_cast<rma_receiver*>(entry.op_context);
                    rcv->handle_rma_read_completion();
                }
                else if (entry.flags == (FI_MSG | FI_SEND)) {
                    LOG_DEBUG_MSG("Received a txcq RMA send completion");
                    sender* handler = reinterpret_cast<sender*>(entry.op_context);
                    handler->handle_send_completion();
                }
                else {
                    LOG_DEBUG_MSG("$$$$$ Received an unknown txcq completion ***** "
                        << decnumber(entry.flags));
                    std::terminate();
                }
                return 1;
            }
            else if (ret==0 || ret==-FI_EAGAIN) {
                // do nothing, we will try again on the next check
                LOG_TIMED_MSG(poll, DEVEL, 10, "txcq FI_EAGAIN");
            }
            else if (ret == -FI_EAVAIL) {
                struct fi_cq_err_entry e = {};
                int err_sz = fi_cq_readerr(txcq_, &e ,0);
                // from the manpage 'man 3 fi_cq_readerr'
                //
                // On error, a negative value corresponding to
                // 'fabric errno' is returned
                //
                if(e.err == err_sz) {
                    LOG_ERROR_MSG("txcq_ Error with len " << hexlength(e.len)
                        << "context " << hexpointer(e.op_context));
                }
                // flags might not be set correctly
                if (e.flags == (FI_MSG | FI_SEND)) {
                    LOG_ERROR_MSG("txcq Error for FI_SEND with len " << hexlength(e.len)
                        << "context " << hexpointer(e.op_context));
                }
                if (e.flags & FI_RMA) {
                    LOG_ERROR_MSG("txcq Error for FI_RMA with len " << hexlength(e.len)
                        << "context " << hexpointer(e.op_context));
                }
                rma_base *base = reinterpret_cast<rma_base*>(e.op_context);
                base->handle_error(e);
            }
            else {
                LOG_ERROR_MSG("unknown error in completion txcq read");
            }
            return 0;
        }

        // --------------------------------------------------------------------
        int poll_recv_queue()
        {
            LOG_TIMED_INIT(poll);
            LOG_TIMED_BLOCK(poll, DEVEL, 5.0, { LOG_DEBUG_MSG("poll_recv_queue"); });

            int result = 0;
            fi_addr_t src_addr;
            fi_cq_msg_entry entry;

            // receives will use fi_cq_readfrom as we want the source address
            int ret = 0;
            {
                std::unique_lock<mutex_type> l(polling_mutex_, std::try_to_lock);
                if (l)
                    ret = fi_cq_readfrom(rxcq_, &entry, 1, &src_addr);
            }
            if (ret>0) {
                LOG_DEBUG_MSG("Completion rxcq wr_id "
                    << fi_tostr(&entry.flags, FI_TYPE_OP_FLAGS)
                    << " (" << decnumber(entry.flags) << ") "
                    << "source " << hexpointer(src_addr)
                    << "context " << hexpointer(entry.op_context)
                    << "length " << hexuint32(entry.len));
                if (src_addr == FI_ADDR_NOTAVAIL)
                {
                    LOG_DEBUG_MSG("Source address not available...\n");
                    std::terminate();
                }
//                     if ((entry.flags & FI_RMA) == FI_RMA) {
//                         LOG_DEBUG_MSG("Received an rxcq RMA completion");
//                     }
                else if (entry.flags == (FI_MSG | FI_RECV)) {
                    LOG_DEBUG_MSG("Received an rxcq recv completion "
                        << hexpointer(entry.op_context));
                    reinterpret_cast<receiver *>(entry.op_context)->
                        handle_recv(src_addr, entry.len);
                }
                else {
                    LOG_DEBUG_MSG("Received an unknown rxcq completion "
                        << decnumber(entry.flags));
                    std::terminate();
                }
                result = 1;
            }
            else if (ret==0 || ret==-FI_EAGAIN) {
                // do nothing, we will try again on the next check
                LOG_TIMED_MSG(poll, DEVEL, 10, "rxcq FI_EAGAIN");
            }
            else if (ret == -FI_EAVAIL) {
                struct fi_cq_err_entry e = {};
                int err_sz = fi_cq_readerr(rxcq_, &e ,0);
                // from the manpage 'man 3 fi_cq_readerr'
                //
                // On error, a negative value corresponding to
                // 'fabric errno' is returned
                //
                if(e.err == err_sz) {
                    LOG_ERROR_MSG("txcq_ Error with len " << hexlength(e.len)
                        << "context " << hexpointer(e.op_context));
                }
                LOG_ERROR_MSG("rxcq Error with flags " << hexlength(e.flags)
                    << "len " << hexlength(e.len));
            }
            else {
                LOG_ERROR_MSG("unknown error in completion rxcq read");
            }
            return result;
        }

        // --------------------------------------------------------------------
        int poll_event_queue(bool stopped=false)
        {
            LOG_TIMED_INIT(poll);
            LOG_TIMED_BLOCK(poll, DEVEL, 5.0,
                {
                    LOG_DEBUG_MSG("Polling event completion channel");
                }
            )
            struct fi_eq_cm_entry *cm_entry;
//             struct fi_eq_entry    *entry;
            struct fid_ep         *new_ep;
//             uint32_t *addr;
            uint32_t event;
            std::array<char, 256> buffer;
            ssize_t rd = fi_eq_read(event_queue_, &event,
                buffer.data(), sizeof(buffer), 0);
            if (rd > 0) {
                LOG_DEBUG_MSG("fi_eq_cm_entry " << decnumber(sizeof(fi_eq_cm_entry))
                    << " fi_eq_entry " << decnumber(sizeof(fi_eq_entry)));
                LOG_DEBUG_MSG("got event " << event << " with bytes = " << decnumber(rd));
                switch (event) {
                case FI_CONNREQ:
                {
                    cm_entry = reinterpret_cast<struct fi_eq_cm_entry*>(buffer.data());
                    locality::locality_data addressinfo;
                    std::memcpy(addressinfo.data(), cm_entry->info->dest_addr,
                        locality::array_size);
                    locality loc(addressinfo);
                    LOG_DEBUG_MSG("FI_CONNREQ                 from "
                        << ipaddress(loc.ip_address()) << "-> "
                        << ipaddress(here_.ip_address())
                        << "( " << ipaddress(here_.ip_address()) << " )");
                    {
                        auto result = insert_new_future(loc.ip_address());
                        // if the insert fails, it means we have a connection
                        // already in progress, reject if we are a lower ip address
                        if (!result.first && loc.ip_address()>here_.ip_address()) {
                            LOG_DEBUG_MSG("FI_CONNREQ priority fi_reject   "
                                << ipaddress(loc.ip_address()) << "-> "
                                << ipaddress(here_.ip_address())
                                << "( " << ipaddress(here_.ip_address()) << " )");
//                            int ret = fi_reject(ep_passive_, cm_entry->info->handle,
//                                nullptr, 0);
//                            if (ret) throw fabric_error(ret, "new_ep fi_reject failed");
                            fi_freeinfo(cm_entry->info);
                            return 0;
                        }
                        // create a new endpoint for this request and accept it
                        new_endpoint_active(cm_entry->info, &new_ep);
                        LOG_DEBUG_MSG("Calling fi_accept               "
                            << ipaddress(loc.ip_address()) << "-> "
                            << ipaddress(here_.ip_address())
                            << "( " << ipaddress(here_.ip_address()) << " )");
                        int ret = fi_accept(new_ep, &here_.ip_address(),
                            sizeof(uint32_t));
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
                    LOG_DEBUG_MSG("FI_CONNECTED " << hexpointer(new_ep)
                        << ipaddress(locality::ip_address(address)) << "<> "
                        << ipaddress(here_.ip_address())
                        << "( " << ipaddress(here_.ip_address()) << " )");

                    // call parcelport connection function before setting future
                    connection_function_(new_ep, address[1]);

                    // if there is an entry for a locally started connection on this IP
                    // then set the future ready with the verbs endpoint
                    LOG_DEBUG_MSG("FI_CONNECTED setting future     "
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
        inline rma_memory_pool<libfabric_region_provider>& get_memory_pool() {
            return *memory_pool_;
        }

        // --------------------------------------------------------------------
        void create_completion_queues(struct fi_info *info, int N)
        {
            FUNC_START_DEBUG_MSG;

            // only one thread must be allowed to create queues,
            // and it is only required once
            scoped_lock lock(initialization_mutex_);
            if (txcq_!=nullptr || rxcq_!=nullptr || av_!=nullptr) {
                return;
            }

            int ret;

            fi_cq_attr cq_attr = {};
            // @TODO - why do we check this
//             if (cq_attr.format == FI_CQ_FORMAT_UNSPEC) {
                LOG_DEBUG_MSG("Setting CQ attribute to FI_CQ_FORMAT_MSG");
                cq_attr.format = FI_CQ_FORMAT_MSG;
//             }

            // open completion queue on fabric domain and set context ptr to tx queue
            cq_attr.wait_obj = FI_WAIT_NONE;
            cq_attr.size = info->tx_attr->size;
            info->tx_attr->op_flags |= FI_COMPLETION;
            cq_attr.flags = 0;//|= FI_COMPLETION;
            LOG_DEBUG_MSG("Creating CQ with tx size " << decnumber(info->tx_attr->size));
            ret = fi_cq_open(fabric_domain_, &cq_attr, &txcq_, &txcq_);
            if (ret) throw fabric_error(ret, "fi_cq_open");

            // open completion queue on fabric domain and set context ptr to rx queue
            cq_attr.size = info->rx_attr->size;
            LOG_DEBUG_MSG("Creating CQ with rx size " << decnumber(info->rx_attr->size));
            ret = fi_cq_open(fabric_domain_, &cq_attr, &rxcq_, &rxcq_);
            if (ret) throw fabric_error(ret, "fi_cq_open");


            fi_av_attr av_attr = {};
            if (info->ep_attr->type == FI_EP_RDM || info->ep_attr->type == FI_EP_DGRAM) {
                if (info->domain_attr->av_type != FI_AV_UNSPEC)
                    av_attr.type = info->domain_attr->av_type;
                else {
                    LOG_DEBUG_MSG("Setting map type to FI_AV_MAP");
                    av_attr.type  = FI_AV_MAP;
                    av_attr.count = N;
                }

                LOG_DEBUG_MSG("Creating address vector ");
                ret = fi_av_open(fabric_domain_, &av_attr, &av_, nullptr);
                if (ret) throw fabric_error(ret, "fi_av_open");
            }
            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        std::pair<bool, hpx::shared_future<struct fid_ep*>> insert_new_future(
            uint32_t remote_ip)
        {

            LOG_DEBUG_MSG("insert_new_future : Obsolete in RDM mode");
            std::terminate();

            LOG_DEBUG_MSG("Inserting future in map         "
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
                LOG_DEBUG_MSG("Must safely delete promise");
            }

            // get the future that was inserted or already present
            // the future will become ready when remote end accepts/rejects connection
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
            LOG_DEBUG_MSG("inserting address in vector "
                << ipaddress(remote.ip_address()));
            fi_addr_t result = 0xffffffff;
            int ret = fi_av_insert(av_, remote.fabric_data(), 1, &result, 0, nullptr);
            if (ret < 0) {
                fabric_error(ret, "fi_av_insert");
            }
            else if (ret != 1) {
                fabric_error(ret, "fi_av_insert did not return 1");
            }
            endpoint_av_.insert(std::make_pair(remote.ip_address(), result));
            LOG_DEBUG_MSG("Address inserted in vector "
                << ipaddress(remote.ip_address()) << hexuint64(result));
            FUNC_END_DEBUG_MSG;
            return result;
        }

        // --------------------------------------------------------------------
        hpx::shared_future<struct fid_ep*> connect_to_server(const locality &remote)
        {

            LOG_DEBUG_MSG("connect_to_server : Obsolete in RDM mode");
            std::terminate();

            const uint32_t &remote_ip = remote.ip_address();

            // Has a connection been started from here already?
            // Note: The future must be created before we call fi_connect
            // otherwise a connection may complete before the future is setup
            auto connection = insert_new_future(remote_ip);

            // if a connection is already underway, just return the future
            if (!connection.first) {
                LOG_DEBUG_MSG("connect to server : returning existing future");
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
            LOG_DEBUG_MSG("Calling fi_connect         from "
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

        void disconnect_all() {}

        bool active() { return false; }

    private:
        // store info about local device
        std::string  device_;
        std::string  interface_;
        sockaddr_in  local_addr_;

        // callback functions used for connection event handling
        ConnectionFunction    connection_function_;
        DisconnectionFunction disconnection_function_;

        // Pinned memory pool used for allocating buffers
        std::unique_ptr<rma_memory_pool<libfabric_region_provider>>  memory_pool_;

        // Shared completion queue for all endoints
        // Count outstanding receives posted to SRQ + Completion queue
        std::vector<receiver> receivers_;

        // only allow one thread to handle connect/disconnect events etc
        mutex_type            initialization_mutex_;
        mutex_type            endpoint_map_mutex_;
        mutex_type            polling_mutex_;

        // used to skip polling event channel too frequently
        typedef std::chrono::time_point<std::chrono::system_clock> time_type;
        time_type event_check_time_;
        uint32_t  event_pause_;

    };

    // Smart pointer for libfabric_controller obje
    typedef std::shared_ptr<libfabric_controller> libfabric_controller_ptr;

}}}}

#endif
