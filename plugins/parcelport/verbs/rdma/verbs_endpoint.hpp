// Copyright (c) 2016 John Biddiscombe
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at

#ifndef HPX_PARCELSET_POLICIES_VERBS_ENDPOINT_HPP
#define HPX_PARCELSET_POLICIES_VERBS_ENDPOINT_HPP

#include <hpx/lcos/promise.hpp>
#include <hpx/lcos/future.hpp>
//
#include <hpx/config/parcelport_verbs_defines.hpp>
//
#include <plugins/parcelport/verbs/rdma/rdma_logging.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_error.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_event_channel.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_shared_receive_queue.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_sender_receiver.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_completion_queue.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_memory_region.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_protection_domain.hpp>
//
#include <inttypes.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <iostream>
#include <iomanip>
#include <atomic>

#define HPX_PARCELPORT_VERBS_MAX_WORK_REQUESTS 1024

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    // Connection for RDMA operations with a remote partner.
    class verbs_endpoint : public verbs_sender_receiver
    {
    public:
        // ---------------------------------------------------------------------------
        // enum definition macro that generates logging string conversion
        // see rdma_logging.hpp for macro
        DEFINE_ENUM_WITH_STRING_CONVERSIONS(connection_state,
            (uninitialized)
            (resolving_address)
            (address_resolved)
            (resolving_route)
            (route_resolved)
            (connecting)
            (accepting)
            (rejecting)
            (connected)
            (disconnecting)
            (disconnected)
            (terminated)
            (aborted)
        )

        std::atomic<connection_state> state_;

        // ---------------------------------------------------------------------------
        // This constructor is used to create a local server endpoint.
        // It can accept incoming connections, or make outgoing ones.
        // Every node will create one server endpoint, and then for each connection
        // made to another node a new client endpoint will be constructed.
        // The endpoint constructed here will represent the local node.
        verbs_endpoint(
            struct sockaddr_in local_address)
            : verbs_sender_receiver(nullptr)
         {
            LOG_DEVEL_MSG("verbs_endpoint Listening Server Constructor");
            //
            init();
            create_cm_id();
            bind(local_address);
         }

        // ---------------------------------------------------------------------------
        // This constructor is used when we have received a connection request and
        // create a client endpoint to represent the remote end of the link.
        // NB. we only use one CompletionQ, send/recv share the same one.
        // This could be changed if need arises
        verbs_endpoint(
            struct sockaddr_in localAddress,
            struct rdma_cm_id *cmId,
            verbs_protection_domain_ptr domain,
            verbs_completion_queue_ptr CompletionQ,
            rdma_memory_pool_ptr pool,
            verbs_shared_receive_queue_ptr SRQ,
            verbs_event_channel_ptr event_channel) :
                verbs_sender_receiver(nullptr)
        {
            LOG_DEVEL_MSG("verbs_endpoint receive connection constructor");
            //
            event_channel_    = event_channel;
            init();

            // Use the input rdma connection management id.
            cmId_             = cmId;
            local_address_    = localAddress;
            remote_address_   = *(sockaddr_in*)(&cmId->route.addr.dst_addr);
            srq_              = SRQ;
            completion_queue_ = CompletionQ;
            memory_pool_      = pool;
            domain_           = domain;
            event_channel_    = event_channel;

            LOG_DEVEL_MSG("endpoint created with CQ "
                << hexpointer(completion_queue_.get()));

            // Create a queue pair. Both send and receive share a completion queue
            create_queue_pair(domain_, completion_queue_, completion_queue_,
                HPX_PARCELPORT_VERBS_MAX_WORK_REQUESTS, false);
        }

        // ---------------------------------------------------------------------------
        // This constructor is used when we make/start a new connection to a remote
        // node (as opposed to when they connect to us).
        // The constructed endpoint will represent the remote node.
        verbs_endpoint(
            struct sockaddr_in localAddress,
            struct sockaddr_in remoteAddress,
            verbs_protection_domain_ptr domain,
            verbs_completion_queue_ptr CompletionQ,
            rdma_memory_pool_ptr pool,
            verbs_shared_receive_queue_ptr SRQ,
            verbs_event_channel_ptr event_channel) :
                verbs_sender_receiver(nullptr)
        {
            LOG_DEVEL_MSG("verbs_endpoint create connection constructor "
                << sockaddress(&localAddress) << "to "
                << sockaddress(&remoteAddress));
            //
            event_channel_        = event_channel;
            init();
            create_cm_id();
            //
            srq_                  = SRQ;
            completion_queue_     = CompletionQ;
            memory_pool_          = pool;
            domain_               = domain;
            initiated_connection_ = true;

            // resolve ib addresses
            resolve_address(&localAddress, &remoteAddress,
                std::chrono::milliseconds(10000));
        }

        // ---------------------------------------------------------------------------
        ~verbs_endpoint(void)
        {
            LOG_DEVEL_MSG("reset domain");
            domain_.reset();

            // Destroy the rdma cm id and queue pair.
            if (cmId_ != nullptr) {
                if (cmId_->qp != nullptr) {
                    rdma_destroy_qp(cmId_); // No return code
                    LOG_DEVEL_MSG("destroyed queue pair");
                }

                if (rdma_destroy_id(cmId_) == 0) {
                    LOG_DEVEL_MSG("destroyed rdma cm id " << cmId_);
                    cmId_ = nullptr;
                }
                else {
                    int err = errno;
                    LOG_ERROR_MSG(
                        "error destroying rdma cm id " << cmId_ << ": "
                        << rdma_error::error_string(err));
                }
            }

            LOG_DEVEL_MSG("reset CQ ");
            completion_queue_.reset();

            LOG_DEBUG_MSG("releasing memory pool reference");
            memory_pool_.reset();

            // event channel is cleaned up by unique ptr destructor
            LOG_DEBUG_MSG("destroyed connection");
        }

        // ---------------------------------------------------------------------------
        verbs_completion_queue_ptr& get_completion_queue(void) {
            return completion_queue_;
        }

        // ---------------------------------------------------------------------------
        void refill_preposts(unsigned int preposts, bool force=true) {
            //            LOG_DEBUG_MSG("Entering refill size of waiting receives is "
            //                << decnumber(get_receive_count()));
            while (get_receive_count() < preposts) {
                // if the pool has spare small blocks (just use 0 size) then
                // refill the queues, but don't wait, just abort if none are available
                if (force || this->memory_pool_->can_allocate_unsafe(
                    this->memory_pool_->small_.chunk_size_))
                {
                    LOG_TRACE_MSG("Pre-Posting a receive to client size "
                        << hexnumber(this->memory_pool_->small_.chunk_size_));
                    verbs_memory_region *region =
                        this->get_free_region(
                            this->memory_pool_->small_.chunk_size_);
                    this->post_recv_region_as_id_counted_srq(region,
                        region->get_size(), getsrq());
                }
                else {
                    LOG_DEVEL_MSG("aborting refill can_allocate_unsafe false");
                    break; // don't block, if there are no free memory blocks
                }
            }
        }

        // ---------------------------------------------------------------------------
        inline void set_memory_pool(rdma_memory_pool_ptr pool) {
            this->memory_pool_ = pool;
        }

        // ---------------------------------------------------------------------------
        inline verbs_memory_region *get_free_region(size_t size)
        {
            verbs_memory_region* region = this->memory_pool_->allocate_region(size);
            if (!region) {
                LOG_ERROR_MSG("Error creating free memory region");
            }
            region->set_message_length(size);

            return region;
        }

        // ---------------------------------------------------------------------------
        // Called by server side prior to listening for connections
        int bind(struct sockaddr_in local_address)
        {
           local_address_ = local_address;
           // Bind address to the listening connection.
           LOG_DEVEL_MSG("binding " << sockaddress(&local_address_)
               << "to port " << decnumber(local_address_.sin_port));
           //
           int err = rdma_bind_addr(cmId_, (struct sockaddr *)&local_address_);
           if (err != 0) {
              err = abs(err);
              LOG_ERROR_MSG("error binding to address "
                  << sockaddress(&local_address_) << ": "
                  << rdma_error::error_string(err));
              return err;
           }
           LOG_DEBUG_MSG("bound rdma cm id to address " << sockaddress(&local_address_));
           return 0;
        }

        // ---------------------------------------------------------------------------
        // called by server side to enable clients to connect
        int listen(int backlog)
        {
           // Start listening for connections.
           int err = rdma_listen(cmId_, backlog);
           if (err != 0) {
              err = abs(err);
              LOG_ERROR_MSG("error listening for connections: "
                  << rdma_error::error_string(err));
              return err;
           }
           LOG_DEBUG_MSG("listening for connections with backlog " << backlog);
           return 0;
        }

        // ---------------------------------------------------------------------------
        // this poll for event function is used by the main server endpoint when
        // it is waiting for connection/disconnection requests etc
        // ack_event, deletes the cm_event data structure allocated by the CM,
        // so we do not ack and alow the event handler routine to do it
        template<typename Func>
        int poll_for_event(Func &&f)
        {
            return event_channel_->poll_verbs_event_channel(
                [this, &f]()
                {
                    struct rdma_cm_event *cm_event;
                    int err = event_channel_->get_event(verbs_event_channel::no_ack_event,
                        nullptr, cm_event);
                    if (err != 0) return 0;
                    return f(cm_event);
                }
            );
        }

        // ---------------------------------------------------------------------------
        int get_event(verbs_event_channel::event_ack_type ack,
            rdma_cm_event_type event, struct rdma_cm_event *&cm_event)
        {
            return event_channel_->get_event(ack, &event, cm_event);
        }

        // ---------------------------------------------------------------------------
        // resolve_address is called before we wish to make a connection to another node.
        // an address resolved event will be generated once this completes
        int resolve_address(
            struct sockaddr_in *localAddr,
            struct sockaddr_in *remoteAddr,
            std::chrono::milliseconds timeout)
        {
            // Resolve the addresses.
            LOG_DEVEL_MSG("resolving remote address "
                << sockaddress(localAddr) << ": "
                << sockaddress(remoteAddr));

            state_ = connection_state::resolving_address;

            // set our port to zero and let it find one
            localAddr->sin_port = 0 ;
            int rc = rdma_resolve_addr(cmId_,
                (struct sockaddr *) localAddr,
                (struct sockaddr *) remoteAddr, 1000); // Configurable timeout?

            if (rc != 0) {
                rdma_error e(errno, "rdma_resolve_addr() failed");
                LOG_ERROR_MSG("error resolving remote address "
                    << sockaddress(localAddr) << ": "
                    << sockaddress(remoteAddr) << ": "
                    << rdma_error::error_string(e.error_code()));
                throw e;
            }

            // Save the addresses.
            memcpy(&remote_address_, remoteAddr, sizeof(struct sockaddr_in));
            if (localAddr != nullptr) {
                memcpy(&local_address_, localAddr, sizeof(struct sockaddr_in));
            }

            LOG_DEVEL_MSG("rdma_resolve_addr     "
                << hexnumber(event_channel_->get_file_descriptor()) << "from "
                << sockaddress(&local_address_)
                << "to " << sockaddress(&remote_address_)
                << "( " << sockaddress(&local_address_) << ")");

            return 0;
        }

        // ---------------------------------------------------------------------------
        // called when we receive address resolved event
        int handle_addr_resolved(struct rdma_cm_event *event)
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (state_!=connection_state::resolving_address) {
                rdma_error e(0, "invalid state in resolving_address");
                std::terminate();
                return -1;
            }
            verbs_event_channel::ack_event(event);

            // set new state
            state_ = connection_state::address_resolved;

            LOG_DEVEL_MSG("resolved to " << sockaddress(&remote_address_)
                << "Current state is " << ToString(state_));

            // after resolving address, we must resolve route
            resolve_route();

            return 0;
        }

        // ---------------------------------------------------------------------------
        // after resolving address and before making connection, we must resolve route
        int resolve_route(void)
        {
            if (state_!=connection_state::address_resolved) {
                rdma_error e(0, LOG_FORMAT_MSG("invalid state in resolve_route"));
                std::terminate();
                throw e;
            }

            LOG_DEBUG_MSG("Calling rdma_resolve_route   "
                << "from " << sockaddress(&local_address_)
                << "to " << sockaddress(&remote_address_)
                << "( " << sockaddress(&local_address_) << ")");

            state_ = connection_state::resolving_route;

            // Resolve a route.
            int rc = rdma_resolve_route(cmId_, 1000); // Configurable timeout?
            if (rc != 0) {
                rdma_error e(rc, LOG_FORMAT_MSG("error resolving route to "
                    << sockaddress(&remote_address_)
                    << "from " << sockaddress(&local_address_)));
                return rc;
            }
            return 0;
        }

        // ---------------------------------------------------------------------------
        // Handles route_resolved event and starts a connection to the remote endpoint
        int handle_route_resolved(struct rdma_cm_event *event)
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (state_!=connection_state::resolving_route) {
                rdma_error e(0, "invalid state in handle_route_resolved");
                std::terminate();
                return -1;
            }
            state_ = connection_state::route_resolved;
            verbs_event_channel::ack_event(event);

            LOG_DEBUG_MSG("resolved route to " << sockaddress(&remote_address_)
                << "Current state is " << ToString(state_));

            // after resolving route, we must create a queue pair
            create_queue_pair(domain_, completion_queue_, completion_queue_,
                HPX_PARCELPORT_VERBS_MAX_WORK_REQUESTS, false);

            // open the connection between this endpoint and the remote endpoint
            return connect();
        }

        // ---------------------------------------------------------------------------
        // if we start a connection but have to abort it, then this function sets the
        // aborted state
        int abort()
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (state_==connection_state::resolving_address ||
                state_==connection_state::address_resolved ||
                state_==connection_state::resolving_route ||
                state_==connection_state::route_resolved ||
                state_==connection_state::connecting ||
                state_==connection_state::terminated)
            {
                state_ = connection_state::aborted;
            }
            else {
                rdma_error e(0, "invalid state in abort");
                std::terminate();
                return -1;
            }
            LOG_DEBUG_MSG("Aborted " << sockaddress(&remote_address_)
                << "Current state is " << ToString(state_));
            return 0;
        }

        // ---------------------------------------------------------------------------
        int accept()
        {
            LOG_DEVEL_MSG("Calling rdma_accept          from "
                << sockaddress(&remote_address_)
                << "to " << sockaddress(&local_address_)
                << "( " << sockaddress(&local_address_) << ")");

            // Accept the connection request.
            struct rdma_conn_param param;
            memset(&param, 0, sizeof(param));
            param.responder_resources = 1;
            param.initiator_depth     = 1;
            param.retry_count         = 0;  // ignored in accept (connect sets it)
            param.rnr_retry_count     = 7;  // 7 = special code for infinite retries
            //
            int rc = rdma_accept(cmId_, &param);
            if (rc != 0) {
                int err = errno;
                LOG_ERROR_MSG(
                    "error accepting connection: " << rdma_error::error_string(err));
                return err;
            }
            LOG_DEBUG_MSG("accepted connection from client "
                << sockaddress(&remote_address_));

            state_ = connection_state::accepting;
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            return 0;
        }

        // ---------------------------------------------------------------------------
        int reject(rdma_cm_id *id)
        {
            //
            // Debugging code to get ip address of soure/dest of event
            // NB: The src and dest fields refer to the message - not the connect request
            // so we are actually receiving a request from dest (but src of the msg)
            //
            struct sockaddr *ip_src = &cmId_->route.addr.src_addr;
            struct sockaddr_in *addr_src =
                reinterpret_cast<struct sockaddr_in *>(ip_src);
            //
            local_address_ = *addr_src;

            LOG_DEVEL_MSG("Calling rdma_reject          from "
                << sockaddress(&remote_address_)
                << "to " << sockaddress(&local_address_)
                << "( " << sockaddress(&local_address_) << ")");

            // Reject a connection request.
            int err = rdma_reject(id, 0, 0);
            if (err != 0) {
                LOG_ERROR_MSG("error rejecting connection on cmid "
                    << hexpointer(id) << rdma_error::error_string(errno));
                return -1;
            }

            LOG_DEVEL_MSG("Rejected connection from new client");
            return 0;
        }

        // ---------------------------------------------------------------------------
        // Initiate a connection to another node's server endpoint
        int connect()
        {
            LOG_DEVEL_MSG("rdma_connect          "
                << hexnumber(event_channel_->get_file_descriptor()) << "from "
                << sockaddress(&local_address_)
                << "to " << sockaddress(&remote_address_)
                << "( " << sockaddress(&local_address_) << ")");

            // Connect to the server.
            struct rdma_conn_param param;
            memset(&param, 0, sizeof(param));
            param.responder_resources = 1;
            param.initiator_depth = 2;
            // retry_count * 4.096 x 2 ^ qpattr.timeout microseconds
            param.retry_count     = 0;  // retries before ack timeout signals error
            param.rnr_retry_count = 7;  // 7 = special code for infinite retries
            //
            int rc = rdma_connect(cmId_, &param);
            if (rc != 0) {
                int err = errno;
                LOG_ERROR_MSG("error in rdma_connect to "
                    << sockaddress(&remote_address_)
                    << "from " << sockaddress(&local_address_)
                    << ": " << rdma_error::error_string(err));
                return err;
            }
            state_ = connection_state::connecting;
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            return 0;
        }

        // ---------------------------------------------------------------------------
        int handle_establish(struct rdma_cm_event *event)
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (event->event == RDMA_CM_EVENT_ESTABLISHED) {
                if (state_!=connection_state::connecting &&
                    state_!=connection_state::accepting)
                {
                    rdma_error e(0, "invalid state in handle_establish for "
                        "RDMA_CM_EVENT_ESTABLISHED");
                    std::terminate();
                    return -1;
                }
                state_ = connection_state::connected;
                verbs_event_channel::ack_event(event);
                LOG_DEVEL_MSG("connected to " << sockaddress(&remote_address_)
                    << "Current state is " << ToString(state_));

/*
                struct ibv_qp_attr      attr;
                struct ibv_qp_init_attr init_attr;
                //
                if (ibv_query_qp(cmId_->qp, &attr,
                    IBV_QP_STATE | IBV_QP_TIMEOUT, &init_attr)) {
                    LOG_DEVEL_MSG("Failed to query QP state\n");
                    std::terminate();
                    return -1;
                }
                LOG_DEVEL_MSG("Current state is " << attr.qp_state);
                LOG_DEVEL_MSG("Current timeout is " << int(attr.timeout));

                // set retry counter timeout value
                // 4.096 x 2 ^ attr.timeout microseconds
                attr.timeout = 13; // 8589935 usec (8.58 sec);
                //
                LOG_DEVEL_MSG("Modifying qp " << decnumber(cmId_->qp->qp_num));
                if (ibv_modify_qp(cmId_->qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT))
                {
                    rdma_error e(errno,
                        LOG_FORMAT_MSG("Failed to set QP timeout : qp "
                            << decnumber(cmId_->qp->qp_num)));
                    throw e;
                }
*/

            }
            else if (event->event == RDMA_CM_EVENT_REJECTED) {
                if (state_!=connection_state::aborted &&
                    state_!=connection_state::connecting)
                {
                    rdma_error e(0, "invalid state in handle_establish for "
                        "RDMA_CM_EVENT_REJECTED");
                    std::terminate();
                    return -1;
                }
                LOG_DEVEL_MSG("2: Connection rejected for "
                    << sockaddress(&remote_address_)
                    << "( " << sockaddress(&local_address_) << ")");
                verbs_event_channel::ack_event(event);
                state_ = connection_state::terminated;
                LOG_DEVEL_MSG("Current state is " << ToString(state_));
                return -2;
            }
            return 0;
        }

        // ---------------------------------------------------------------------------
        int disconnect()
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (state_!=connection_state::connected &&
                state_!=connection_state::aborted   &&
                state_!=connection_state::terminated)
            {
                rdma_error e(0, LOG_FORMAT_MSG("invalid state in disconnect "));
                std::terminate();
                throw e;
            }
            state_ = connection_state::disconnecting;

            LOG_DEVEL_MSG("Sending disconnect to " << sockaddress(&remote_address_)
                << "Current state is " << ToString(state_));

            // Disconnect the connection.
            int err = rdma_disconnect(cmId_);
            if (err != 0) {
                err = abs(err);
                LOG_ERROR_MSG(
                    "error disconnect: " << rdma_error::error_string(err));
                return err;
            }

            return 0;
        }

        // ---------------------------------------------------------------------------
        int handle_disconnect(struct rdma_cm_event *event)
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (state_!=connection_state::disconnecting &&
                state_!=connection_state::terminated &&
                state_!=connection_state::connected)
            {
                rdma_error e(0, "invalid state in handle_disconnect");
                std::terminate();
                return -1;
            }

            state_ = connection_state::disconnected;
            verbs_event_channel::ack_event(event);

            flush();

            LOG_DEBUG_MSG("Disconnected               "
                << "from " << sockaddress(&remote_address_)
                << "( " << sockaddress(&local_address_) << ") "
                << "Current state is " << ToString(state_));

            return 0;
        }

        // ---------------------------------------------------------------------------
        int handle_time_wait_exit(struct rdma_cm_event *event)
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (state_!=connection_state::disconnecting &&
                state_!=connection_state::terminated &&
                state_!=connection_state::connected)
            {
                rdma_error e(0, "invalid state in handle_disconnect");
                return -1;
            }

            state_ = connection_state::disconnected;
            verbs_event_channel::ack_event(event);

            flush();

            LOG_DEBUG_MSG("Time Wait Exit               "
                << "from " << sockaddress(&remote_address_)
                << "( " << sockaddress(&local_address_) << ") "
                << "Current state is " << ToString(state_));
            return 0;
        }

        // ---------------------------------------------------------------------------
        int create_srq(verbs_protection_domain_ptr domain)
        {
            try {
                srq_ = std::make_shared<verbs_shared_receive_queue>(domain);
            }
            catch (...) {
                return 0;
            }
            return 1;
        }

        // ---------------------------------------------------------------------------
        uint32_t get_qp_num(void) const {
            return cmId_->qp ? cmId_->qp->qp_num : 0;
        }

        // ---------------------------------------------------------------------------
        struct ibv_context *get_device_context(void) const {
            return cmId_->verbs;
        }

        // ---------------------------------------------------------------------------
        inline uint32_t get_local_ip_address(void) const {
            return local_address_.sin_addr.s_addr;
        }

        // ---------------------------------------------------------------------------
        inline in_port_t get_local_port(void) {
            if (local_address_.sin_port == 0) {
                local_address_.sin_port = rdma_get_src_port(cmId_);
            }
            return local_address_.sin_port;
        }

        // ---------------------------------------------------------------------------
        inline struct sockaddr_in *get_remote_address(void) {
            return &remote_address_;
        }

        // ---------------------------------------------------------------------------
        inline uint32_t get_remote_ip_address(void) const {
            return remote_address_.sin_addr.s_addr;
        }

        // ---------------------------------------------------------------------------
        inline in_port_t get_remote_port(void) const {
            return remote_address_.sin_port;
        }

        // ---------------------------------------------------------------------------
        inline verbs_shared_receive_queue_ptr SRQ() {
            return srq_;
        }

        // ---------------------------------------------------------------------------
        virtual inline struct ibv_srq *getsrq() const {
            return srq_ ? srq_->getsrq() : nullptr;
        }

        // ---------------------------------------------------------------------------
        inline bool is_client_endpoint(void) const {
            return initiated_connection_;
        }

        // ---------------------------------------------------------------------------
        connection_state get_state(void) const {
            return state_;
        }

        // ---------------------------------------------------------------------------
        void set_state(connection_state s) {
            state_ = s;
        }

        // ---------------------------------------------------------------------------
        verbs_event_channel_ptr get_event_channel(void) const {
            return event_channel_;
        }

        // ---------------------------------------------------------------------------
        // Transition the qp to an error state
        void flush()
        {
            // do noting if the qp was never created
            if (!cmId_->qp) {
                return;
            }
            //
            state_ = connection_state::disconnected;
            //
            struct ibv_qp_attr attr;
            memset(&attr, 0, sizeof(attr));
            attr.qp_state = IBV_QPS_ERR;
            //
            LOG_DEVEL_MSG("Flushing qp " << decnumber(cmId_->qp->qp_num));
            if (ibv_modify_qp(cmId_->qp, &attr, IBV_QP_STATE))
            {
                rdma_error e(errno,
                    LOG_FORMAT_MSG("Failed to flush qp "
                        << decnumber(cmId_->qp->qp_num)));
                throw e;
            }
            LOG_DEVEL_MSG("Current state is (flushed) " << ToString(state_));
        }

    protected:

        // ---------------------------------------------------------------------------
        void init(void)
        {
            // Initialize private data.
            memset(&local_address_, 0, sizeof(local_address_));
            memset(&remote_address_, 0, sizeof(remote_address_));
            if (!event_channel_) {
                event_channel_ = std::make_shared<verbs_event_channel>();
            }
            clear_counters();
            initiated_connection_ = false;
            state_ = connection_state::uninitialized;
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            return;
        }

        // ---------------------------------------------------------------------------
        void create_cm_id(void)
        {
            // Create the event channel.
            if (!event_channel_->get_verbs_event_channel()) {
                event_channel_->create_channel();
            }
            // Create the rdma cm id.
            LOG_DEVEL_MSG("Creating cmid with event channel "
                << hexnumber(event_channel_->get_file_descriptor()));
            int err = rdma_create_id(
                event_channel_->get_verbs_event_channel(), &cmId_, this, RDMA_PS_TCP);
            if (err != 0) {
                rdma_error e(err, "rdma_create_id() failed");
                LOG_ERROR_MSG(
                    "error creating rdma cm id: " << rdma_error::error_string(err));
                throw e;
            }
            LOG_DEVEL_MSG("created rdma cm id " << cmId_);
        }

        // ---------------------------------------------------------------------------
        void create_queue_pair(verbs_protection_domain_ptr domain,
            verbs_completion_queue_ptr sendCompletionQ,
            verbs_completion_queue_ptr recvCompletionQ,
            uint32_t maxWorkRequests, bool signalSendQueue)
        {
            // Create a queue pair.
            struct ibv_qp_init_attr qpAttributes;
            memset(&qpAttributes, 0, sizeof qpAttributes);
            qpAttributes.cap.max_send_wr  = maxWorkRequests;
            qpAttributes.cap.max_recv_wr  = maxWorkRequests;
            qpAttributes.cap.max_send_sge = 3; // 6;
            qpAttributes.cap.max_recv_sge = 3; // 6;
            qpAttributes.qp_context       = this;    // Save this pointer
            qpAttributes.sq_sig_all       = signalSendQueue;
            qpAttributes.qp_type          = IBV_QPT_RC;
            qpAttributes.send_cq          = sendCompletionQ->getQueue();
            qpAttributes.recv_cq          = recvCompletionQ->getQueue();
            LOG_DEVEL_MSG("Setting SRQ to " << getsrq());
            qpAttributes.srq              = getsrq();
            //
            int rc = rdma_create_qp(cmId_, domain->getDomain(), &qpAttributes);
            if (rc != 0) {
                rdma_error e(errno, "rdma_create_qp() failed");
                LOG_ERROR_MSG(
                    "error creating queue pair: " << hexpointer(this)
                    "local address " << sockaddress(&local_address_)
                << " remote address " << sockaddress(&remote_address_)
                << rdma_error::error_string(e.error_code()));
                throw e;
            }

            LOG_DEVEL_MSG("created queue pair " << decnumber(cmId_->qp->qp_num)
                << "max inline data is " << hexnumber(qpAttributes.cap.max_inline_data));

            return;
        }

        rdma_memory_pool_ptr memory_pool_;

        verbs_shared_receive_queue_ptr srq_;

        // Event channel for notification of RDMA connection management events.
        verbs_event_channel_ptr event_channel_;

        // Address of this (local) side of the connection.
        struct sockaddr_in local_address_;

        // Address of other (remote) side of the connection.
        struct sockaddr_in remote_address_;

        // if the client connected to the server, then set this flag so that
        // at shutdown, we use the correct flag for disconnect(mode)
        bool initiated_connection_;

        // Memory region for inbound messages.
        verbs_protection_domain_ptr domain_;

        // Completion queue.
        verbs_completion_queue_ptr completion_queue_;
    };

    // Smart pointer for verbs_endpoint object.
    typedef std::shared_ptr<verbs_endpoint> verbs_endpoint_ptr;

}}}}

#endif
