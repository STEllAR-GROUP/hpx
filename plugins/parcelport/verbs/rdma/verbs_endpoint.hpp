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
        )

        std::atomic<connection_state> state_;

        // ---------------------------------------------------------------------------
        // This constructor is used to create a local server endpoint.
        // It can accept incoming connections, or make outgoing ones.
        // Every node will create one server endpoint, and then for each connection
        // made to another node a new client endpoint will be constructed.
        // The endpoint constructed here will represent the local node.
        verbs_endpoint(struct sockaddr_in local_address) : verbs_sender_receiver(nullptr)
         {
            LOG_DEVEL_MSG("verbs_endpoint Listening Server Constructor");
            //
            init();
            create_id();
            bind(local_address);
         }

        // ---------------------------------------------------------------------------
        // This constructor is used when we have received a connection request and
        // create a client endpoint represent the remote end of the link.
        // NB. we only use one CompletionQ, send/recv share the same one.
        // This could be changed if need arises
        verbs_endpoint(
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
            remote_address_   = *(sockaddr_in*)(&cmId->route.addr.dst_addr);
            srq_              = SRQ;
            completion_queue_ = CompletionQ;
            memory_pool_      = pool;
            domain_           = domain;
            event_channel_    = event_channel;

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
            verbs_event_channel_ptr event_channel) :
                verbs_sender_receiver(nullptr)
        {
            LOG_DEVEL_MSG("verbs_endpoint create connection constructor "
                << sockaddress(&localAddress) << "to "
                << sockaddress(&remoteAddress));
            //
            event_channel_    = event_channel;
            init();
            create_id();
            //
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

            LOG_DEVEL_MSG("reset CQ ");
            completion_queue_.reset();

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
        void refill_preposts(unsigned int preposts) {
            //            LOG_DEBUG_MSG("Entering refill size of waiting receives is "
            //                << decnumber(get_receive_count()));
            while (get_receive_count() < preposts) {
                // if the pool has spare small blocks (just use 0 size) then
                // refill the queues, but don't wait, just abort if none are available
                if (this->memory_pool_->can_allocate_unsafe(
                    this->memory_pool_->small_.chunk_size_))
                {
                    LOG_TRACE_MSG("Pre-Posting a receive to client size "
                        << hexnumber(this->memory_pool_->small_.chunk_size_));
                    verbs_memory_region *region =
                        this->get_free_region(
                            this->memory_pool_->small_.chunk_size_);
                    this->post_recv_region_as_id_counted(region,
                        region->get_size());
                }
                else {
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
                        rdma_cm_event_type(-1), cm_event);
                    if (err != 0) return 0;
                    return f(cm_event);
                }
            );
        }

        // ---------------------------------------------------------------------------
        int get_event(verbs_event_channel::event_ack_type ack,
            rdma_cm_event_type event, struct rdma_cm_event *&cm_event)
        {
            return event_channel_->get_event(ack, event, cm_event);
        }

        // ---------------------------------------------------------------------------
        // resolve_address is called when we wish to make a connection to another node.
        // This endpoint is created on demand, so it does not share an event channel
        // between other nodes (unlike server endpoints)
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
        int handle_addr_resolved(struct rdma_cm_event *event, bool aborted=false)
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (state_!=connection_state::resolving_address) {
                rdma_error e(0, LOG_FORMAT_MSG("invalid state in resolving_address"
                    << aborted));
                return -1;
            }
            state_ = connection_state::route_resolved;
            verbs_event_channel::ack_event(event);

            // if this connection attempt has been aborted, exit cleanly
            if (aborted) {
                LOG_DEBUG_MSG("resolved addr aborted " << sockaddress(&remote_address_));
                return 0;
            }

            state_ = connection_state::address_resolved;

            LOG_DEVEL_MSG("resolved to " << sockaddress(&remote_address_)
                << "Current state is " << ToString(state_));

            // after resolving address, we must resolve route
            resolve_route();
            return 0;
        }

        // ---------------------------------------------------------------------------
        int resolve_route(void)
        {
            if (state_!=connection_state::address_resolved) {
                rdma_error e(0, LOG_FORMAT_MSG("invalid state in resolve_route"));
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
        int handle_route_resolved(struct rdma_cm_event *event, bool aborted=false)
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (state_!=connection_state::resolving_route) {
                rdma_error e(0, LOG_FORMAT_MSG("invalid state in handle_route_resolved"
                    << aborted));
                return -1;
            }
            state_ = connection_state::route_resolved;
            verbs_event_channel::ack_event(event);

            // if this connection attempt has been aborted, exit cleanly
            if (aborted) {
                LOG_DEBUG_MSG("resolved route aborted " << sockaddress(&remote_address_));
                return 0;
            }

            LOG_DEBUG_MSG("resolved route to " << sockaddress(&remote_address_)
                << "Current state is " << ToString(state_));

            // after resolving route, we must create a queue pair
            create_queue_pair(domain_, completion_queue_, completion_queue_,
                HPX_PARCELPORT_VERBS_MAX_WORK_REQUESTS, false);

            // make sure client has preposted receives
            // @TODO, when we use a shared receive queue, fix this
            refill_preposts(HPX_PARCELPORT_VERBS_MAX_PREPOSTS);

            // open the connection between this endpoint and the remote server endpoint
            return connect();
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
            param.initiator_depth = 1;
            param.rnr_retry_count = 7; // 7 = special code for infinite retries
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
            param.rnr_retry_count = 7; // 7 = special code for infinite retries
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
        int handle_establish(struct rdma_cm_event *event, bool aborted=false)
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (state_!=connection_state::connecting &&
                state_!=connection_state::accepting)
            {
                rdma_error e(0, LOG_FORMAT_MSG("invalid state in handle_establish "
                    << aborted));
                return -1;
            }

            if (event->event == RDMA_CM_EVENT_REJECTED) {
                LOG_DEVEL_MSG("2: Connection rejected for "
                    << sockaddress(&remote_address_)
                    << "( " << sockaddress(&local_address_) << ")");
                verbs_event_channel::ack_event(event);
                state_ = connection_state::terminated;
                LOG_DEVEL_MSG("Current state is " << ToString(state_));
                return -2;
            }
            else if (event->event == RDMA_CM_EVENT_ESTABLISHED) {
                state_ = connection_state::connected;
                verbs_event_channel::ack_event(event);
                LOG_DEVEL_MSG("connected to " << sockaddress(&remote_address_)
                    << "Current state is " << ToString(state_));
            }
            return 0;
        }

        // ---------------------------------------------------------------------------
        int disconnect()
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (state_!=connection_state::connected)
            {
                rdma_error e(0, LOG_FORMAT_MSG("invalid state in disconnect"));
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
        int handle_disconnect(struct rdma_cm_event *event, bool aborted=false)
        {
            LOG_DEVEL_MSG("Current state is " << ToString(state_));
            if (state_!=connection_state::disconnecting &&
                state_!=connection_state::terminated &&
                state_!=connection_state::connected)
            {
                rdma_error e(0, LOG_FORMAT_MSG("invalid state in handle_disconnect"
                    << aborted));
                return -1;
            }

            state_ = connection_state::disconnected;
            verbs_event_channel::ack_event(event);

            // if this connection attempt has been aborted, exit cleanly
            if (aborted) {
                LOG_DEBUG_MSG("resolved route aborted " << sockaddress(&remote_address_));
                return 0;
            }

            flush();

            LOG_DEBUG_MSG("Disconnected               "
                << "from " << sockaddress(&remote_address_)
                << "( " << sockaddress(&local_address_) << ") "
                << "Current state is " << ToString(state_));

            return 0;
        }

        // ---------------------------------------------------------------------------
        int create_srq(verbs_protection_domain_ptr domain)
        {
            try {
                srq_ = std::make_shared < verbs_shared_receive_queue
                    > (cmId_, domain);
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
        virtual inline struct ibv_srq *getsrq_() const {
            if (srq_ == nullptr)
                return nullptr;
            return srq_->getsrq_();
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

        // Tranaition the qp to an error state
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
        void create_id(void)
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
            qpAttributes.cap.max_send_wr = maxWorkRequests;
            qpAttributes.cap.max_recv_wr = maxWorkRequests;
            qpAttributes.cap.max_send_sge = 6; // 6;
            qpAttributes.cap.max_recv_sge = 6; // 6;
            qpAttributes.qp_context = this; // Save the pointer this object.
            qpAttributes.sq_sig_all = signalSendQueue;
            qpAttributes.qp_type = IBV_QPT_RC;
            qpAttributes.send_cq = sendCompletionQ->getQueue();
            qpAttributes.recv_cq = recvCompletionQ->getQueue();
            LOG_DEBUG_MSG("Setting SRQ to " << getsrq_());
            qpAttributes.srq = getsrq_();

            int rc = rdma_create_qp(cmId_, domain->getDomain(),
                &qpAttributes);

            LOG_DEBUG_MSG("After Create QP, SRQ is " << getsrq_());

            //   cmId_->qp = ibv_create_qp(domain->getDomain(), &qpAttributes);
            //   int rc = (cmId_->qp==nullptr);

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
                << " max inline data is " << hexnumber(qpAttributes.cap.max_inline_data));

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
