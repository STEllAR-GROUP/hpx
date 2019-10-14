//  Copyright (c) 2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_RDMA_CONTROLLER_HPP
#define HPX_PARCELSET_POLICIES_VERBS_RDMA_CONTROLLER_HPP

// config
#include <hpx/config/defines.hpp>
//
#include <hpx/lcos/local/shared_mutex.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/lcos/future.hpp>
//
#include <plugins/parcelport/parcelport_logging.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_error.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_locks.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_endpoint.hpp>
//
#include <plugins/parcelport/unordered_map.hpp>
//
#include <atomic>
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
// @TODO : Remove the client map pair as we have a copy in the verbs_parcelport class
// that does almost the same job.
// @TODO : Most of this code could be moved into the main parcelport, or the endpoint
// classes
namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    class rdma_controller
    {
    public:
        typedef hpx::lcos::local::spinlock mutex_type;
        typedef hpx::parcelset::policies::verbs::unique_lock<mutex_type> unique_lock;
        typedef hpx::parcelset::policies::verbs::scoped_lock<mutex_type> scoped_lock;

        // constructor gets infor from device and sets up all necessary
        // maps, queues and server endpoint etc
        rdma_controller(const char *device, const char *interface, int port);

        // clean up all resources
        ~rdma_controller();

        // initiate a listener for connections
        int startup();

        // returns true when all connections have been disconnected and none are active
        bool isTerminated() {
            return (qp_endpoint_map_.size() == 0);
        }

        // types we need for connection and disconnection callback functions
        // into the main parcelport code.
        typedef std::function<void(verbs_endpoint_ptr)>       ConnectionFunction;
        typedef std::function<int(verbs_endpoint_ptr client)> DisconnectionFunction;

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
        int poll_endpoints(bool stopped=false);

        int poll_for_work_completions(bool stopped=false);

        inline verbs_completion_queue *get_completion_queue() const {
            return completion_queue_.get();
        }

        inline verbs_endpoint *get_client_from_completion(struct ibv_wc &completion)
        {
            return qp_endpoint_map_.at(completion.qp_num).get();
        }

        inline verbs_protection_domain_ptr get_protection_domain() {
            return this->protection_domain_;
        }

        verbs_endpoint* get_endpoint(uint32_t remote_ip) {
            return (std::get<0>(connections_started_.find(remote_ip)->second)).get();

        }

        inline rdma_memory_pool_ptr get_memory_pool() {
            return memory_pool_;
        }

        typedef std::function<int(struct ibv_wc completion, verbs_endpoint *client)>
            CompletionFunction;

        void setCompletionFunction(CompletionFunction f) {
            this->completion_function_ = f;
        }

        void refill_client_receives(bool force=true);

        hpx::shared_future<verbs_endpoint_ptr> connect_to_server(uint32_t remote_ip);

        void disconnect_from_server(verbs_endpoint_ptr client);

        void disconnect_all();

        bool active();

    private:
        void debug_connections();

        int handle_event(struct rdma_cm_event *cm_event, verbs_endpoint *client);

        int handle_connect_request(
            struct rdma_cm_event *cm_event, std::uint32_t remote_ip);

        int start_server_connection(uint32_t remote_ip);

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
        verbs_protection_domain_ptr protection_domain_;
        // Pinned memory pool used for allocating buffers
        rdma_memory_pool_ptr        memory_pool_;
        // Server/Listener for RDMA connections.
        verbs_endpoint_ptr          server_endpoint_;
        // Shared completion queue for all endoints
        verbs_completion_queue_ptr  completion_queue_;
        // Count outstanding receives posted to SRQ + Completion queue
        std::atomic<uint16_t>       preposted_receives_;

        // only allow one thread to handle connect/disconnect events etc
        mutex_type            controller_mutex_;

        typedef std::tuple<
            verbs_endpoint_ptr,
            hpx::promise<verbs_endpoint_ptr>,
            hpx::shared_future<verbs_endpoint_ptr>
        > promise_tuple_type;
        //
        typedef std::pair<const uint32_t, promise_tuple_type> ClientMapPair;
        typedef std::pair<const uint32_t, verbs_endpoint_ptr> QPMapPair;

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

        typedef std::pair<uint32_t, verbs_endpoint_ptr> qp_map_type;
        hpx::concurrent::unordered_map<uint32_t, verbs_endpoint_ptr> qp_endpoint_map_;

        // used to skip polling event channel too frequently
        typedef std::chrono::time_point<std::chrono::system_clock> time_type;
        time_type event_check_time_;
        uint32_t  event_pause_;

    };

    // Smart pointer for rdma_controller object.
    typedef std::shared_ptr<rdma_controller> rdma_controller_ptr;

}}}}

#endif
