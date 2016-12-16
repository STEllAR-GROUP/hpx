//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_RDMA_CONTROLLER_HPP
#define HPX_PARCELSET_POLICIES_VERBS_RDMA_CONTROLLER_HPP

// config
#include <hpx/config/defines.hpp>

//#define USE_SHARED_RECEIVE_QUEUE
#include <hpx/lcos/local/shared_mutex.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/lcos/future.hpp>
//
#include <plugins/parcelport/verbs/rdma/rdma_logging.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_error.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_locks.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_endpoint.hpp>
//
#include <plugins/parcelport/verbs/unordered_map.hpp>
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

        rdma_controller(const char *device, const char *interface, int port);

        ~rdma_controller();

        int startup();

        bool isTerminated() {
            return (connections_started_.size() == 0);
        }

        typedef std::function<void(uint32_t, verbs_endpoint_ptr)> ConnectionFunction;
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

        int cleanup(void);

        int poll_endpoints();
        int poll_for_completions();
        int poll_for_events();

        verbs_completion_queue *get_shared_receive_queue() {
            return nullptr;
        }
        verbs_endpoint *get_client_from_completion(
            struct ibv_wc &completion) {
            return nullptr;
        }

        verbs_protection_domain_ptr get_protection_domain() {
            return this->protection_domain_;
        }

        verbs_endpoint* get_endpoint(uint32_t remote_ip) {
            return (std::get<0>(connections_started_.find(remote_ip)->second)).get();

        }

        rdma_memory_pool_ptr get_memory_pool() {
            return memory_pool_;
        }

        //    RdmaCompletionChannelPtr GetCompletionChannel() {
            //    return this->_completionChannel;
            //}

        typedef std::function<
            int(struct ibv_wc completion, verbs_endpoint *client)>
        CompletionFunction;
        void setCompletionFunction(CompletionFunction f) {
            this->completion_function_ = f;
        }

        int num_clients() {
            return connections_started_.size();
        }

        void refill_client_receives();

        hpx::shared_future<verbs_endpoint_ptr> connect_to_server(uint32_t remote_ip);

        void disconnect_from_server(verbs_endpoint_ptr client);

        void disconnect_all();

        bool active();

    private:

        int handle_event(struct rdma_cm_event *cm_event, verbs_endpoint *client);

        int handle_connect_request(
            struct rdma_cm_event *cm_event, std::uint32_t remote_ip);

        int start_server_connection(uint32_t remote_ip);

        bool completionChannelHandler(uint64_t requestId);

        bool connection_present(uint32_t dest_ip);

        // store info about local device
        std::string           device_;
        std::string           interface_;
        sockaddr_in           local_addr_;

        // callback function for handling a completion event
        CompletionFunction    completion_function_;

        // callback functions used for connection event handling
        ConnectionFunction    connection_function_;
        DisconnectionFunction disconnection_function_;

        // Server/Listener for RDMA connections.
        verbs_endpoint_ptr    server_endpoint_;

        // Pinned memory pool used for allocating buffers
        rdma_memory_pool_ptr  memory_pool_;

        // Protection domain for all resources.
        verbs_protection_domain_ptr protection_domain_;

        // only allow one thread to handle connect/disconnect events etc
        mutex_type            controller_mutex_;

        typedef std::tuple<
            verbs_endpoint_ptr,
            hpx::promise<verbs_endpoint_ptr>,
            hpx::shared_future<verbs_endpoint_ptr>
        > promise_tuple_type;
        //
        typedef std::pair<const uint32_t, promise_tuple_type> ClientMapPair;

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
        hpx::concurrent::unordered_map<uint32_t, verbs_endpoint_ptr> connections_aborted_;

        // used to skip polling event channel too frequently
        typedef std::chrono::time_point<std::chrono::system_clock> time_type;
        time_type event_check_time_;

        // Completion channel for all completion queues.
        //    RdmaCompletionChannelPtr completion_channel_;
    };

    // Smart pointer for rdma_controller object.
    typedef std::shared_ptr<rdma_controller> rdma_controller_ptr;

}}}}

#endif
