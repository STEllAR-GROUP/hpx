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
        typedef std::pair<uint32_t, verbs_endpoint_ptr> ClientMapPair;

        typedef hpx::lcos::local::spinlock mutex_type;
        typedef hpx::parcelset::policies::verbs::unique_lock<mutex_type> unique_lock;

        rdma_controller(const char *device, const char *interface, int port);

        ~rdma_controller();

        int startup();

        bool isTerminated() {
            return (clients_.size() == 0);
        }

        typedef std::function<int(struct sockaddr_in *, struct sockaddr_in *)>
        ConnectRequestFunction;
        typedef std::function<
            int(std::pair<uint32_t, uint64_t>, verbs_endpoint_ptr)>
        ConnectionFunction;
        typedef std::function<int(verbs_endpoint_ptr client)>
        DisconnectionFunction;

        // Set a callback which will be called immediately after
        // RDMA_CM_EVENT_CONNECT_REQUEST has been received. This can be useful to
        // prevent two connections being established at the same time.
        void setConnectRequestFunction(ConnectRequestFunction f) {
            this->request_connect_function_ = f;
        }

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

        int eventMonitor(int Nevents);
        int pollCompletionQueues();

        verbs_completion_queue *get_shared_receive_queue() {
            return nullptr;
        }
        verbs_endpoint *get_client_from_completion(
            struct ibv_wc &completion) {
            return nullptr;
        }

        // Listener for RDMA connections.
        verbs_endpoint_ptr getServer() {
            return this->server_endpoint_;
        }

        verbs_protection_domain_ptr getProtectionDomain() {
            return this->protection_domain_;
        }

        verbs_endpoint* getClient(uint32_t qp) {
            return clients_[qp].get();
        }

        rdma_memory_pool_ptr getMemoryPool() {
            return memory_pool_;
        }

        template<typename Function>
        void for_each_client(Function lambda)
        {
            std::for_each(clients_.begin(), clients_.end(), lambda);
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
            return clients_.size();
        }

        void refill_client_receives();

        verbs_endpoint_ptr connect_to_server(uint32_t remote_ip);

        void removeServerToServerConnection(verbs_endpoint_ptr client);

        void removeAllInitiatedConnections();
    private:

        int handle_event(struct rdma_cm_event *cm_event);

        bool completionChannelHandler(uint64_t requestId);

        // store info about local device
        std::string device_;
        std::string interface_;
        sockaddr_in local_addr_;

        // callback function used for connection event handling
        CompletionFunction completion_function_;
        ConnectRequestFunction request_connect_function_;
        ConnectionFunction connection_function_;
        DisconnectionFunction disconnection_function_;

        // only allow one thread to handle connect/disconnect events etc
        mutex_type event_channel_mutex_;

        // Server/Listener for RDMA connections.
        verbs_endpoint_ptr server_endpoint_;

        // Pinned memory pool used for allocating buffers
        rdma_memory_pool_ptr memory_pool_;

        // Protection domain for all resources.
        verbs_protection_domain_ptr protection_domain_;

        // Map of all active clients indexed by queue pair number.
        hpx::concurrent::unordered_map<uint32_t, verbs_endpoint_ptr> clients_;
        typedef hpx::concurrent::unordered_map<uint32_t, verbs_endpoint_ptr>
        ::map_read_lock_type map_read_lock_type;

        // used to skip polling event channel too frequently
        std::chrono::time_point<std::chrono::system_clock> event_check_time_;

        // Completion channel for all completion queues.
        //    RdmaCompletionChannelPtr completion_channel_;
    };

    // Smart pointer for rdma_controller object.
    typedef std::shared_ptr<rdma_controller> rdma_controller_ptr;

}}}}

#endif
