//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/parcelport_verbs_defines.hpp>
//
#include <plugins/parcelport/verbs/rdma/rdma_error.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_event_channel.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_device.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_controller.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_completion_queue.hpp>
#include <plugins/parcelport/verbs/readers_writers_mutex.hpp>
#include <plugins/parcelport/verbs/rdma/verbs_device.hpp>
//
#include <boost/lexical_cast.hpp>
//
#include <poll.h>
#include <errno.h>
#include <iomanip>
#include <sstream>
#include <queue>
#include <stdio.h>
#include <thread>
#include <fstream>

// network stuff, only need inet_ntoa
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

const int hpx::parcelset::policies::verbs::verbs_completion_queue::MaxQueueSize;


using namespace hpx::parcelset::policies::verbs;

/*---------------------------------------------------------------------------*/
rdma_controller::rdma_controller(const char *device, const char *interface, int port)
{
    device_    = device;
    interface_ = interface;
    //
    local_addr_.sin_family      = AF_INET;
    local_addr_.sin_port        = port;
    local_addr_.sin_addr.s_addr = 0xFFFFFFFF;
}

/*---------------------------------------------------------------------------*/
rdma_controller::~rdma_controller()
{
    LOG_DEBUG_MSG("rdma_controller destructor clearing clients");
    clients_.clear();
    LOG_DEBUG_MSG("rdma_controller destructor closing server");
    this->server_endpoint_.reset();
    LOG_DEBUG_MSG("rdma_controller destructor freeing memory pool");
    this->memory_pool_.reset();
    LOG_DEBUG_MSG("rdma_controller destructor releasing protection domain");
    this->protection_domain_.reset();
//    LOG_DEBUG_MSG("rdma_controller destructor deleting completion channel");
//    this->_completionChannel.reset();
    LOG_DEBUG_MSG("rdma_controller destructor done");
}

/*---------------------------------------------------------------------------*/
int rdma_controller::startup()
{
    // Find the address of the I/O link device.
    verbs_device_ptr linkDevice;
    try {
        LOG_DEVEL_MSG("creating InfiniBand device for " << device_
            << " using interface " << interface_);
        linkDevice = hpx::parcelset::policies::verbs::verbs_device_ptr(new hpx::parcelset::policies::verbs::verbs_device(device_, interface_));
    } catch (rdma_error& e) {
        LOG_ERROR_MSG("error opening InfiniBand device: " << e.what());
        return e.error_code();
    }

    LOG_DEVEL_MSG(
        "created InfiniBand device for " << linkDevice->get_device_name()
        << " using interface " << linkDevice->get_interface_name());

    local_addr_.sin_addr.s_addr = linkDevice->get_address();
    LOG_DEVEL_MSG("Device returns IP address " << sockaddress(&local_addr_));

    // Create server/listener for RDMA connections.
    try {
        //
        server_endpoint_ = hpx::parcelset::policies::verbs::verbs_endpoint_ptr(
            new hpx::parcelset::policies::verbs::verbs_endpoint(local_addr_));

        if (server_endpoint_->get_local_port() != local_addr_.sin_port) {
            local_addr_.sin_port = server_endpoint_->get_local_port();
            LOG_DEBUG_MSG("verbs_endpoint port changed to " << std::dec << decnumber(local_addr_.sin_port));
        }
    } catch (rdma_error& e) {
        LOG_ERROR_MSG("error creating listening RDMA connection: " << e.what());
        return e.error_code();
    }

    LOG_DEBUG_MSG(
        "created listening RDMA connection on port " << decnumber(local_addr_.sin_port)
        << " IP address " << sockaddress(&local_addr_));

    // Create a protection domain object.
    try {
        protection_domain_ = verbs_protection_domain_ptr(
            new verbs_protection_domain(server_endpoint_->get_device_context()));
    } catch (rdma_error& e) {
        LOG_ERROR_MSG("error allocating protection domain: " << e.what());
        return e.error_code();
    }
    LOG_DEBUG_MSG("created protection domain " << protection_domain_->getHandle());

    // Create a completion channel object.
    try {
//        _completionChannel = RdmaCompletionChannelPtr(
//            new RdmaCompletionChannel(server_endpoint_->get_device_context(), false));
    } catch (rdma_error& e) {
        LOG_ERROR_MSG("error constructing completion channel: " << e.what());
        return e.error_code();
    }

    // Create a memory pool for pinned buffers
    memory_pool_ = std::make_shared < rdma_memory_pool > (protection_domain_);

#ifdef USE_SHARED_RECEIVE_QUEUE
    // create a shared receive queue
    LOG_DEBUG_MSG("Creating SRQ shared receive queue ");
    server_endpoint_->create_srq(protection_domain_);
#endif

    // Listen for connections.
    LOG_DEVEL_MSG("Calling LISTEN function on "
        << ipaddress(local_addr_.sin_addr.s_addr));
    int err = server_endpoint_->listen(256);
    if (err != 0) {
        LOG_ERROR_MSG(
            "error listening for new RDMA connections: " << rdma_error::error_string(err));
        return err;
    }
    LOG_DEVEL_MSG("LISTEN enabled for new RDMA connections on "
        << ipaddress(local_addr_.sin_addr.s_addr));

    return 0;
}
/*---------------------------------------------------------------------------*/
int rdma_controller::cleanup(void) {
    return 0;
}

/*---------------------------------------------------------------------------*/
void rdma_controller::refill_client_receives() {
    // make sure all clients have a pre-posted receive in their queues
    map_read_lock_type read_lock(clients_.read_write_mutex());
    //
    std::for_each(clients_.begin(), clients_.end(),
        [](rdma_controller::ClientMapPair _client) {
        _client.second->refill_preposts(HPX_PARCELPORT_VERBS_MAX_PREPOSTS);
    });
}

/*---------------------------------------------------------------------------*/
int rdma_controller::pollCompletionQueues()
{
    int ntot = 0, nc = 0;
    //
    map_read_lock_type read_lock(clients_.read_write_mutex());
    //
    if (this->get_shared_receive_queue() == nullptr) {
        for (auto _client : clients_) {
            verbs_endpoint *client = _client.second.get();
            verbs_completion_queue *completionQ = client->getCompletionQ().get();

            // Remove work completions from the completion queue until it is empty.
            do {
                struct ibv_wc completion;
                nc = completionQ->poll_completion(&completion);
                if (nc > 0) {
                    if (completion.status != IBV_WC_SUCCESS) {
                        LOG_ERROR_MSG(
                            "Message failed current receive count is " << client->get_receive_count());
                        LOG_DEBUG_MSG("pollCompletionQueues - removing wr_id " << hexpointer(completion.wr_id) << " "
                            << verbs_completion_queue::wc_opcode_str(completion.opcode));
                        std::terminate();
                    }
                    if (this->completion_function_) {
                        this->completion_function_(completion, client);
                    }
                    else {
                        LOG_ERROR_MSG("No completion function set");
                        std::terminate();
                    }
                }
                ntot += nc;
            } while (nc > 0);
        }
    }
    else {
        verbs_completion_queue *completionQ = get_shared_receive_queue();

        // Remove work completions from the completion queue until it is empty.
        do {
            struct ibv_wc completion;
            nc = completionQ->poll_completion(&completion);
            if (nc > 0) {
                if (completion.status != IBV_WC_SUCCESS) {
                    LOG_DEBUG_MSG("pollCompletionQueues - removing wr_id " << hexpointer(completion.wr_id) << " "
                        << verbs_completion_queue::wc_opcode_str(completion.opcode));
                    std::terminate();
                }
                verbs_endpoint *client = get_client_from_completion(completion);
                if (this->completion_function_) {
                    this->completion_function_(completion, client);
                }
            }
            ntot += nc;
        } while (nc > 0);

    }
    return ntot;
}

/*---------------------------------------------------------------------------*/
int rdma_controller::eventMonitor(int Nevents)
{
    // completions of work requests
    int completions_handled = pollCompletionQueues();

    // no need to check for events very often
    using namespace std::chrono;
    time_point<system_clock> now = system_clock::now();
    if (duration_cast<microseconds>(now-event_check_time_).count()>100)
    {
        event_check_time_ = now;
        // scoped lock around event channel handler as it makes things easier if
        // one one thread handles connection/disconnection events etc
        unique_lock lock(event_channel_mutex_, std::try_to_lock);
        if (lock.owns_lock()) {
            completions_handled += server_endpoint_->poll_for_event(
                [this](struct rdma_cm_event *cm_event) {
                    return handle_event(cm_event);
                }
            );
        }
    }
    return completions_handled;
}

/*---------------------------------------------------------------------------*/
int rdma_controller::handle_event(struct rdma_cm_event *cm_event)
{
    // Debugging code to get ip address of soure/dest of event
    // NB: The src and dest fields refer to the message and not the connect request
    // so we are actually receiving a request from dest (it is the src of the msg)
    //
    struct sockaddr *ip_src = &cm_event->id->route.addr.src_addr;
    struct sockaddr_in *addr_src = reinterpret_cast<struct sockaddr_in *>(ip_src);
    //
    struct sockaddr *ip_dst = &cm_event->id->route.addr.dst_addr;
    struct sockaddr_in *addr_dst = reinterpret_cast<struct sockaddr_in *>(ip_dst);

    // Handle the event : NB ack_event will delete the event, do not use it afterwards.
    switch (cm_event->event) {

    case RDMA_CM_EVENT_CONNECT_REQUEST: {
        LOG_DEVEL_MSG("RDMA_CM_EVENT_CONNECT_REQUEST     "
            << sockaddress(ip_dst) << "to "
            << sockaddress(ip_src)
            << "( " << sockaddress(&local_addr_) << ")");
        //
        // prevent two connections from taking place between the same endpoints
        //
        if (this->request_connect_function_) {
            LOG_DEVEL_MSG("Connection request, callback from "
                << sockaddress(ip_dst) << "to "
                << sockaddress(ip_src)
                << "( " << sockaddress(&local_addr_) << ")");
            //
            if (!this->request_connect_function_(addr_dst, addr_src)) {
                LOG_ERROR_MSG("Connect request callback rejected");
                // reject() does not wait or ack
                server_endpoint_->reject(cm_event->id);
                LOG_DEVEL_MSG("Rejected connection request, from "
                    << sockaddress(ip_dst) << "to "
                    << sockaddress(ip_src)
                    << "( " << sockaddress(&local_addr_) << ")");
                break;
            }
        }

        // Construct a verbs_completion_queue object for the new client.
        verbs_completion_queue_ptr completionQ;
        try {
            //completionQ = std::make_shared<verbs_completion_queue>
            //    (cm_event->id->verbs, verbs_completion_queue::MaxQueueSize,
            //    _completionChannel->getChannel());
            completionQ = std::make_shared < verbs_completion_queue >
                (cm_event->id->verbs, verbs_completion_queue::MaxQueueSize,
                (ibv_comp_channel*) NULL);
        } catch (rdma_error& e) {
            LOG_ERROR_MSG("error creating completion queue: " << e.what());
            break;
        }

        // Construct a new verbs_endpoint object for the new client.
        verbs_endpoint_ptr client;
        try {
            client = std::make_shared < verbs_endpoint >
            (cm_event->id, protection_domain_, completionQ,
                memory_pool_, server_endpoint_->SRQ());
        } catch (rdma_error& e) {
            LOG_ERROR_MSG("error creating rdma client: %s\n" << e.what());
            completionQ.reset();
            break;
        }

        LOG_DEBUG_MSG("adding a new client with qpnum "
            << decnumber(client->get_qp_num()));

        // make sure client has preposted receives
        // @TODO, when we use a shared receive queue, fix this
        client->refill_preposts(HPX_PARCELPORT_VERBS_MAX_PREPOSTS);

        // Add new client to map of active clients.
        clients_.insert(
            std::pair<uint32_t, verbs_endpoint_ptr>(client->get_qp_num(), client));

        // Add completion queue to completion channel.
        //    _completionChannel->addCompletionQ(completionQ);

        // Accept the connection from the new client.
        // accept() does not wait or ack
        if (client->accept() != 0) {
            LOG_ERROR_MSG("error accepting client connection: %s "
                << rdma_error::error_string(errno));
            clients_.erase(client->get_qp_num());
            //      _completionChannel->removeCompletionQ(completionQ);
            std::terminate();
            //            client->reject();
            client.reset();
            completionQ.reset();
            break;
        }
        LOG_DEBUG_MSG("accepted connection from "
            << sockaddress(client->get_remote_address()));
        break;
    }

    case RDMA_CM_EVENT_ESTABLISHED: {
        LOG_DEVEL_MSG("RDMA_CM_EVENT_ESTABLISHED         "
            << ipaddress(addr_dst->sin_addr.s_addr) << "to "
            << ipaddress(addr_src->sin_addr.s_addr)
            << "( " << ipaddress(local_addr_.sin_addr.s_addr) << ")");

        // Find connection associated with this event.
        verbs_endpoint_ptr client = clients_[cm_event->id->qp->qp_num];
        LOG_INFO_MSG("connection established with "
            << sockaddress(client->get_remote_address()));
        if (this->connection_function_) {
            LOG_DEVEL_MSG("calling connection callback from  "
                << ipaddress(addr_dst->sin_addr.s_addr) << "to "
                << ipaddress(addr_src->sin_addr.s_addr)
                << "( " << ipaddress(local_addr_.sin_addr.s_addr) << ")");
            this->connection_function_(
                std::make_pair(cm_event->id->qp->qp_num, 0), client);
        }
        break;
    }

    case RDMA_CM_EVENT_DISCONNECTED: {
        LOG_DEVEL_MSG("RDMA_CM_EVENT_DISCONNECTED        "
            << ipaddress(addr_dst->sin_addr.s_addr) << "to "
            << ipaddress(addr_src->sin_addr.s_addr)
            << "( " << ipaddress(local_addr_.sin_addr.s_addr) << ")");
        // Find connection associated with this event.
        verbs_endpoint_ptr client = clients_[cm_event->id->qp->qp_num];
        verbs_completion_queue_ptr completionQ = client->getCompletionQ();

        // Complete disconnect initiated by peer.
        int err = client->disconnect(false);
        if (err == 0) {
            LOG_INFO_MSG("disconnected from "
                << sockaddress(client->get_remote_address()));
        } else {
            LOG_ERROR_MSG(
                "error disconnecting from peer: " << rdma_error::error_string(err));
        }

        // ack the event (before removing the rdma cm id).
        verbs_event_channel::ack_event(cm_event);

        // Remove connection from map of active connections.
        clients_.erase(cm_event->id->qp->qp_num);

        // Destroy connection object.
        LOG_DEBUG_MSG("destroying RDMA connection to client "
            << sockaddress(client->get_remote_address()));
        client.reset();

        // Remove completion queue from the completion channel.
        //    _completionChannel->removeCompletionQ(completionQ);

        // Destroy the completion queue.
        LOG_DEBUG_MSG("destroying completion queue " << completionQ->getHandle());
        completionQ.reset();

        // even already ack'ed - do not do it again, just return
        return 0;
    }

    default: {
        LOG_ERROR_MSG(
            "RDMA event: " << rdma_event_str(cm_event->event) << " is not supported");
        break;
    }
    }

    // Acknowledge the event. This is always necessary because it tells
    // rdma_cm that it can delete the structure it allocated for the event data
    return verbs_event_channel::ack_event(cm_event);
}
/*
// ---------------------------------------------------------------------------
bool rdma_controller::completionChannelHandler(uint64_t requestId) { //, lock_type2 &&lock) {
    verbs_endpoint *client;
    try {
        // Get the notification event from the completion channel.
        verbs_completion_queue *completionQ = _completionChannel->getEvent();

        // Remove work completions from the completion queue until it is empty.
        while (completionQ->removeCompletions() != 0) {
            // Get the next work completion.
            struct ibv_wc *completion;
            // the completion queue isn't yet thread safe, so only allow one thread at a time to pop a completion
            {
                completion = completionQ->popCompletion();
                LOG_DEBUG_MSG("Controller completion - removing wr_id " << hexpointer(completion->wr_id));
                // Find the connection that received the message.
                client = clients_[completion->qp_num].get();
            }
            if (this->completion_function_) {
                //                this->completion_function_(completion, client);
            }
        }
    }

    catch (const rdma_error& e) {
        LOG_ERROR_MSG(
            "error removing work completions from completion queue: " << rdma_error::error_string(e.error_code()));
    }

    return true;
}
*/
/*
 struct sockaddr_in {
 short            sin_family;   // e.g. AF_INET
 unsigned short   sin_port;     // e.g. htons(3490)
 struct in_addr   sin_addr;     // see struct in_addr, below
 char             sin_zero[8];  // zero this if you want to
 };

 struct in_addr {
 unsigned long s_addr;  // load with inet_aton()
 };
 */

/*---------------------------------------------------------------------------*/
// return the client
verbs_endpoint_ptr rdma_controller::connect_to_server(
    uint32_t remote_ip)
{
    sockaddr_in remote_addr;
    sockaddr_in local_addr;
    //
    remote_addr.sin_family      = AF_INET;
    remote_addr.sin_port        = local_addr_.sin_port;
    remote_addr.sin_addr.s_addr = remote_ip;
    local_addr.sin_port         = 0;
    local_addr                  = local_addr_;

    LOG_DEVEL_MSG("connect_to_server from "
        << ipaddress(local_addr_.sin_addr.s_addr)
        << "to " << ipaddress(remote_ip)
        << "( " << ipaddress(local_addr_.sin_addr.s_addr) << ")");

    verbs_completion_queue_ptr completionQ;
    try {
        //    completionQ = std::make_shared < verbs_completion_queue >
        //      (server_endpoint_->get_device_context(), verbs_completion_queue::MaxQueueSize, _completionChannel->getChannel());
        completionQ = std::make_shared < verbs_completion_queue >
            (server_endpoint_->get_device_context(), verbs_completion_queue::MaxQueueSize,
            (ibv_comp_channel*) NULL);
    } catch (rdma_error& e) {
        LOG_ERROR_MSG("error creating completion queue: " << e.what());
    }

    verbs_endpoint_ptr newClient;
    try {
        newClient = std::make_shared < verbs_endpoint >
        (local_addr, remote_addr, protection_domain_, completionQ, memory_pool_);
    } catch (rdma_error& e) {
        LOG_ERROR_MSG("error creating rdma client: %s\n" << e.what());
        completionQ.reset();
        return NULL;
    }

    // make a connection
    LOG_DEVEL_MSG("Calling connect on new endpoint");

    auto polling_function = [this]()
        {
            server_endpoint_->poll_for_event(
                [this](struct rdma_cm_event *cm_event) {
                    LOG_DEVEL_MSG("Got an event in lambda poll, calling handler");
                    int temp = handle_event(cm_event);
                    LOG_DEVEL_MSG("Completed lambda poll");
                    return temp;
                });
        };

    if (newClient->connect(polling_function) == -1)
    {
        LOG_DEBUG_MSG("connect failed in connect_to_server from "
            << ipaddress(local_addr_.sin_addr.s_addr)
            << "to " << ipaddress(remote_ip));
        // throw away our client endpoint
        newClient.reset();
        LOG_ERROR_MSG("Reset completion Q");
        completionQ.reset();
        LOG_ERROR_MSG("returning NULL ");
        return NULL;
    }

    // make sure client has preposted receives
    // @TODO, when we use a shared receive queue, fix this
    newClient->refill_preposts(HPX_PARCELPORT_VERBS_MAX_PREPOSTS);

    LOG_DEVEL_MSG("Inserting QP " << decnumber(newClient->get_qp_num())
        << "in clients map from "
        << ipaddress(local_addr_.sin_addr.s_addr)
        << "to " << ipaddress(remote_ip)
        << "( " << ipaddress(local_addr_.sin_addr.s_addr) << ")");

    // Add new client to map of active clients.
    clients_.insert(
        std::pair<uint32_t, verbs_endpoint_ptr>(newClient->get_qp_num(), newClient));

    // Add completion queue to completion channel.
    //  _completionChannel->addCompletionQ(completionQ);

//    this->refill_client_receives();

    LOG_DEBUG_MSG("Added a server-server client with qpnum "
        << decnumber(newClient->get_qp_num()));
    return newClient;
}

/*---------------------------------------------------------------------------*/
void rdma_controller::removeServerToServerConnection(verbs_endpoint_ptr client)
{
    LOG_DEBUG_MSG("Removing Server-Server client object");
    // Find connection associated with this event.
    verbs_completion_queue_ptr completionQ = client->getCompletionQ();
    uint32_t qp = client->get_qp_num();

    // disconnect initiated by us
    int err = client->disconnect(true);
    if (err == 0) {
        LOG_INFO_MSG("disconnected from "
            << sockaddress(client->get_remote_address()));
    } else {
        LOG_ERROR_MSG("error disconnecting from peer: " << rdma_error::error_string(err));
    }

    // Remove connection from map of active connections.
    clients_.erase(qp);

    // Destroy connection object.
    LOG_DEBUG_MSG("destroying RDMA connection to client "
        << sockaddress(client->get_remote_address()));
    client.reset();

    // Remove completion queue from the completion channel.
    //    _completionChannel->removeCompletionQ(completionQ);

    // Destroy the completion queue.
    LOG_DEBUG_MSG("destroying completion queue " << completionQ->getHandle());
    completionQ.reset();
}

/*---------------------------------------------------------------------------*/
void rdma_controller::removeAllInitiatedConnections()
{
    while (std::count_if(clients_.begin(), clients_.end(),
        [](const std::pair<uint32_t, verbs_endpoint_ptr> & c) {
        return c.second->getInitiatedConnection();
    }) > 0)
    {
        hpx::concurrent::unordered_map<uint32_t, verbs_endpoint_ptr>::iterator c =
            clients_.begin();
        while (c != clients_.end()) {
            if (c->second->getInitiatedConnection()) {
                LOG_DEBUG_MSG("Removing a connection");
                removeServerToServerConnection(c->second);
                break;
            }
        }
    }
}
