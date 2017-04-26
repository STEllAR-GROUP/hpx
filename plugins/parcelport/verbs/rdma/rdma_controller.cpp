//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)
#include <hpx/config/parcelport_verbs_defines.hpp>
//
#include <plugins/parcelport/verbs/rdma/rdma_error.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_locks.hpp>
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
#include <memory>
#include <utility>
#include <cstdint>
#include <cstring>
//
#include <netinet/in.h>

const int hpx::parcelset::policies::verbs::verbs_completion_queue::MaxQueueSize;

using namespace hpx::parcelset::policies::verbs;

//----------------------------------------------------------------------------
rdma_controller::rdma_controller(const char *device, const char *interface, int port)
{
    device_    = device;
    interface_ = interface;
    //
    local_addr_.sin_family      = AF_INET;
    local_addr_.sin_port        = port;
    local_addr_.sin_addr.s_addr = 0xFFFFFFFF;
    //
    event_pause_ = 0;
}

//----------------------------------------------------------------------------
rdma_controller::~rdma_controller()
{
    //
    if (memory_pool_ && server_endpoint_)
    {
        memory_pool_->small_.decrement_used_count(
            server_endpoint_->get_receive_count()
        );
    }
    //
    LOG_DEVEL_MSG("rdma_controller destructor clearing clients");
    connections_started_.clear();
    LOG_DEVEL_MSG("rdma_controller destructor closing server");
    this->server_endpoint_.reset();
    LOG_DEVEL_MSG("rdma_controller destructor freeing memory pool");
    this->memory_pool_.reset();
    LOG_DEVEL_MSG("rdma_controller destructor releasing protection domain");
    this->protection_domain_.reset();
    LOG_DEVEL_MSG("rdma_controller destructor deleting completion queue");
    this->completion_queue_.reset();
    LOG_DEVEL_MSG("rdma_controller destructor done");
}

//----------------------------------------------------------------------------
int rdma_controller::startup()
{
    LOG_DEVEL_MSG("creating InfiniBand device for " << device_
        << " using interface " << interface_);

    // Find the address of the Infiniband link device.
    verbs_device linkDevice(device_, interface_);

    LOG_DEVEL_MSG(
        "created InfiniBand device for " << linkDevice.get_device_name()
        << " using interface " << linkDevice.get_interface_name());

    local_addr_.sin_addr.s_addr = linkDevice.get_address();
    LOG_DEVEL_MSG("Device returns IP address " << sockaddress(&local_addr_));

    // Create server/listener for RDMA connections.
    try {
        //
        server_endpoint_ = std::make_shared<verbs_endpoint>(local_addr_);
        //
        if (server_endpoint_->get_local_port() != local_addr_.sin_port)
        {
            local_addr_.sin_port = server_endpoint_->get_local_port();
            LOG_DEVEL_MSG("verbs_endpoint port changed to "
                << decnumber(local_addr_.sin_port));
        }
    } catch (rdma_error& e) {
        LOG_ERROR_MSG("error creating listening RDMA connection: " << e.what());
        return e.error_code();
    }

    LOG_DEVEL_MSG(
        "created listening RDMA connection " << hexpointer(server_endpoint_.get())
        << " on port " << decnumber(local_addr_.sin_port)
        << " IP address " << sockaddress(&local_addr_));

    // Create a protection domain object.
    try {
        protection_domain_ = verbs_protection_domain_ptr(
            new verbs_protection_domain(server_endpoint_->get_device_context()));
    } catch (rdma_error& e) {
        LOG_ERROR_MSG("error allocating protection domain: " << e.what());
        return e.error_code();
    }
    LOG_DEVEL_MSG("created protection domain " << protection_domain_->get_handle());

    // Create a memory pool for pinned buffers
    memory_pool_ = std::make_shared<rdma_memory_pool> (protection_domain_);

    // Construct a completion queue object that will be shared by all endpoints
    completion_queue_ = std::make_shared<verbs_completion_queue>(
        server_endpoint_->get_device_context(),
        verbs_completion_queue::MaxQueueSize, (ibv_comp_channel*) nullptr);

    // create a shared receive queue
    LOG_DEVEL_MSG("Creating SRQ shared receive queue ");
    server_endpoint_->create_srq(protection_domain_);
    LOG_DEVEL_MSG("SRQ is " << hexpointer(server_endpoint_->getsrq()));
    // preposts are made via the server endpoint when using SRQ, so make sure
    // the memory pool is setup correctly
    server_endpoint_->set_memory_pool(memory_pool_);
    server_endpoint_->refill_preposts(HPX_PARCELPORT_VERBS_MAX_PREPOSTS, true);

    // Listen for connections.
    LOG_DEVEL_MSG("Calling LISTEN function on "
        << sockaddress(&local_addr_));
    int err = server_endpoint_->listen(256);
    if (err != 0) {
        LOG_ERROR_MSG(
            "error listening for new RDMA connections: "
            << rdma_error::error_string(err));
        return err;
    }
    LOG_DEVEL_MSG("LISTEN enabled for new RDMA connections on "
        << sockaddress(&local_addr_));

    return 0;
}

//----------------------------------------------------------------------------
void rdma_controller::refill_client_receives(bool force)
{
    // a copy of the shared receive queue is held by the server_endpoint
    // so pre-post receives to that to ensure all clients are 'ready'
    LOG_DEVEL_MSG("refill_client_receives");
    server_endpoint_->refill_preposts(HPX_PARCELPORT_VERBS_MAX_PREPOSTS, force);
}

//----------------------------------------------------------------------------
int rdma_controller::poll_endpoints(bool stopped)
{
    // completions of work requests
    int handled = poll_for_work_completions(stopped);

    // no need to check for connection events very often, use a backoff so that
    // when an event is received, we check frequently, when not, we gradually slow
    // down our checks to avoid wasting too much time
    using namespace std::chrono;
    time_point<system_clock> now = system_clock::now();
    if (duration_cast<microseconds>(now - event_check_time_).count() > event_pause_)
    {
        event_check_time_ = now;
        // only active when logging is enabled
        LOG_TIMED_INIT(event_poll);
        LOG_TIMED_BLOCK(event_poll, DEVEL, 5.0,
            {
                LOG_DEVEL_MSG("Polling event channel");
                debug_connections();
            }
        )
        int events = server_endpoint_->poll_for_event(
            [this](struct rdma_cm_event *cm_event) {
            return handle_event(cm_event, server_endpoint_.get());
        }
        );
        if (events>0) {
            event_pause_ = 0;
        }
        else {
            event_pause_ = (event_pause_<500) ? event_pause_ + 10 : 500;
        }
        handled += events;
    }
    return handled;
}

//----------------------------------------------------------------------------
int rdma_controller::poll_for_work_completions(bool stopped)
{
    LOG_TIMED_INIT(completion_poll);
    LOG_TIMED_BLOCK(completion_poll, DEVEL, 5.0,
        {
            LOG_DEVEL_MSG("Polling completion_poll channel");
        }
    )

    struct ibv_wc completion;
    int ntot = 0, nc = 0;
    //
    verbs_completion_queue *completionQ = get_completion_queue();

    // Remove work completions from the completion queue until it is empty.
    do {
        nc = completionQ->poll_completion(&completion);
        // positive result means completion ok
        if (nc > 0 && !stopped) {
            verbs_endpoint *client = get_client_from_completion(completion);
            // handle the completion
            this->completion_function_(completion, client);
            ++ntot;
        }
        // negative result indicates flushed receive
        else if (nc < 0) {
            // flushed receive completion, delete it, disconnection has started
            verbs_memory_region *region = (verbs_memory_region *)completion.wr_id;
            // let go of this region
            memory_pool_->deallocate(region);
            LOG_DEVEL_MSG("Flushed receive on qp " << decnumber(completion.qp_num));
        }
        if (nc != 0 && completion.opcode==IBV_WC_RECV) {
            // bookkeeping : decrement counter that keeps preposted queue full
            server_endpoint_->pop_receive_count();
            if (server_endpoint_->get_receive_count() <
                HPX_PARCELPORT_VERBS_MAX_PREPOSTS/2)
            {
                LOG_DEVEL_MSG("refilling preposts");
                server_endpoint_->refill_preposts(
                    HPX_PARCELPORT_VERBS_MAX_PREPOSTS, false);
            }
        }
    } while (nc != 0);
    //
    return ntot;
}

//----------------------------------------------------------------------------
void rdma_controller::debug_connections()
{
    map_read_lock_type read_lock(connections_started_.read_write_mutex());
    //
    LOG_DEVEL_MSG("qp_endpoint_map_ entries");
    std::for_each(qp_endpoint_map_.begin(), qp_endpoint_map_.end(),
        [this](const rdma_controller::QPMapPair &_client) {
            verbs_endpoint_ptr endpoint = _client.second;
            if (endpoint->is_client_endpoint()) {
                LOG_DEVEL_MSG("Status of connection         from "
                    << sockaddress(&local_addr_) << "to "
                    << sockaddress(endpoint->get_remote_address())
                    << "client " << decnumber(endpoint->get_qp_num())
                    << " state " << verbs_endpoint::ToString(endpoint->get_state()));
            }
            else {
                LOG_DEVEL_MSG("Status of connection         from "
                    << sockaddress(endpoint->get_remote_address()) << "to "
                    << sockaddress(&local_addr_)
                    << "server " << decnumber(endpoint->get_qp_num())
                    << " state " << verbs_endpoint::ToString(endpoint->get_state()));
            }
        }
    );
    LOG_DEVEL_MSG("connections_started_ entries");
    std::for_each(connections_started_.begin(), connections_started_.end(),
        [this](const rdma_controller::ClientMapPair &_client) {
            verbs_endpoint_ptr endpoint = std::get<0>(_client.second);
            if (endpoint->is_client_endpoint()) {
                LOG_DEVEL_MSG("Status of connection         from "
                    << sockaddress(&local_addr_) << "to "
                    << sockaddress(endpoint->get_remote_address())
                    << "client " << decnumber(endpoint->get_qp_num())
                    << " state " << verbs_endpoint::ToString(endpoint->get_state()));
            }
            else {
                LOG_DEVEL_MSG("Status of connection         from "
                    << sockaddress(endpoint->get_remote_address()) << "to "
                    << sockaddress(&local_addr_)
                    << "server " << decnumber(endpoint->get_qp_num())
                    << " state " << verbs_endpoint::ToString(endpoint->get_state()));
            }
        }
    );
}
//----------------------------------------------------------------------------
int rdma_controller::handle_event(struct rdma_cm_event *cm_event,
    verbs_endpoint *a_client)
{
    // Get ip address of source/dest of event
    // NB: The src and dest fields refer to the event and not the 'request'
    struct sockaddr *ip_src = &cm_event->id->route.addr.src_addr;
    struct sockaddr *ip_dst = &cm_event->id->route.addr.dst_addr;
    struct sockaddr_in *addr_src = reinterpret_cast<struct sockaddr_in *>(ip_src);
    struct sockaddr_in *addr_dst = reinterpret_cast<struct sockaddr_in *>(ip_dst);

    LOG_DEVEL_MSG("event src is " << sockaddress(addr_src)
        << "( " << sockaddress(&local_addr_) << ")");

    verbs_endpoint_ptr event_client;
    uint32_t qpnum = (cm_event->id->qp) ? cm_event->id->qp->qp_num : 0;
    if (qpnum>0) {
        // Find connection associated with this event if it's not a new request
        LOG_DEVEL_MSG("handle_event : Looking for qp in map " << decnumber(qpnum));
        auto present = qp_endpoint_map_.is_in_map(qpnum);
        if (present.second) {
            event_client = present.first->second;
        }
        else {
            if (cm_event->event == RDMA_CM_EVENT_TIMEWAIT_EXIT) {
                // do nothing
                verbs_event_channel::ack_event(cm_event);
                return 0;
            }
            else {
                LOG_DEVEL_MSG("handle_event : could not find client for "
                    << decnumber(qpnum));
                std::terminate();
            }
        }
    }
    else {
        LOG_DEVEL_MSG("handle_event : qp num is zero");
        auto present = connections_started_.is_in_map(addr_dst->sin_addr.s_addr);
        if (present.second) {
            event_client = std::get<0>(present.first->second);
        }
    }
    if (!event_client) {
        LOG_DEVEL_MSG("handle_event : event client not found");
    }
    //
    struct sockaddr_in *conn_src = reinterpret_cast<struct sockaddr_in *>(ip_src);
    struct sockaddr_in *conn_dst = reinterpret_cast<struct sockaddr_in *>(ip_dst);
    //
    // are we the server or client end of the connection, flip the src/dst
    // pointers if we are the server end (clients init connections to servers)
    if (event_client && !event_client->is_client_endpoint()) {
        conn_src = reinterpret_cast<struct sockaddr_in *>(ip_dst);
        conn_dst = reinterpret_cast<struct sockaddr_in *>(ip_src);
    }

    // Handle the event : NB ack_event will delete the event, do not use it afterwards.
    switch (cm_event->event) {

    // a connect request event will only ever occur on the server_endpoint_
    // in response to a new connection request from a client
    case RDMA_CM_EVENT_CONNECT_REQUEST: {
        LOG_DEVEL_MSG("RDMA_CM_EVENT_CONNECT_REQUEST     "
            << sockaddress(conn_dst) << "to "
            << sockaddress(conn_src)
            << "( " << sockaddress(&local_addr_) << ")");

        // We must not allow an new outgoing connection and a new incoming
        // connect to be started simultaneously - to avoid races on the
        // connection maps
        unique_lock lock(controller_mutex_);
        handle_connect_request(cm_event, conn_dst->sin_addr.s_addr);
        break;
    }

    // this event will be generated after an accept or reject
    // RDMA_CM_EVENT_ESTABLISHED is sent to both ends of a new connection
    case RDMA_CM_EVENT_REJECTED:
    case RDMA_CM_EVENT_ESTABLISHED: {
        // we use addr_dst because it is the remote end of the connection
        // regardless of whether we are connecting to, or being connected from
        uint32_t remote_ip = addr_dst->sin_addr.s_addr;

        LOG_DEVEL_MSG(rdma_event_str(cm_event->event) << "    from "
            << sockaddress(conn_src) << "to "
            << sockaddress(conn_dst)
            << "( " << sockaddress(&local_addr_) << ")");

        // process the established event
        int established = event_client->handle_establish(cm_event);

        // connection established without problem
        if (established==0)
        {
            LOG_DEVEL_MSG("calling connection callback  from "
                << sockaddress(conn_src) << "to "
                << sockaddress(conn_dst)
                << "( " << sockaddress(&local_addr_) << ")");

            // call connection function before making the future ready
            // to avoid a race in the parcelport get connection routines
            this->connection_function_(event_client);

            LOG_DEVEL_MSG("established connection       from "
                << sockaddress(conn_src) << "to "
                << sockaddress(conn_dst)
                << "and making future ready, qp = " << decnumber(qpnum));

            // if there is an entry for a locally started connection on this IP
            // then set the future ready with the verbs endpoint
            auto present = connections_started_.is_in_map(remote_ip);
            if (present.second) {
                std::get<1>(connections_started_.find(remote_ip)->second).
                    set_value(event_client);
                // once the future is set, the entry can be removed
                connections_started_.erase(remote_ip);
            }
        }

        // @TODO remove this aborted event handler once all is working
        // send the event to that
        else if (established==-1)
        {
            std::terminate();
        }

        // the remote end rejected our connection, so we must abort and clean up
        else if (established==-2)
        {
            // we need to delete the started connection and replace it with a new one
            LOG_DEVEL_MSG("Abort old connect, rejected from "
                << sockaddress(addr_src) << "to "
                << sockaddress(addr_dst)
                << "( " << sockaddress(&local_addr_) << ")"
                << "qp = " << decnumber(qpnum));

            // if this was a connection started by remote, remove it from the map
            qp_endpoint_map_.erase(qpnum);
        }
        // event acked by handle_establish
        return 0;
    }

    // this event is only ever received on the client end of a connection
    // after starting to make a connection to a remote server
    case RDMA_CM_EVENT_ADDR_RESOLVED: {
        LOG_DEVEL_MSG("RDMA_CM_EVENT_ADDR_RESOLVED         "
            << sockaddress(conn_src) << "to "
            << sockaddress(conn_dst)
            << "( " << sockaddress(&local_addr_) << ")");

        // When a new connection is started (start_server_connection),
        // this event might be received before the new endpoint has been added to the map.
        // protect with the controller lock
        unique_lock lock(controller_mutex_);
        //
        verbs_endpoint_ptr temp_client =
            std::get<0>(connections_started_.find(conn_dst->sin_addr.s_addr)->second);
        if (temp_client->handle_addr_resolved(cm_event)==-1) {
            std::terminate();
        }
        // event acked by handle_addr_resolved
        return 0;
    }

    // this event is only ever received on the client end of a connection
    // after starting to make a connection to a remote server
    case RDMA_CM_EVENT_ROUTE_RESOLVED: {
        LOG_DEVEL_MSG("RDMA_CM_EVENT_ROUTE_RESOLVED        "
            << sockaddress(conn_src) << "to "
            << sockaddress(conn_dst)
            << "( " << sockaddress(&local_addr_) << ")");

        // we don't need the lock on controller_mutex_ here because we cannot get here
        // until addr_resolved has been completed.
        verbs_endpoint_ptr temp_client =
            std::get<0>(connections_started_.find(conn_dst->sin_addr.s_addr)->second);
        if (temp_client->handle_route_resolved(cm_event)==-1) {
            std::terminate();
        }
        // handle_route_resolved makes the queue-pair valid, add it to qp map
        uint32_t qpnum = temp_client->get_qp_num();
        LOG_DEVEL_MSG("Adding new_client to qp_endpoint_map " << decnumber(qpnum)
            << "in start_server_connection");
        qp_endpoint_map_.insert(std::make_pair(qpnum, temp_client));

        // event acked by handle_route_resolved
        return 0;
    }

    case RDMA_CM_EVENT_DISCONNECTED: {
        LOG_DEVEL_MSG("RDMA_CM_EVENT_DISCONNECTED          "
            << sockaddress(addr_src) << "to "
            << sockaddress(addr_dst)
            << "( " << sockaddress(&local_addr_) << ")");
        //
        if (event_client->handle_disconnect(cm_event)==-1) {
            std::terminate();
        }

        LOG_DEVEL_MSG("Erasing client from qp_endpoint_map "
            << decnumber(event_client->get_qp_num()));
        qp_endpoint_map_.erase(event_client->get_qp_num());

        // get cq before we delete client
//        verbs_completion_queue_ptr completionQ = event_client->get_completion_queue();
//        uint32_t remote_ip = event_client->get_remote_ip_address();

        // event acked by handle_disconnect
        return 0;
    }

    default: {
        LOG_ERROR_MSG(
            "RDMA event: " << rdma_event_str(cm_event->event)
            << " is not supported "
            << " event came with "
            << hexpointer(cm_event->param.conn.private_data) << " , "
            << decnumber((int)(cm_event->param.conn.private_data_len)) << " , "
            << decnumber(cm_event->id->qp->qp_num));

        break;
    }
    }

    // Acknowledge the event. This is always necessary because it tells
    // rdma_cm that it can delete the structure it allocated for the event data
    return verbs_event_channel::ack_event(cm_event);
}

//----------------------------------------------------------------------------
// This function is only ever called from inside the event handler and is therefore
// protected by the controller mutex
int rdma_controller::handle_connect_request(
    struct rdma_cm_event *cm_event, std::uint32_t remote_ip)
{
    auto present = connections_started_.is_in_map(remote_ip);
    if (present.second)
    {
        LOG_DEVEL_MSG("Race connection, check priority   "
            << ipaddress(remote_ip) << "to "
            << sockaddress(&local_addr_)
            << "( " << sockaddress(&local_addr_) << ")");

        // if a connection to this ip address is already being made, and we have
        // a lower ip than the remote end, reject the incoming connection
        if (remote_ip>local_addr_.sin_addr.s_addr &&
             std::get<0>(present.first->second)->get_state() !=
                 verbs_endpoint::connection_state::terminated)
         {
            LOG_DEVEL_MSG("Reject connection , priority from "
                << ipaddress(remote_ip) << "to "
                << sockaddress(&local_addr_)
                << "( " << sockaddress(&local_addr_) << ")");
            //
            server_endpoint_->reject(cm_event->id);
            return 0;
        }
        else {
            // we need to delete the connection we started and replace it with a new one
            LOG_DEVEL_MSG("Priorty to new, Aborting old from "
                << sockaddress(&local_addr_) << "to "
                << ipaddress(remote_ip)
                << "( " << sockaddress(&local_addr_) << ")");

            verbs_endpoint_ptr aborted_client = std::get<0>(present.first->second);
            aborted_client->abort();
        }
    }

    // Construct a new verbs_endpoint object for the new client.
    verbs_endpoint_ptr new_client;
    new_client = std::make_shared<verbs_endpoint>
        (local_addr_, cm_event->id, protection_domain_, completion_queue_,
        memory_pool_, server_endpoint_->SRQ(),
        server_endpoint_->get_event_channel());
    LOG_DEVEL_MSG("Created a new endpoint with pointer "
        << hexpointer(new_client.get())
        << "qp " << decnumber(new_client->get_qp_num()));

    uint32_t qpnum = new_client->get_qp_num();
    LOG_DEVEL_MSG("Adding new_client to qp_endpoint_map " << decnumber(qpnum)
        << "in handle_connect_request");
    qp_endpoint_map_.insert(std::make_pair(qpnum, new_client));

    LOG_DEVEL_MSG("CR.Map<ip <endpoint,promise>>from "
        << ipaddress(remote_ip) << "to "
        << sockaddress(&local_addr_)
        << "( " << sockaddress(&local_addr_) << ")"
        << decnumber(qpnum));

    if (present.second)
    {
        // previous attempt was aborted, reset the endpoint in the connection map
        // use find, because iterator from present.first is const
        std::get<0>(connections_started_.find(remote_ip)->second) = new_client;
    }

    // Accept the connection from the new client.
    // accept() does not wait or ack
    if (new_client->accept() != 0)
    {
        LOG_ERROR_MSG("error accepting client connection: %s "
            << rdma_error::error_string(errno));
        // @TODO : Handle failed connection - is there a correct thing to do
        std::terminate();
        return -1;
    }

    LOG_DEVEL_MSG("accepted connection from "
        << ipaddress(remote_ip)
        << "qp = " << decnumber(qpnum));

    return 0;
}

//----------------------------------------------------------------------------
// This function is only called from connect_to_server and is therefore
// holding the controller_mutex_ lock already
int rdma_controller::start_server_connection(uint32_t remote_ip)
{
    sockaddr_in remote_addr;
    sockaddr_in local_addr;
    //
    std::memset(&remote_addr, 0, sizeof(remote_addr));
    remote_addr.sin_family      = AF_INET;
    remote_addr.sin_port        = local_addr_.sin_port;
    remote_addr.sin_addr.s_addr = remote_ip;
    local_addr.sin_port         = 0;
    local_addr                  = local_addr_;

    LOG_DEVEL_MSG("start_server_connection      from "
        << sockaddress(&local_addr_)
        << "to " << ipaddress(remote_ip)
        << "( " << sockaddress(&local_addr_) << ")");

    // create a new client object for the remote endpoint
    verbs_endpoint_ptr new_client = std::make_shared<verbs_endpoint>(
        local_addr, remote_addr, protection_domain_, completion_queue_,
        memory_pool_, server_endpoint_->SRQ(),
        server_endpoint_->get_event_channel());

    LOG_DEVEL_MSG("SS.Map<ip <endpoint,promise>>from "
        << sockaddress(&local_addr_) << "to "
        << sockaddress(&remote_addr)
        << "( " << sockaddress(&local_addr_) << ")");

    // create a future for this connection
    hpx::promise<verbs_endpoint_ptr> new_endpoint_promise;
    hpx::future<verbs_endpoint_ptr>  new_endpoint_future =
        new_endpoint_promise.get_future();

    connections_started_.insert(
        std::make_pair(
            remote_ip,
            std::make_tuple(
                new_client,
                std::move(new_endpoint_promise),
                std::move(new_endpoint_future))));

    return 0;
}

//----------------------------------------------------------------------------
// return a future to a client - it will become ready when the
// connection is setup and ready for use
hpx::shared_future<verbs_endpoint_ptr>
rdma_controller::connect_to_server(uint32_t remote_ip)
{
    // Prevent an incoming event handler connection request,
    // and an outgoing server connect request from colliding
    scoped_lock lock(controller_mutex_);

    bool delete_connection_on_exit = false;

    // has a connection been started from here already?
    bool connection = connections_started_.is_in_map(remote_ip).second;
    LOG_DEVEL_MSG("connect to server : connections_started_.is_in_map " << connection)

    // has someone tried to connect to us already?
    if (!connection) {
        for (const auto &client_pair : qp_endpoint_map_) {
            verbs_endpoint *client = client_pair.second.get();
            if (client->get_remote_ip_address() == remote_ip)
            {
                LOG_DEVEL_MSG("connect_to_server : Found a remote connection ip "
                    << ipaddress(remote_ip));
                // we must create a future for this connection as there is no entry
                // in the connections_started_ map (a connect request from remote ip)
                hpx::promise<verbs_endpoint_ptr> new_endpoint_promise;
                hpx::future<verbs_endpoint_ptr>  new_endpoint_future =
                    new_endpoint_promise.get_future();
                //
                // if the connection was made by a connection request from outside
                // it might have already become established/ready but won't have set
                // the future ready, so do it here
                if (client->get_state()==verbs_endpoint::connection_state::connected) {
                    LOG_DEVEL_MSG("state already connected - setting promise"
                        << ipaddress(remote_ip));
                    new_endpoint_promise.set_value(client_pair.second);
                    // once the future is set, the entry can be removed
                    delete_connection_on_exit = true;
                }

                auto position = connections_started_.insert(
                    std::make_pair(
                        remote_ip,
                        std::make_tuple(
                            client_pair.second,
                            std::move(new_endpoint_promise),
                            std::move(new_endpoint_future))));

                connection = true;
                break;
            }
        }
        LOG_DEVEL_MSG("connect to server : qp_endpoint_map_.is_in_map " << connection)
    }

    // if no connection either to or from here to the remote_ip has been started ...
    if (!connection) {
        start_server_connection(remote_ip);
    }

    // the future will become ready when the remote end accepts/rejects our connection
    // or we accept a connection from a remote
    auto it = connections_started_.find(remote_ip);
    hpx::shared_future<verbs_endpoint_ptr> result = std::get<2>(it->second);
    if (delete_connection_on_exit) {
        connections_started_.erase(it);
    }
    return result;
}

//----------------------------------------------------------------------------
void rdma_controller::disconnect_all()
{
    // removing connections will affect the map, so lock it and loop over
    // each element triggering a disconnect on each
    map_read_lock_type read_lock(qp_endpoint_map_.read_write_mutex());
    //
    std::for_each(qp_endpoint_map_.begin(), qp_endpoint_map_.end(),
        [this](const rdma_controller::QPMapPair &_client) {
//            if (!_client.second->is_client_endpoint()) {
                LOG_DEVEL_MSG("Removing a connection        from "
                    << sockaddress(&local_addr_) << "to "
                    << sockaddress(_client.second->get_remote_address())
                    << "( " << sockaddress(&local_addr_) << ")");
                _client.second->disconnect();
//            }
        }
    );
}

//----------------------------------------------------------------------------
bool rdma_controller::active()
{
    map_read_lock_type read_lock(qp_endpoint_map_.read_write_mutex());
    //
    for (const auto &_client : qp_endpoint_map_) {
        verbs_endpoint *client = _client.second.get();
        if (client->get_state()!=verbs_endpoint::connection_state::terminated) {
            LOG_TIMED_INIT(terminated);
            LOG_TIMED_BLOCK(terminated, DEVEL, 5.0,
                {
                    LOG_DEVEL_MSG("still active because client in state "
                        << verbs_endpoint::ToString(client->get_state()));
                }
            )
            return true;
        }
    }
    return false;
}

#endif
