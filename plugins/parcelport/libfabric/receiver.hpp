//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_RECEIVER_HPP

#include <plugins/parcelport/libfabric/libfabric_region_provider.hpp>
#include <plugins/parcelport/performance_counter.hpp>
#include <plugins/parcelport/rma_memory_pool.hpp>
#include <plugins/parcelport/libfabric/header.hpp>
#include <plugins/parcelport/libfabric/rma_receiver.hpp>

#include <hpx/util/atomic_count.hpp>

#include <boost/container/small_vector.hpp>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    struct parcelport;
    // The receiver is responsible for handling incoming messages. For that purpose,
    // it posts receive buffers. Incoming messages can be of two kinds:
    //      1) An ACK message which has been sent from an rma_receiver, to signal
    //         the sender about the succesful retreival of an incoming message.
    //      2) An incoming parcel, that consists of an header and an eventually
    //         piggy backed message. If the message is not piggy backed or zero
    //         copy RMA chunks need to be read, a rma_receiver is created to
    //         complete the transfer of the message
    struct receiver
    {
        typedef libfabric_region_provider                      region_provider;
        typedef rma_memory_region<region_provider>             region_type;
        typedef boost::container::small_vector<region_type*,8> zero_copy_vector;

        // --------------------------------------------------------------------
        // construct receive object
        receiver(parcelport* pp, fid_ep* endpoint,
                 rma_memory_pool<region_provider>& memory_pool);

        // --------------------------------------------------------------------
        // these constructors are provided because boost::lockfree::stack requires them
        // they should not be used
        receiver(receiver&& other);
        receiver& operator=(receiver&& other);

        // --------------------------------------------------------------------
        // destruct receive object
        ~receiver();

        // --------------------------------------------------------------------
        // A received message is routed by the controller into this function.
        // it might be an incoming message or just an ack sent to inform that
        // all rdma reads are complete from a previous send operation.
        void handle_recv(fi_addr_t const& src_addr, std::uint64_t len);

        // --------------------------------------------------------------------
        // the receiver posts a single receive buffer to the queue, attaching
        // itself as the context, so that when a message is received
        // the owning reciever is called to handle processing of the buffer
        void pre_post_receive();

        // --------------------------------------------------------------------
        // The cleanup call deletes resources and sums counters from internals
        // once cleanup is done, the recevier should not be used, other than
        // dumping counters
        void cleanup();

    private:
        parcelport                       *pp_;
        fid_ep                           *endpoint_;
        region_type                      *header_region_ ;
        rma_memory_pool<region_provider>  *memory_pool_;
        //
        friend class libfabric_controller;
        //
        performance_counter<unsigned int> messages_handled_;
        performance_counter<unsigned int> acks_received_;
        // from the internal rma_receivers
        performance_counter<unsigned int> msg_plain_;
        performance_counter<unsigned int> msg_rma_;
        performance_counter<unsigned int> sent_ack_;
        performance_counter<unsigned int> rma_reads_;
        performance_counter<unsigned int> recv_deletes_;
        //
        boost::lockfree::stack<
            rma_receiver*,
            boost::lockfree::capacity<HPX_PARCELPORT_LIBFABRIC_MAX_PREPOSTS>,
            boost::lockfree::fixed_sized<true>
        > rma_receivers_;

        typedef hpx::lcos::local::spinlock mutex_type;
        mutex_type active_receivers_mtx_;
        hpx::lcos::local::detail::condition_variable active_receivers_cv_;
        hpx::util::atomic_count active_receivers_;
    };
}}}}

#endif
