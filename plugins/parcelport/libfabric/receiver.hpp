//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_RECEIVER_HPP

#include <hpx/runtime/parcelset/rma/detail/memory_region_impl.hpp>
#include <hpx/runtime/parcelset/rma/memory_pool.hpp>
#include <hpx/util/atomic_count.hpp>
//
#include <plugins/parcelport/performance_counter.hpp>
//
#include <plugins/parcelport/libfabric/libfabric_region_provider.hpp>
#include <plugins/parcelport/libfabric/header.hpp>
#include <plugins/parcelport/libfabric/rma_base.hpp>
#include <plugins/parcelport/libfabric/rma_receiver.hpp>
//
#include <boost/container/small_vector.hpp>
//
#include <cstdint>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    struct parcelport;
    class controller;

    // The receiver is responsible for handling incoming messages. For that purpose,
    // it posts receive buffers. Incoming messages can be of two kinds:
    //      1) An ACK message which has been sent from an rma_receiver, to signal
    //         the sender about the succesful retreival of an incoming message.
    //      2) An incoming parcel, that consists of an header and an eventually
    //         piggy backed message. If the message is not piggy backed or zero
    //         copy RMA chunks need to be read, a rma_receiver is created to
    //         complete the transfer of the message
    struct receiver : public rma_base
    {
        typedef libfabric_region_provider                        region_provider;
        typedef rma::detail::memory_region_impl<region_provider> region_type;
        typedef boost::container::small_vector<region_type*,8>   zero_copy_vector;

        // --------------------------------------------------------------------
        // construct receive object
        receiver(parcelport* pp, fid_ep* endpoint,
                 rma::memory_pool<region_provider>& memory_pool);

        // --------------------------------------------------------------------
        // destruct receive object
        ~receiver();

        // --------------------------------------------------------------------
        bool handle_new_connection(controller *controller, std::uint64_t len);

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

        rma_receiver* create_rma_receiver(bool push_to_stack);
        rma_receiver* get_rma_receiver(fi_addr_t const& src_addr);

        // --------------------------------------------------------------------
        // The cleanup call deletes resources and sums counters from internals
        // once cleanup is done, the recevier should not be used, other than
        // dumping counters
        void cleanup();

        // --------------------------------------------------------------------
        // Not used, provided for potential/future rma_base compatibility
        void handle_error(struct fi_cq_err_entry err) {};

    private:
        parcelport                        *parcelport_;
        fid_ep                            *endpoint_;
        region_type                       *header_region_ ;
        rma::memory_pool<region_provider> *memory_pool_;
        //
        friend class controller;

        // shared performance counters used by all receivers
        static performance_counter<unsigned int> messages_handled_;
        static performance_counter<unsigned int> acks_received_;
        static performance_counter<unsigned int> receives_pre_posted_;
        static performance_counter<unsigned int> active_rma_receivers_;

        //
        typedef boost::lockfree::stack<
            rma_receiver*,
            boost::lockfree::capacity<HPX_PARCELPORT_LIBFABRIC_MAX_PREPOSTS>,
            boost::lockfree::fixed_sized<true>
        > rma_stack;
        static rma_stack rma_receivers_;
    };
}}}}

#endif
