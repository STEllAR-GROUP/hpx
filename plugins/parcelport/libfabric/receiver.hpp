//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_RECEIVER_HPP

#include <plugins/parcelport/libfabric/libfabric_memory_region.hpp>
#include <plugins/parcelport/libfabric/rdma_memory_pool.hpp>
#include <plugins/parcelport/libfabric/header.hpp>
#include <plugins/parcelport/libfabric/performance_counter.hpp>

#include <boost/container/small_vector.hpp>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    struct parcelport;
    // The receiver is responsible for handling incoming messages. For that purpose,
    // it posts receive buffers. Incoming messages can be of two kinds:
    //      1) An ACK message which has been sent from a remote receiver, to signal
    //         the sender about the succesful retreival of an incoming message.
    //      2) An incoming parcel, that consists of an header and an eventually
    //         piggy backed message. If the message is not piggy backed or zero
    //         copy RMA chunks need to be read, RMA operations are scheduled
    struct receiver
    {
        // --------------------------------------------------------------------
        typedef serialization::serialization_chunk chunk_struct;
        typedef hpx::util::function_nonser<void(receiver*)> completion_handler;

        typedef header<HPX_PARCELPORT_LIBFABRIC_MESSAGE_HEADER_SIZE> header_type;
        static constexpr unsigned int header_size = header_type::header_block_size;

        typedef boost::container::small_vector<libfabric_memory_region*,4> zero_copy_vector;

        // --------------------------------------------------------------------
        // construct receive object
        receiver(parcelport* pp, fid_ep* endpoint, rdma_memory_pool *memory_pool);

        // --------------------------------------------------------------------
        // destruct receive object
        ~receiver();

        // --------------------------------------------------------------------
        // A received message is routed by the controller into this function.
        // it might be an incoming message or just an ack sent to inform that
        // all rdma reads are complete from a previous send operation.
        void handle_recv(fi_addr_t const& src_addr, std::uint64_t len);

        // --------------------------------------------------------------------
        // This function processes the incoming message and dispatches
        // it to the non_rma or with_rma depending on piggyback/rma data status
        void read_message(fi_addr_t const& src_addr);

        // --------------------------------------------------------------------
        // a simple piggybacked messege is read in one pass, no further processing
        void handle_message_no_rma();

        // --------------------------------------------------------------------
        // a message with rma zerocopy chunks or non piggybacked data requires
        // extr read operations to be performed from the original sender of the message
        void handle_message_with_zerocopy_rma();

        // --------------------------------------------------------------------
        // process a read completion event. If we are doing RMA, then count
        // all RMA operations and trigger complete when done
        void handle_rma_read_completion();

        // --------------------------------------------------------------------
        // An ack message is sent back to the sender when we have completed all
        // RMA operations
        void send_rdma_complete_ack();

        // --------------------------------------------------------------------
        // final cleanup before we can reuse/repost this receiver again
        void cleanup_receive();

        // --------------------------------------------------------------------
        // the receiver posts a single receive buffer to the queue, attaching
        // itself as the context, so that when a message is received
        // the owning reciever is called to handle processing of the buffer
        void pre_post_receive();

    private:
        parcelport                  *pp_;
        fid_ep                      *endpoint_;
        libfabric_memory_region     *header_region_ ;
        libfabric_memory_region     *message_region_;
        header_type                 *header_;
        std::vector<chunk_struct>    chunks_;
        zero_copy_vector             rma_regions_;
        rdma_memory_pool            *memory_pool_;
        fi_addr_t                    src_addr_;
        hpx::util::atomic_count      rma_count_;
        //
        performance_counter<unsigned int> receives_handled_;
        performance_counter<unsigned int> total_reads_;
        performance_counter<unsigned int> recv_deletes_;
    };
}}}}

#endif
