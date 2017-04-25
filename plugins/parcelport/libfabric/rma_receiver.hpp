//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_RMA_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_RMA_RECEIVER_HPP

#include <plugins/parcelport/libfabric/libfabric_region_provider.hpp>
#include <plugins/parcelport/performance_counter.hpp>
#include <plugins/parcelport/rma_memory_pool.hpp>
#include <plugins/parcelport/libfabric/header.hpp>
#include <plugins/parcelport/libfabric/rma_base.hpp>
//
#include <hpx/util/atomic_count.hpp>
//
#include <boost/container/small_vector.hpp>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    struct parcelport;


    // The rma_receiver is repsonsible for receiving the
    // mising chunks of the message:
    //      1) Non-piggy backed non-zero copy chunks (if existing)
    //      2) The zero copy chunks from serialization
    struct rma_receiver : public rma_base
    {
        typedef libfabric_region_provider                      region_provider;
        typedef rma_memory_region<region_provider>             region_type;
        typedef rma_memory_pool<region_provider>               memory_pool_type;
        typedef boost::container::small_vector<region_type*,8> zero_copy_vector;

        typedef header<HPX_PARCELPORT_LIBFABRIC_MESSAGE_HEADER_SIZE> header_type;
        static constexpr unsigned int header_size = header_type::header_block_size;

        typedef serialization::serialization_chunk chunk_struct;
        typedef hpx::util::function_nonser<void(rma_receiver*)> completion_handler;

        rma_receiver(
            parcelport *pp,
            fid_ep *endpoint,
            memory_pool_type *memory_pool,
            completion_handler&& handler);

        ~rma_receiver();

        // --------------------------------------------------------------------
        // the main entry point when a message is received, this function
        // will despatch to either read with or without rma depending on
        // whether there are zero copy chunks to handle
        void read_message(region_type* region, fi_addr_t const& src_addr);

        // --------------------------------------------------------------------
        // Process a message that has no zero copy chunks
        void handle_message_no_rma();

        // --------------------------------------------------------------------
        // Process a message that has zero copy chunks. for each chunk we
        // make an RMA read request
        void handle_message_with_zerocopy_rma();

        // --------------------------------------------------------------------
        // Each RMA read completion will enter this function and count down until
        // all are done, then we can process the parcel and cleanup
        void handle_rma_read_completion();

        // --------------------------------------------------------------------
        // Once all RMA reads are complete, we must send an ack to the origin
        // of the parcel so that it can release the RMA regions it is holding onto
        void send_rdma_complete_ack();

        // --------------------------------------------------------------------
        // After message processing is complete, this routine cleans up and resets
        void cleanup_receive();

        // --------------------------------------------------------------------
        void handle_error(struct fi_cq_err_entry err) override;

    private:
        parcelport                   *pp_;
        fid_ep                       *endpoint_;
        region_type                  *header_region_;
        region_type                  *message_region_;
        header_type                  *header_;
        std::vector<chunk_struct>     chunks_;
        zero_copy_vector              rma_regions_;
        rma_memory_pool<region_provider> *memory_pool_;
        fi_addr_t                     src_addr_;
        completion_handler            handler_;
        hpx::util::atomic_count       rma_count_;

        double start_time_;

        //
        friend class receiver;
        performance_counter<unsigned int> msg_plain_;
        performance_counter<unsigned int> msg_rma_;
        performance_counter<unsigned int> sent_ack_;
        performance_counter<unsigned int> rma_reads_;
        performance_counter<unsigned int> recv_deletes_;

    };
}}}}

#endif
