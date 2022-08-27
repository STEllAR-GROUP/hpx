//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/datastructures/detail/small_vector.hpp>
#include <hpx/modules/thread_support.hpp>

#include <hpx/parcelport_libfabric/config/defines.hpp>
#include <hpx/parcelport_libfabric/header.hpp>
#include <hpx/parcelport_libfabric/libfabric_region_provider.hpp>
#include <hpx/parcelport_libfabric/performance_counter.hpp>
#include <hpx/parcelport_libfabric/rma_base.hpp>
#include <hpx/parcelport_libfabric/rma_memory_pool.hpp>

#include <vector>

namespace hpx { namespace parcelset { namespace policies { namespace libfabric {
    struct parcelport;

    // The rma_receiver is responsible for receiving the
    // missing chunks of the message:
    //      1) Non-piggy backed non-zero copy chunks (if existing)
    //      2) The zero copy chunks from serialization
    struct rma_receiver : public rma_base
    {
        typedef libfabric_region_provider region_provider;
        typedef rma_memory_region<region_provider> region_type;
        typedef rma_memory_pool<region_provider> memory_pool_type;
        typedef hpx::detail::small_vector<region_type*, 8> zero_copy_vector;

        typedef header<HPX_PARCELPORT_LIBFABRIC_MESSAGE_HEADER_SIZE>
            header_type;
        static constexpr unsigned int header_size =
            header_type::header_block_size;

        typedef serialization::serialization_chunk chunk_struct;
        typedef hpx::function<void(rma_receiver*)> completion_handler;

        rma_receiver(parcelport* pp, fid_ep* endpoint,
            memory_pool_type* memory_pool, completion_handler&& handler);

        ~rma_receiver();

        // --------------------------------------------------------------------
        // the main entry point when a message is received, this function
        // will dispatch to either read with or without rma depending on
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
        // Process a message where the chunk inf0ormation did not fit into
        // the header. An extra RMA read of chunk data must be made before
        // the chunks can be identified (and possibly retrieved from the remote node)
        void handle_message_no_chunk_data();

        // --------------------------------------------------------------------
        // After remote chunks have been read by rma, process the chunk list
        // and initiate further rma reads if necessary
        int handle_chunks_read_message();

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

        // --------------------------------------------------------------------
        // convenience function to execute a read for each zero-copy chunk
        // in the chunks_ variable
        void read_chunk_list();

        // --------------------------------------------------------------------
        // convenience function to execute a read, given the right params
        void read_one_chunk(fi_addr_t src_addr, region_type* get_region,
            const void* remoteAddr, uint64_t rkey);

    private:
        parcelport* pp_;
        fid_ep* endpoint_;
        region_type* header_region_;
        region_type* chunk_region_;
        region_type* message_region_;
        header_type* header_;
        std::vector<chunk_struct> chunks_;
        zero_copy_vector rma_regions_;
        rma_memory_pool<region_provider>* memory_pool_;
        fi_addr_t src_addr_;
        completion_handler handler_;
        hpx::util::atomic_count rma_count_;
        bool chunk_fetch_;

        double start_time_;

        //
        friend class receiver;
        performance_counter<unsigned int> msg_plain_;
        performance_counter<unsigned int> msg_rma_;
        performance_counter<unsigned int> sent_ack_;
        performance_counter<unsigned int> rma_reads_;
        performance_counter<unsigned int> recv_deletes_;
    };
}}}}    // namespace hpx::parcelset::policies::libfabric
