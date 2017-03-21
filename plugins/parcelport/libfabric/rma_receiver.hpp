//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_RMA_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_RMA_RECEIVER_HPP

#include <plugins/parcelport/libfabric/libfabric_memory_region.hpp>
#include <plugins/parcelport/libfabric/rdma_memory_pool.hpp>
#include <plugins/parcelport/libfabric/header.hpp>

#include <hpx/util/atomic_count.hpp>

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
    struct rma_receiver
    {
        typedef header<HPX_PARCELPORT_LIBFABRIC_MESSAGE_HEADER_SIZE> header_type;
        static constexpr unsigned int header_size = header_type::header_block_size;

//         typedef pinned_memory_vector<char, header_size> rcv_data_type;
//         typedef parcel_buffer<rcv_data_type, std::vector<char>>    rcv_buffer_type;

        typedef serialization::serialization_chunk chunk_struct;
        typedef hpx::util::function_nonser<void(rma_receiver*)> completion_handler;

        rma_receiver()
          : header_region_(nullptr),
            message_region_(nullptr),
            rma_count_(0)
        {}

        rma_receiver(
            parcelport* pp,
            fid_ep* endpoint,
            rdma_memory_pool* memory_pool,
            completion_handler&& handler);

        rma_receiver(rma_receiver&& other);

        rma_receiver& operator=(rma_receiver&& other);

        void async_read(
            libfabric_memory_region* region,
            fi_addr_t const& src_addr);

        ~rma_receiver();

        void handle_non_rma();

        void handle_read_completion();

        void send_ack();

    private:
        parcelport* pp_;
        fid_ep* endpoint_;
        libfabric_memory_region* header_region_;
        libfabric_memory_region* message_region_;
        header_type* header_;
        std::vector<chunk_struct> chunks_;
        std::vector<libfabric_memory_region*> rma_regions_;
        rdma_memory_pool* memory_pool_;
        fi_addr_t src_addr_;
        completion_handler handler_;
    public:
        hpx::util::atomic_count rma_count_;
    };
}}}}

#endif
