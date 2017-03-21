//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_RECEIVER_HPP

#include <plugins/parcelport/libfabric/libfabric_memory_region.hpp>
#include <plugins/parcelport/libfabric/rdma_memory_pool.hpp>
#include <plugins/parcelport/libfabric/rma_receiver.hpp>

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
        receiver()
          : region_(nullptr)
          , memory_pool_(nullptr)
        {}

        receiver(parcelport* pp, fid_ep* endpoint, rdma_memory_pool& memory_pool);

        ~receiver();

        receiver(receiver&& other);

        receiver& operator=(receiver&& other);

        void handle_recv(fi_addr_t const& src_addr, std::uint64_t len);

        void post_recv();

        libfabric_memory_region* region_;
        parcelport* pp_;
        fid_ep* endpoint_;
        rdma_memory_pool* memory_pool_;

        boost::lockfree::stack<
            rma_receiver*,
            boost::lockfree::capacity<HPX_PARCELPORT_LIBFABRIC_THROTTLE_SENDS>,
            boost::lockfree::fixed_sized<true>
        > rma_receivers_;
    };
}}}}

#endif
