//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <plugins/parcelport/libfabric/receiver.hpp>

#include <plugins/parcelport/libfabric/libfabric_memory_region.hpp>
#include <plugins/parcelport/libfabric/rdma_memory_pool.hpp>
#include <plugins/parcelport/libfabric/pinned_memory_vector.hpp>
#include <plugins/parcelport/libfabric/header.hpp>
#include <plugins/parcelport/libfabric/parcelport_libfabric.hpp>
#include <plugins/parcelport/libfabric/sender.hpp>

#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>

#include <hpx/util/detail/yield_k.hpp>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    receiver::receiver(parcelport* pp, fid_ep* endpoint, rdma_memory_pool& memory_pool)
      : region_(memory_pool.allocate_region(memory_pool.small_.chunk_size()))
      , pp_(pp)
      , endpoint_(endpoint)
      , memory_pool_(&memory_pool)
    {
        LOG_DEBUG_MSG("created receiver: " << this);
        // Once constructed, we need to post the receive...
        post_recv();
    }

    receiver::~receiver()
    {
        if (region_ && memory_pool_)
            memory_pool_->deallocate(region_);

        rma_receiver *rcv = nullptr;
        while(rma_receivers_.pop(rcv))
        {
            delete rcv;
        }
    }

    receiver::receiver(receiver&& other)
      : region_(other.region_)
      , pp_(other.pp_)
      , endpoint_(other.endpoint_)
      , memory_pool_(other.memory_pool_)
    {
        other.region_ = nullptr;
        other.memory_pool_ = nullptr;
    }

    receiver& receiver::operator=(receiver&& other)
    {
        region_ = other.region_;
        pp_ = other.pp_;
        endpoint_ = other.endpoint_;
        memory_pool_ = other.memory_pool_;
        other.region_ = nullptr;
        other.memory_pool_ = nullptr;

        return *this;
    }

    void receiver::handle_recv(fi_addr_t const& src_addr, std::uint64_t len)
    {
        FUNC_START_DEBUG_MSG;
        static_assert(sizeof(std::uint64_t) == sizeof(std::size_t),
            "sizeof(std::uint64_t) != sizeof(std::size_t)");

        // If we recieve a message smaller than 8 byte, we got a tag and need to handle
        // the tag completion...
        if (len <= sizeof(std::uint64_t))
        {
            /// @TODO: fixme immediate tag retreival
            sender* snd = *reinterpret_cast<sender **>(region_->get_address());
            post_recv();
            LOG_DEBUG_MSG("Handling sender completion: " << hexpointer(snd));
            snd->handle_message_completion();
            return;
        }

        LOG_DEBUG_MSG("Handling message");
        rma_receiver* recv = nullptr;
        if(!rma_receivers_.pop(recv))
        {
            auto f = [this](rma_receiver* recv)
            {
                if(!rma_receivers_.push(recv))
                    delete recv;
            };
            recv = new rma_receiver(pp_, endpoint_, memory_pool_, std::move(f));
        }

        HPX_ASSERT(recv);

        // We save the received region and swap it with a newly allocated
        // to be able to post a recv again as soon as possible.
        libfabric_memory_region* region = region_;
        region_ = memory_pool_->allocate_region(
            memory_pool_->small_.chunk_size());
        post_recv();

        // we dispatch our work to our rma_receiver once it completed the
        // prior message
        recv->async_read(region, src_addr);

        FUNC_END_DEBUG_MSG;
    }

    void receiver::post_recv()
    {
        FUNC_START_DEBUG_MSG;
        void* desc = region_->get_desc();
        LOG_DEBUG_MSG("Pre-Posting a receive to client size "
            << hexnumber(memory_pool_->small_.chunk_size())
            << " descriptor " << hexpointer(desc) << " context " << hexpointer(this));

        int ret = 0;
        for (std::size_t k = 0; true; ++k)
        {
            ret = fi_recv(
                endpoint_,
                region_->get_address(),
                region_->get_size(),
                desc, 0, this);

            if (ret == -FI_EAGAIN)
            {
                LOG_DEBUG_MSG("reposting recv\n");
                hpx::util::detail::yield_k(k,
                    "libfabric::receiver::post_recv");
                continue;
            }
            if (ret!=0)
            {
                // TODO: proper error message
                throw fabric_error(ret, "pp_post_rx");
            }
            break;
        }
        FUNC_END_DEBUG_MSG;
    }
}}}}
