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
    // --------------------------------------------------------------------
    receiver::receiver(parcelport* pp, fid_ep* endpoint, rdma_memory_pool& memory_pool)
        : pp_(pp)
        , endpoint_(endpoint)
        , header_region_(memory_pool.allocate_region(memory_pool.small_.chunk_size()))
        , memory_pool_(&memory_pool)
        , messages_handled_(0)
        , acks_received_(0)
        , active_receivers_(0)
    {
        LOG_DEBUG_MSG("created receiver: " << hexpointer(this));
        // Once constructed, we need to post the receive...
        pre_post_receive();
    }

    // these constructors are provided because boost::lockfree::stack requires them
    // they should not be used
    receiver::receiver(receiver&& other)
        : active_receivers_(0)
    {
        std::terminate();
    }
    receiver& receiver::operator=(receiver&& other)
    {
        std::terminate();
    }

    // --------------------------------------------------------------------
    receiver::~receiver()
    {
        if (header_region_ && memory_pool_) {
            memory_pool_->deallocate(header_region_);
        }
        // this is safe to call twice - it might have been called already
        // to collect counter information by the fabric controller
        cleanup();
    }

    // --------------------------------------------------------------------
    void receiver::cleanup()
    {
        rma_receiver *rcv = nullptr;
        while(receiver::rma_receivers_.pop(rcv))
        {
            msg_plain_    += rcv->msg_plain_;
            msg_rma_      += rcv->msg_rma_;
            sent_ack_     += rcv->sent_ack_;
            rma_reads_    += rcv->rma_reads_;
            recv_deletes_ += rcv->recv_deletes_;
            delete rcv;
        }
    }

    // --------------------------------------------------------------------
    // when a receive completes, this callback handler is called
    void receiver::handle_recv(fi_addr_t const& src_addr, std::uint64_t len)
    {
        FUNC_START_DEBUG_MSG;
        static_assert(sizeof(std::uint64_t) == sizeof(std::size_t),
            "sizeof(std::uint64_t) != sizeof(std::size_t)");

        // If we recieve a message of 8 bytes, we got a tag and need to handle
        // the tag completion...
        if (len <= sizeof(std::uint64_t))
        {
            // @TODO: fixme immediate tag retreival
            // Get the sender that has completed rma operations and signal to it
            // that it can now cleanup - all remote get operations are done.
            sender* snd = *reinterpret_cast<sender **>(header_region_->get_address());
            pre_post_receive();
            LOG_DEBUG_MSG("Handling sender tag (RMA ack) completion: " << hexpointer(snd));
            ++acks_received_;
            snd->handle_message_completion_ack();
            return;
        }

        LOG_DEBUG_MSG("Handling message");
        rma_receiver* recv = nullptr;
        if (!receiver::rma_receivers_.pop(recv))
        {
            auto f = [this](rma_receiver* recv)
            {
                --active_receivers_;
                if (!receiver::rma_receivers_.push(recv)) {
                    // if the capacity overflowed, just delete this one
                    delete recv;
                }
                // Notify one possibly waiting reciever that one receive just finished
                {
                    std::unique_lock<mutex_type> l(active_receivers_mtx_);
                    active_receivers_cv_.notify_one(std::move(l));
                }
            };
            // throttle the creation of new receivers. Wait until the active_receivers_
            // count drops below the maximum. This can not be a busy wait since it could
            // potentially block all background threads.
            const long max_receivers =
                HPX_PARCELPORT_LIBFABRIC_THROTTLE_SENDS;
            while (active_receivers_ > max_receivers)
            {
                std::unique_lock<mutex_type> l(active_receivers_mtx_);
                active_receivers_cv_.wait(l);
            }

            recv = new rma_receiver(pp_, endpoint_, memory_pool_, std::move(f));
        }
        ++active_receivers_;

        HPX_ASSERT(recv);

        // We save the received region and swap it with a newly allocated
        // to be able to post a recv again as soon as possible.
        libfabric_memory_region* region = header_region_;
        header_region_ = memory_pool_->allocate_region(memory_pool_->small_.chunk_size());
        pre_post_receive();

        // we dispatch our work to our rma_receiver once it completed the
        // prior message. The saved region is passed to the rma handler
        ++messages_handled_;
        recv->read_message(region, src_addr);

        FUNC_END_DEBUG_MSG;
    }

    // --------------------------------------------------------------------
    void receiver::pre_post_receive()
    {
        FUNC_START_DEBUG_MSG;
        void* desc = header_region_->get_desc();
        LOG_DEBUG_MSG("Pre-Posting receive "
            << *header_region_
            << "context " << hexpointer(this));

        int ret = 0;
        for (std::size_t k = 0; true; ++k)
        {
            // post a receive using 'this' as the context, so that this receiver object
            // can be used to handle the incoming receive/request
            ret = fi_recv(
                endpoint_,
                header_region_->get_address(),
                header_region_->get_size(),
                desc, 0, this);

            if (ret == -FI_EAGAIN)
            {
                LOG_ERROR_MSG("reposting fi_recv\n");
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
