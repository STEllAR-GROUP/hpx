//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/parcelport_libfabric/config/defines.hpp>

#include <hpx/assert.hpp>
#include <hpx/modules/execution_base.hpp>

#include <hpx/parcelport_libfabric/header.hpp>
#include <hpx/parcelport_libfabric/libfabric_region_provider.hpp>
#include <hpx/parcelport_libfabric/parcelport_libfabric.hpp>
#include <hpx/parcelport_libfabric/receiver.hpp>
#include <hpx/parcelport_libfabric/rma_memory_pool.hpp>
#include <hpx/parcelport_libfabric/sender.hpp>

#include <hpx/parcelset/decode_parcels.hpp>
#include <hpx/parcelset/parcel_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <utility>

namespace hpx::parcelset::policies::libfabric {

    // --------------------------------------------------------------------
    receiver::receiver(parcelport* pp, fid_ep* endpoint,
        rma_memory_pool<region_provider>& memory_pool)
      : pp_(pp)
      , endpoint_(endpoint)
      , header_region_(
            memory_pool.allocate_region(memory_pool.small_.chunk_size()))
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
    receiver::receiver(receiver&&)
      : active_receivers_(0)
    {
        std::terminate();
    }
    receiver& receiver::operator=(receiver&&)
    {
        std::terminate();
    }

    // --------------------------------------------------------------------
    receiver::~receiver()
    {
        if (header_region_ && memory_pool_)
        {
            memory_pool_->deallocate(header_region_);
        }
        // this is safe to call twice - it might have been called already
        // to collect counter information by the fabric controller
        cleanup();
    }

    // --------------------------------------------------------------------
    void receiver::cleanup()
    {
        rma_receiver* rcv = nullptr;
        while (receiver::rma_receivers_.pop(rcv))
        {
            msg_plain_ += rcv->msg_plain_;
            msg_rma_ += rcv->msg_rma_;
            sent_ack_ += rcv->sent_ack_;
            rma_reads_ += rcv->rma_reads_;
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

        // If we receive a message of 8 bytes, we got a tag and need to handle
        // the tag completion...
        if (len <= sizeof(std::uint64_t))
        {
            // @TODO: fixme immediate tag retrieval
            // Get the sender that has completed rma operations and signal to it
            // that it can now cleanup - all remote get operations are done.
            sender* snd =
                *reinterpret_cast<sender**>(header_region_->get_address());
            pre_post_receive();
            LOG_DEBUG_MSG("Handling sender tag (RMA ack) completion: "
                << hexpointer(snd));
            ++acks_received_;
            snd->handle_message_completion_ack();
            return;
        }

        LOG_DEBUG_MSG("Handling message");
        rma_receiver* recv = nullptr;
        if (!receiver::rma_receivers_.pop(recv))
        {
            auto f = [this](rma_receiver* recv) {
                --active_receivers_;
                if (!receiver::rma_receivers_.push(recv))
                {
                    // if the capacity overflowed, just delete this one
                    delete recv;
                }
                // Notify one possibly waiting receiver that one receive just
                // finished
                if (threads::threadmanager_is_at_least(hpx::state::running) &&
                    hpx::threads::get_self_ptr())
                {
                    std::unique_lock<mutex_type> l(active_receivers_mtx_);
                    active_receivers_cv_.notify_one(HPX_MOVE(l));
                }
            };
            // throttle the creation of new receivers. Wait until the
            // active_receivers_count drops below the maximum. This can not be
            // a busy wait since it could potentially block all background
            // threads.
            const long max_receivers = HPX_PARCELPORT_LIBFABRIC_MAX_PREPOSTS;
            if (threads::threadmanager_is_at_least(hpx::state::running) &&
                hpx::threads::get_self_ptr())
            {
                while (active_receivers_ > max_receivers)
                {
                    std::unique_lock<mutex_type> l(active_receivers_mtx_);
                    active_receivers_cv_.wait(l);
                }
            }

            recv = new rma_receiver(pp_, endpoint_, memory_pool_, HPX_MOVE(f));
        }
        ++active_receivers_;

        HPX_ASSERT(recv);

        // We save the received region and swap it with a newly allocated one
        // so that we can post a recv again as soon as possible.
        region_type* region = header_region_;
        header_region_ =
            memory_pool_->allocate_region(memory_pool_->small_.chunk_size());
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
        LOG_DEBUG_MSG("Pre-Posting receive " << *header_region_ << "context "
                                             << hexpointer(this));

        hpx::util::yield_while(
            [this, desc]() {
                // post a receive using 'this' as the context, so that this
                // receiver object can be used to handle the incoming
                // receive/request
                int ret = fi_recv(this->endpoint_,
                    this->header_region_->get_address(),
                    this->header_region_->get_size(), desc, 0, this);

                if (ret == -FI_EAGAIN)
                {
                    LOG_ERROR_MSG("reposting fi_recv\n");
                    return true;
                }
                else if (ret != 0)
                {
                    throw fabric_error(ret, "pp_post_rx");
                }

                return false;
            },
            "libfabric::receiver::post_recv");

        FUNC_END_DEBUG_MSG;
    }
}    // namespace hpx::parcelset::policies::libfabric
