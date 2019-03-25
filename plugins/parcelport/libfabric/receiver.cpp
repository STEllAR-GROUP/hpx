//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/parcelset/rma/memory_pool.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
//
#include <plugins/parcelport/libfabric/receiver.hpp>
#include <plugins/parcelport/libfabric/libfabric_region_provider.hpp>
#include <plugins/parcelport/libfabric/header.hpp>
#include <plugins/parcelport/libfabric/sender.hpp>
#include <plugins/parcelport/libfabric/parcelport_libfabric.hpp>
#include <plugins/parcelport/libfabric/controller.hpp>
//
#include <hpx/runtime/parcelset/decode_parcels.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
//
#include <hpx/util/assert.hpp>
//
#include <utility>
#include <cstddef>
#include <cstdint>
#include <chrono>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    performance_counter<unsigned int> receiver::messages_handled_(0);
    performance_counter<unsigned int> receiver::acks_received_(0);
    performance_counter<unsigned int> receiver::receives_pre_posted_(0);
    performance_counter<unsigned int> receiver::active_rma_receivers_(0);
    receiver::rma_stack               receiver::rma_receivers_;

    // --------------------------------------------------------------------
    receiver::receiver(parcelport* pp, fid_ep* endpoint,
        rma::memory_pool<region_provider>& memory_pool)
        : rma_base(ctx_receiver)
        , parcelport_(pp)
        , endpoint_(endpoint)
        , header_region_(memory_pool.allocate_region(memory_pool.small_.chunk_size()))
        , memory_pool_(&memory_pool)
    {
        LOG_TRACE_MSG("created receiver: " << hexpointer(this));
        // create an rma_receivers per receive and push it onto the rma stack
        create_rma_receiver(true);
        // Once constructed, we need to post the receive...
        pre_post_receive();
    }

    // --------------------------------------------------------------------
    // constructor provided because boost::lockfree::stack requires it
    // (should not be used)
    receiver::receiver(receiver&& other) : rma_base(other.context_type())
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
//        rma_receiver *rcv = nullptr;
        //
//        while(receiver::rma_receivers_.pop(rcv))
//        {
//            msg_plain_    += rcv->msg_plain_;
//            msg_rma_      += rcv->msg_rma_;
//            sent_ack_     += rcv->sent_ack_;
//            rma_reads_    += rcv->rma_reads_;
//            recv_deletes_ += rcv->recv_deletes_;
//            LOG_ERROR_MSG("Cleanup rma_receiver " << decnumber(--active_rma_receivers_));
//            delete rcv;
//        }
    }

    // --------------------------------------------------------------------
    // A new connection only contains a locality address of the sender
    // so it can be handled directly without creating an rma_receiver
    // just get the address and add it to the parclport address_vector
    bool receiver::handle_new_connection(controller *controller, std::uint64_t len)
    {
        FUNC_START_DEBUG_MSG;
        LOG_DEBUG_MSG("Processing new connection message of length "
                      << decnumber(len)
                      << "pre-posted " << decnumber(--receives_pre_posted_));

        // We save the received region and swap it with a newly allocated one
        // so that we can post a recv again as soon as possible.
        region_type* region = header_region_;
        header_region_ = memory_pool_->allocate_region(memory_pool_->small_.chunk_size());
        pre_post_receive();

        LOG_TRACE_MSG(CRC32_MEM(region->get_address(),
                                len, "Header region (new connection)"));

        rma_receiver::header_type *header =
                reinterpret_cast<rma_receiver::header_type*>(region->get_address());

        // The message size should match the locality data size
        HPX_ASSERT(header->message_size() == locality::array_size);

        parcelset::policies::libfabric::locality source_addr;
        std::memcpy(source_addr.fabric_data_writable(),
                    header->message_data(),
                    source_addr.array_size);
        LOG_DEBUG_MSG("Received connection bootstrap locality "
                      << iplocality(source_addr));

        // free up the region we consumed
        memory_pool_->deallocate(region);

        // Add the sender's address to the address vector and update it
        // with the fi_addr address vector table index (rank)
        source_addr = controller->insert_address(source_addr);
        controller->update_bootstrap_connections();

        FUNC_END_DEBUG_MSG;
        return true;
    }

    // --------------------------------------------------------------------
    // when a receive completes, this callback handler is called
    rma_receiver *receiver::create_rma_receiver(bool push_to_stack)
    {
        // this is the rma_receiver completion handling function
        // it just returns the rma_receiver back to the stack
        auto f = [](rma_receiver* recv)
        {
            ++active_rma_receivers_;
            LOG_DEBUG_MSG("Pushing rma_receiver " << decnumber(active_rma_receivers_));
            if (!receiver::rma_receivers_.push(recv)) {
                // if the capacity overflowed, just delete this one
                LOG_TRACE_MSG("stack full rma_receiver " << decnumber(active_rma_receivers_));
                delete recv;
            }
        };

        // Put a new rma_receiver on the stack
        rma_receiver *recv = new rma_receiver(parcelport_, endpoint_, memory_pool_, std::move(f));
        ++active_rma_receivers_;
        LOG_DEBUG_MSG("Creating new rma_receiver " << decnumber(active_rma_receivers_));
        if (push_to_stack) {
            if (!receiver::rma_receivers_.push(recv)) {
                // if the capacity overflowed, just delete this one
                LOG_TRACE_MSG("stack full new rma_receiver " << decnumber(active_rma_receivers_));
                delete recv;
            }
        }
        else {
            return recv;
        }
        return nullptr;
    }

    // --------------------------------------------------------------------
    rma_receiver* receiver::get_rma_receiver(fi_addr_t const& src_addr)
    {
        rma_receiver *recv = nullptr;
        // cannot yield here - might be called from background thread
        if (!receiver::rma_receivers_.pop(recv)) {
            recv = create_rma_receiver(false);
        }
        --active_rma_receivers_;
        LOG_DEBUG_MSG("rma_receiver " << decnumber(active_rma_receivers_));
        //
        recv->src_addr_       = src_addr;
        recv->endpoint_       = endpoint_;
        recv->header_region_  = nullptr;
        recv->chunk_region_   = nullptr;
        recv->message_region_ = nullptr;
        recv->header_         = nullptr;
        recv->rma_count_      = 0;
        recv->chunk_fetch_    = false;
        return recv;
    }

    // --------------------------------------------------------------------
    // when a receive completes, this callback handler is called
    void receiver::handle_recv(fi_addr_t const& src_addr, std::uint64_t len)
    {
        FUNC_START_DEBUG_MSG;
        static_assert(sizeof(std::uint64_t) == sizeof(std::size_t),
            "sizeof(std::uint64_t) != sizeof(std::size_t)");

        LOG_DEBUG_MSG("handling recv message "
                      << "pre-posted " << decnumber(--receives_pre_posted_));

        // If we recieve a message of 8 bytes, we got a tag and need to handle
        // the tag completion...
        if (len <= sizeof(std::uint64_t))
        {
            // @TODO: fixme immediate tag retrieval
            // Get the sender that has completed rma operations and signal to it
            // that it can now cleanup - all remote get operations are done.
            sender* snd = *reinterpret_cast<sender **>(header_region_->get_address());
            pre_post_receive();
            LOG_DEBUG_MSG("Handling sender tag (RMA ack) completion: "
                << hexpointer(snd));
            ++acks_received_;
            snd->handle_message_completion_ack();
            return;
        }

        rma_receiver* recv = get_rma_receiver(src_addr);

        // We save the received region and swap it with a newly allocated one
        // so that we can post a recv again as soon as possible.
        region_type* region = header_region_;
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
        void *desc = header_region_->get_local_key();
        LOG_DEBUG_MSG("Pre-Posting receive "
            << *header_region_
            << "context " << hexpointer(this)
            << "pre-posted " << decnumber(++receives_pre_posted_));

        // this should never actually return true and yield
        bool ok = false;
        while(!ok) {
            // post a receive using 'this' as the context, so that this
            // receiver object can be used to handle the incoming
            // receive/request
            ssize_t ret = fi_recv(this->endpoint_,
                this->header_region_->get_address(),
                this->header_region_->get_size(), desc, 0, this);

            if (ret ==0) {
                ok = true;
            }
            else if (ret == -FI_EAGAIN)
            {
                LOG_ERROR_MSG("reposting fi_recv\n");
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
            else if (ret != 0)
            {
                throw fabric_error(int(ret), "pp_post_rx");
            }
        }
        FUNC_END_DEBUG_MSG;
    }
}}}}
