//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <plugins/parcelport/libfabric/header.hpp>
#include <plugins/parcelport/libfabric/libfabric_region_provider.hpp>
#include <plugins/parcelport/libfabric/parcelport_libfabric.hpp>
#include <plugins/parcelport/libfabric/sender.hpp>
#include <plugins/parcelport/rma_memory_pool.hpp>
//
#include <hpx/assert.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/timing/high_resolution_timer.hpp>
#include <hpx/execution_base/this_thread.hpp>
//
#include <rdma/fi_endpoint.h>
//
#include <memory>
#include <cstddef>
#include <cstring>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    // --------------------------------------------------------------------
    // The main message send routine
    void sender::async_write_impl()
    {
        buffer_.data_point_.time_ = hpx::chrono::high_resolution_clock::now();
        HPX_ASSERT(message_region_ == nullptr);
        HPX_ASSERT(completion_count_ == 0);
        // increment counter of total messages sent
        ++sends_posted_;

        // for each zerocopy chunk, we must create a memory region for the data
        // do this before creating the header as the chunk details will be copied
        // into the header space
        int index = 0;
        for (auto &c : buffer_.chunks_)
        {
            // Debug only, dump out the chunk info
            LOG_DEBUG_MSG("write : chunk : size " << hexnumber(c.size_)
                    << " type " << decnumber((uint64_t)c.type_)
                    << " rkey " << hexpointer(c.rkey_)
                    << " cpos " << hexpointer(c.data_.cpos_)
                    << " index " << decnumber(c.data_.index_));
            if (c.type_ == serialization::chunk_type_pointer)
            {
                LOG_EXCLUSIVE(chrono::high_resolution_timer regtimer);

                // create a new memory region from the user supplied pointer
                region_type *zero_copy_region =
                    new region_type(domain_, c.data_.cpos_, c.size_);

                rma_regions_.push_back(zero_copy_region);

                // set the region remote access key in the chunk space
                c.rkey_  = zero_copy_region->get_remote_key();
                    LOG_DEBUG_MSG("Time to register memory (ns) "
                        << decnumber(regtimer.elapsed_nanoseconds()));
                LOG_DEBUG_MSG("Created zero-copy rdma Get region "
                    << decnumber(index) << *zero_copy_region
                    << "for rkey " << hexpointer(c.rkey_));

                LOG_TRACE_MSG(
                    CRC32_MEM(zero_copy_region->get_address(),
                        zero_copy_region->get_message_length(),
                        "zero_copy_region (pre-send) "));
            }
            ++index;
        }

        // create the header using placement new in the pinned memory block
        char *header_memory = (char*)(header_region_->get_address());

        LOG_DEBUG_MSG("Placement new for header");
        header_ = new(header_memory) header_type(buffer_, this);
        header_region_->set_message_length(header_->header_length());

        LOG_DEBUG_MSG("sender " << hexpointer(this)
            << ", buffsize " << hexuint32(header_->message_size())
            << ", header_length " << decnumber(header_->header_length())
            << ", chunks zerocopy( " << decnumber(buffer_.num_chunks_.first) << ") "
            << ", normal( " << decnumber(buffer_.num_chunks_.second) << ") "
            << ", chunk_flag " << decnumber(header_->header_length())
            << ", tag " << hexuint64(header_->tag())
        );

        // reserve some space for zero copy information
        rma_regions_.reserve(buffer_.num_chunks_.first);

        // Get the block of pinned memory where the message was encoded
        // during serialization
        message_region_ = buffer_.data_.m_region_;
        message_region_->set_message_length(header_->message_size());

        HPX_ASSERT(header_->message_size() == buffer_.data_.size());
        LOG_DEBUG_MSG("Found region allocated during encode_parcel : address "
            << hexpointer(buffer_.data_.m_array_)
            << " region "<< *message_region_);

        // The number of completions we need before cleaning up:
        // 1 (header block send) + 1 (ack message if we have RMA chunks)
        completion_count_ = 1;
        region_list_[0] = {
            header_region_->get_address(), header_region_->get_message_length() };
        region_list_[1] = {
            message_region_->get_address(), message_region_->get_message_length() };

        desc_[0] = header_region_->get_desc();
        desc_[1] = message_region_->get_desc();
        if (rma_regions_.size()>0 || !header_->message_piggy_back()) {
            completion_count_ = 2;
        }

        if (header_->chunk_data()) {
            LOG_DEBUG_MSG("Sender " << hexpointer(this)
                << "Chunk info is piggybacked");
        }
        else {
            LOG_DEBUG_MSG("Setting up header-chunk rma data with "
                << "zero-copy chunks " << decnumber(rma_regions_.size()));
            auto &cb = header_->chunk_header_ptr()->chunk_rma;
            chunk_region_  = memory_pool_->allocate_region(cb.size_);
            cb.data_.pos_  = chunk_region_->get_address();
            cb.rkey_       = chunk_region_->get_remote_key();
            std::memcpy(cb.data_.pos_, buffer_.chunks_.data(), cb.size_);
            LOG_DEBUG_MSG("Set up header-chunk rma data with "
                << "size " << decnumber(cb.size_)
                << "rkey " << hexpointer(cb.rkey_)
                << "addr " << hexpointer(cb.data_.cpos_));
        }

        if (header_->message_piggy_back())
        {
            LOG_DEBUG_MSG("Sender " << hexpointer(this)
                << "Main message is piggybacked");

            LOG_TRACE_MSG(CRC32_MEM(header_region_->get_address(),
                header_region_->get_message_length(),
                "Header region (send piggyback)"));

            LOG_TRACE_MSG(CRC32_MEM(message_region_->get_address(),
                message_region_->get_message_length(),
                "Message region (send piggyback)"));

            // send 2 regions as one message, goes into one receive
            hpx::util::yield_while([this]()
                {
                    int ret = fi_sendv(this->endpoint_, this->region_list_,
                        this->desc_, 2, this->dst_addr_, this);

                    if (ret == -FI_EAGAIN)
                    {
                        LOG_ERROR_MSG("reposting fi_sendv...\n");
                        return true;
                    }
                    else if (ret)
                    {
                        throw fabric_error(ret, "fi_sendv");
                    }

                    return false;
                }, "sender::async_write");
        }
        else
        {
            header_->set_message_rdma_info(
                message_region_->get_remote_key(), message_region_->get_address());

            LOG_DEBUG_MSG("Sender " << hexpointer(this)
                << "message region NOT piggybacked "
                << hexnumber(buffer_.data_.size())
                << *message_region_);

            LOG_TRACE_MSG(CRC32_MEM(header_region_->get_address(),
                header_region_->get_message_length(),
                "Header region (pre-send)"));

            LOG_TRACE_MSG(CRC32_MEM(message_region_->get_address(),
                message_region_->get_message_length(),
                "Message region (send for rdma fetch)"));

            // send just the header region - a single message
            hpx::util::yield_while([this]()
                {
                    int ret = fi_send(this->endpoint_,
                        this->region_list_[0].iov_base,
                        this->region_list_[0].iov_len,
                        this->desc_[0], this->dst_addr_, this);

                    if (ret == -FI_EAGAIN)
                    {
                        LOG_ERROR_MSG("reposting fi_send...\n");
                        return true;
                    }
                    else if (ret)
                    {
                        throw fabric_error(ret, "fi_sendv");
                    }

                    return false;
                }, "sender::async_write");
        }

        FUNC_END_DEBUG_MSG;
    }

    // --------------------------------------------------------------------
    void sender::handle_send_completion()
    {
        LOG_DEBUG_MSG("Sender " << hexpointer(this)
            << "handle send_completion "
            << "RMA regions " << decnumber(rma_regions_.size())
            << "completion count " << decnumber(completion_count_));
        cleanup();
    }

    // --------------------------------------------------------------------
    void sender::handle_message_completion_ack()
    {
        LOG_DEBUG_MSG("Sender " << hexpointer(this)
            << "handle handle_message_completion_ack ( "
            << "RMA regions " << decnumber(rma_regions_.size())
            << "completion count " << decnumber(completion_count_));
        ++acks_received_;
        cleanup();
    }

    // --------------------------------------------------------------------
    void sender::cleanup()
    {
        LOG_DEBUG_MSG("Sender " << hexpointer(this)
            << "decrementing completion_count from " << decnumber(completion_count_));

        // if we need to wait for more completion events, return without cleaning
        if (--completion_count_ > 0)
            return;

        // track deletions
        ++sends_deleted_;

        error_code ec;
        handler_(ec);
        handler_.reset();

        // cleanup header and message region
        memory_pool_->deallocate(message_region_);
        message_region_ = nullptr;
        header_         = nullptr;
        // cleanup chunk region
        if (chunk_region_) {
            memory_pool_->deallocate(chunk_region_);
            chunk_region_ = nullptr;
        }

        for (auto& region: rma_regions_) {
            memory_pool_->deallocate(region);
        }
        rma_regions_.clear();
        buffer_.data_point_.time_ =
            hpx::chrono::high_resolution_clock::now() - buffer_.data_point_.time_;
        parcelport_->add_sent_data(buffer_.data_point_);
        postprocess_handler_(this);
    }

    // --------------------------------------------------------------------
    void sender::handle_error(struct fi_cq_err_entry err)
    {
        LOG_ERROR_MSG("resending message after error " << hexpointer(this));

        if (header_->message_piggy_back())
        {
            // send 2 regions as one message, goes into one receive
            hpx::util::yield_while([this]()
                {
                    int ret = fi_sendv(this->endpoint_, this->region_list_,
                        this->desc_, 2, this->dst_addr_, this);

                    if (ret == -FI_EAGAIN)
                    {
                        LOG_ERROR_MSG("reposting fi_sendv...\n");
                        return true;
                    }
                    else if (ret)
                    {
                        throw fabric_error(ret, "fi_sendv");
                    }

                    return false;
                }, "libfabric::sender::handle_error");
        }
        else
        {
            header_->set_message_rdma_info(
                message_region_->get_remote_key(), message_region_->get_address());

            // send just the header region - a single message
            hpx::util::yield_while([this]()
                {
                    int ret = fi_send(this->endpoint_,
                        this->region_list_[0].iov_base,
                        this->region_list_[0].iov_len,
                        this->desc_[0], this->dst_addr_, this);

                    if (ret == -FI_EAGAIN)
                    {
                        LOG_ERROR_MSG("reposting fi_send...\n");
                        return true;
                    }
                    else if (ret)
                    {
                        throw fabric_error(ret, "fi_sendv");
                    }

                    return false;
                }, "libfabric::sender::handle_error");
        }
    }

}}}}
