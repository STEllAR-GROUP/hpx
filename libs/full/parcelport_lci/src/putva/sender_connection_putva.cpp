//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/assert.hpp>

#include <hpx/modules/lci_base.hpp>
#include <hpx/parcelport_lci/header.hpp>
#include <hpx/parcelport_lci/helper.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelport_lci/parcelport_lci.hpp>
#include <hpx/parcelport_lci/receiver_base.hpp>
#include "hpx/parcelport_lci/putva/sender_connection_putva.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lci {

    bool sender_connection_putva::can_be_eager_message(size_t eager_threshold)
    {
        int num_zero_copy_chunks = static_cast<int>(buffer_.num_chunks_.first);
        if (num_zero_copy_chunks > 0)
            // if there are non-zero-copy chunks, we have to use iovec
            return false;
        size_t header_size = sizeof(header::header_format_t);
        size_t data_size = buffer_.data_.size();
        size_t tchunk_size = buffer_.transmission_chunks_.size() *
            sizeof(parcel_buffer_type::transmission_chunk_type);
        if (header_size + data_size + tchunk_size <= eager_threshold)
            return true;
        else
            return false;
    }

    void sender_connection_putva::load(
        sender_connection_putva::handler_type&& handler,
        sender_connection_putva::postprocess_handler_type&& parcel_postprocess)
    {
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
        data_point_ = buffer_.data_point_;
        data_point_.time_ = hpx::chrono::high_resolution_clock::now();
#endif
        conn_start_time = util::lci_environment::pcounter_now();
        HPX_ASSERT(!handler_);
        HPX_ASSERT(!postprocess_handler_);
        HPX_ASSERT(!buffer_.data_.empty());
        handler_ = HPX_MOVE(handler);
        postprocess_handler_ = HPX_MOVE(parcel_postprocess);

        // build header
        header header_;
        is_eager = can_be_eager_message(LCI_MEDIUM_SIZE);
        int num_zero_copy_chunks = static_cast<int>(buffer_.num_chunks_.first);
        if (is_eager)
        {
            int retry_count = 0;
            while (LCI_mbuffer_alloc(device_p->device, &mbuffer) != LCI_OK)
                yield_k(retry_count, config_t::mbuffer_alloc_max_retry);
            HPX_ASSERT(mbuffer.length == (size_t) LCI_MEDIUM_SIZE);
            header_ = header(buffer_, (char*) mbuffer.address, mbuffer.length);
            mbuffer.length = header_.size();
            cleanup();
        }
        else
        {
            size_t max_header_size =
                LCI_get_iovec_piggy_back_size(num_zero_copy_chunks + 2);
            char* header_buffer = (char*) malloc(max_header_size);
            header_ = header(buffer_, header_buffer, max_header_size);

            // calculate the exact number of long messages to send
            int long_msg_num = num_zero_copy_chunks;
            if (!header_.piggy_back_data())
                ++long_msg_num;
            // transmission chunks
            if (num_zero_copy_chunks != 0 && !header_.piggy_back_tchunk())
                ++long_msg_num;

            // initialize iovec
            iovec = LCI_iovec_t();
            iovec.piggy_back.address = header_.data();
            iovec.piggy_back.length = header_.size();
            iovec.count = long_msg_num;
            int i = 0;
            iovec.lbuffers =
                (LCI_lbuffer_t*) malloc(iovec.count * sizeof(LCI_lbuffer_t));
            if (!header_.piggy_back_data())
            {
                // data (non-zero-copy chunks)
                iovec.lbuffers[i].address = buffer_.data_.data();
                iovec.lbuffers[i].length = buffer_.data_.size();
                if (config_t::reg_mem)
                {
                    LCI_memory_register(device_p->device,
                        iovec.lbuffers[i].address, iovec.lbuffers[i].length,
                        &iovec.lbuffers[i].segment);
                }
                else
                {
                    iovec.lbuffers[i].segment = LCI_SEGMENT_ALL;
                }
                ++i;
            }
            if (num_zero_copy_chunks != 0)
            {
                // transmission chunk
                if (!header_.piggy_back_tchunk())
                {
                    std::vector<
                        typename parcel_buffer_type::transmission_chunk_type>&
                        tchunks = buffer_.transmission_chunks_;
                    size_t tchunks_length = tchunks.size() *
                        sizeof(parcel_buffer_type::transmission_chunk_type);
                    iovec.lbuffers[i].address = tchunks.data();
                    iovec.lbuffers[i].length = tchunks_length;
                    if (config_t::reg_mem)
                    {
                        LCI_memory_register(device_p->device,
                            iovec.lbuffers[i].address, iovec.lbuffers[i].length,
                            &iovec.lbuffers[i].segment);
                    }
                    else
                    {
                        iovec.lbuffers[i].segment = LCI_SEGMENT_ALL;
                    }
                    ++i;
                }
                // zero-copy chunks
                for (int j = 0; j < (int) buffer_.chunks_.size(); ++j)
                {
                    serialization::serialization_chunk& c = buffer_.chunks_[j];
                    if (c.type_ ==
                            serialization::chunk_type::chunk_type_pointer ||
                        c.type_ ==
                            serialization::chunk_type::chunk_type_const_pointer)
                    {
                        HPX_ASSERT(long_msg_num > i);
                        iovec.lbuffers[i].address =
                            const_cast<void*>(c.data_.cpos_);
                        iovec.lbuffers[i].length = c.size_;
                        if (config_t::reg_mem)
                        {
                            LCI_memory_register(device_p->device,
                                iovec.lbuffers[i].address,
                                iovec.lbuffers[i].length,
                                &iovec.lbuffers[i].segment);
                        }
                        else
                        {
                            iovec.lbuffers[i].segment = LCI_SEGMENT_ALL;
                        }
                        ++i;
                    }
                }
            }
            HPX_ASSERT(long_msg_num == i);
            sharedPtr_p = new std::shared_ptr<sender_connection_putva>(
                std::dynamic_pointer_cast<sender_connection_putva>(
                    shared_from_this()));
        }
        profile_start_hook(header_);
        state.store(connection_state::initialized, std::memory_order_release);
    }

    bool sender_connection_putva::isEager()
    {
        return is_eager;
    }

    sender_connection_putva::return_t sender_connection_putva::send_nb()
    {
        switch (state.load(std::memory_order_acquire))
        {
        case connection_state::initialized:
            return send_msg();

        case connection_state::sent:
            return {return_status_t::done, nullptr};

        case connection_state::locked:
            return {return_status_t::retry, nullptr};

        default:
            throw std::runtime_error("Unexpected send state!");
        }
    }

    sender_connection_putva::return_t sender_connection_putva::send_msg()
    {
        const auto current_state = connection_state::initialized;
        const auto next_state = connection_state::sent;
        HPX_ASSERT(state.load(std::memory_order_acquire) == current_state);

        int ret;
        if (is_eager)
        {
            ret = LCI_putmna(device_p->endpoint_new, mbuffer, dst_rank, 0,
                LCI_DEFAULT_COMP_REMOTE);
            if (ret == LCI_OK)
            {
                state.store(next_state, std::memory_order_release);
                return {return_status_t::done, nullptr};
            }
        }
        else
        {
            void* buffer_to_free = iovec.piggy_back.address;
            LCI_comp_t completion =
                device_p->completion_manager_p->send->alloc_completion();
            // In order to keep the send_connection object from being
            // deallocated. We have to allocate a shared_ptr in the heap
            // and pass a pointer to shared_ptr to LCI.
            // We will get this pointer back via the send completion queue
            // after this send completes.
            state.store(connection_state::locked, std::memory_order_relaxed);
            ret = LCI_putva(device_p->endpoint_new, iovec, completion, dst_rank,
                0, LCI_DEFAULT_COMP_REMOTE, sharedPtr_p);
            // After this point, if ret == OK, this object can be shared by
            // two threads (the sending thread and the thread polling the
            // completion queue). Care must be taken to avoid data race.
            if (ret == LCI_OK)
            {
                free(buffer_to_free);
                state.store(next_state, std::memory_order_release);
                return {return_status_t::wait, completion};
            }
            else
            {
                state.store(current_state, std::memory_order_release);
            }
        }
        return {return_status_t::retry, nullptr};
    }

    void sender_connection_putva::cleanup()
    {
        if (!is_eager)
        {
            HPX_ASSERT(iovec.count > 0);
            for (int i = 0; i < iovec.count; ++i)
            {
                if (iovec.lbuffers[i].segment != LCI_SEGMENT_ALL)
                {
                    LCI_memory_deregister(&iovec.lbuffers[i].segment);
                }
            }
            free(iovec.lbuffers);
        }
        error_code ec;
        handler_(ec);
        handler_.reset();
        buffer_.clear();
    }

    void sender_connection_putva::done()
    {
        profile_end_hook();
        if (!is_eager)
        {
            cleanup();
        }
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
        data_point_.time_ =
            hpx::chrono::high_resolution_clock::now() - data_point_.time_;
        pp_->add_sent_data(data_point_);
#endif
        util::lci_environment::pcounter_add(
            util::lci_environment::send_conn_timer,
            util::lci_environment::pcounter_since(conn_start_time));

        if (postprocess_handler_)
        {
            // Return this connection to the connection cache.
            // After postprocess_handler is invoked, this connection can be
            // obtained by another thread.
            // so make sure to call this at the very end.
            hpx::move_only_function<void(error_code const&,
                parcelset::locality const&,
                std::shared_ptr<sender_connection_base>)>
                postprocess_handler;
            std::swap(postprocess_handler, postprocess_handler_);
            error_code ec2;
            postprocess_handler(ec2, there_, shared_from_this());
        }
    }

    bool sender_connection_putva::tryMerge(
        const std::shared_ptr<sender_connection_base>& other_base)
    {
        std::shared_ptr<sender_connection_putva> other =
            std::dynamic_pointer_cast<sender_connection_putva>(other_base);
        HPX_ASSERT(other);
        if (!isEager() || !other->isEager())
        {
            // we can only merge eager messages
            return false;
        }
        if (mbuffer.length + other->mbuffer.length > (size_t) LCI_MEDIUM_SIZE)
        {
            // The sum of two messages are too large
            return false;
        }
        // can merge
        memcpy((char*) mbuffer.address + mbuffer.length, other->mbuffer.address,
            other->mbuffer.length);
        mbuffer.length += other->mbuffer.length;
        LCI_mbuffer_free(other->mbuffer);
        //        merged_connections.push_back(other);
        return true;
    }
}    // namespace hpx::parcelset::policies::lci

#endif
