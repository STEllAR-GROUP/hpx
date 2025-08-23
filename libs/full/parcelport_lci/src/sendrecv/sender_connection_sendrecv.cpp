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

#include <hpx/modules/lci_base.hpp>
#include <hpx/parcelport_lci/header.hpp>
#include <hpx/parcelport_lci/helper.hpp>
#include <hpx/parcelport_lci/locality.hpp>
#include <hpx/parcelport_lci/parcelport_lci.hpp>
#include <hpx/parcelport_lci/receiver_base.hpp>
#include <hpx/parcelport_lci/sendrecv/sender_connection_sendrecv.hpp>

#include <hpx/assert.hpp>

#include <atomic>
#include <memory>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lci {

    std::atomic<int> sender_connection_sendrecv::next_tag = 0;

    void sender_connection_sendrecv::load(
        sender_connection_sendrecv::handler_type&& handler,
        sender_connection_sendrecv::postprocess_handler_type&&
            parcel_postprocess)
    {
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
        timer_.restart();
#endif
        conn_start_time = util::lci_environment::pcounter_now();
        HPX_ASSERT(!handler_);
        HPX_ASSERT(!postprocess_handler_);
        HPX_ASSERT(!buffer_.data_.empty());
        handler_ = HPX_MOVE(handler);
        postprocess_handler_ = HPX_MOVE(parcel_postprocess);

        // build header
        if (config_t::enable_in_buffer_assembly)
        {
            int retry_count = 0;
            while (
                LCI_mbuffer_alloc(device_p->device, &header_buffer) != LCI_OK)
            {
                if (config_t::bg_work_when_send)
                    pp_->do_background_work(0, parcelport_background_mode::all);
                yield_k(retry_count, config_t::mbuffer_alloc_max_retry);
            }
            HPX_ASSERT(header_buffer.length == (size_t) LCI_MEDIUM_SIZE);
            header_ = header(
                buffer_, (char*) header_buffer.address, header_buffer.length);
            header_buffer.length = header_.size();
        }
        else
        {
            header_buffer_vector.resize(
                header::get_header_size(buffer_, LCI_MEDIUM_SIZE));
            header_ =
                header(buffer_, static_cast<char*>(header_buffer_vector.data()),
                    header_buffer_vector.size());
        }
        HPX_ASSERT((header_.num_zero_copy_chunks() == 0) ==
            buffer_.transmission_chunks_.empty());
        need_send_data = false;
        need_send_tchunks = false;
        // Calculate how many sends we need to make
        int num_send = 0;    // header doesn't need to do tag matching
        if (header_.piggy_back_data() == nullptr)
        {
            need_send_data = true;
            ++num_send;
        }
        if (header_.num_zero_copy_chunks() != 0)
        {
            if (header_.piggy_back_tchunk() == nullptr)
            {
                need_send_tchunks = true;
                ++num_send;
            }
            num_send += header_.num_zero_copy_chunks();
        }
        tag = 0;    // If no need to post send, then tag can be ignored.
        sharedPtr_p = nullptr;
        if (config_t::enable_sendmc || num_send > 0)
        {
            sharedPtr_p = new std::shared_ptr<sender_connection_sendrecv>(
                std::dynamic_pointer_cast<sender_connection_sendrecv>(
                    shared_from_this()));
            if (num_send > 0)
                tag = next_tag.fetch_add(num_send) % LCI_MAX_TAG;
        }
        if ((int) tag <= LCI_MAX_TAG && (int) tag + num_send > LCI_MAX_TAG)
            util::lci_environment::log(
                util::lci_environment::log_level_t::debug, "tag",
                "Rank %d Wrap around!\n", LCI_RANK);
        header_.set_device_idx(device_p->idx);
        header_.set_tag(tag);
        if (!config_t::enable_in_buffer_assembly)
        {
            int retry_count = 0;
            while (
                LCI_mbuffer_alloc(device_p->device, &header_buffer) != LCI_OK)
            {
                if (config_t::bg_work_when_send)
                    pp_->do_background_work(0, parcelport_background_mode::all);
                yield_k(retry_count, config_t::mbuffer_alloc_max_retry);
            }
            memcpy(header_buffer.address, header_buffer_vector.data(),
                header_buffer_vector.size());
            header_buffer.length = header_buffer_vector.size();
        }
        send_chunks_idx = 0;
        completion = nullptr;
        segment_to_use = LCI_SEGMENT_ALL;
        segment_used = LCI_SEGMENT_ALL;
        // set state
        profile_start_hook(header_);
        state.store(connection_state::initialized, std::memory_order_release);
        original_tag = tag;
        util::lci_environment::log(util::lci_environment::log_level_t::debug,
            "send",
            "send connection (%d, %d, %d, %d) tchunks "
            "%d data %d chunks %d start!\n",
            LCI_RANK, dst_rank, original_tag, num_send,
            header_.piggy_back_tchunk() != nullptr,
            header_.piggy_back_data() != nullptr,
            header_.num_zero_copy_chunks());
    }

    sender_connection_sendrecv::return_t sender_connection_sendrecv::send_nb()
    {
        while (
            state.load(std::memory_order_acquire) == connection_state::locked)
        {
            continue;
        }

        switch (state.load(std::memory_order_acquire))
        {
        case connection_state::initialized:
            return send_header();

        case connection_state::sent_header:
            return send_transmission_chunks();

        case connection_state::sent_transmission_chunks:
            return send_data();

        case connection_state::sent_data:
            return send_chunks();

        case connection_state::sent_chunks:
            return {return_status_t::done, nullptr};

        default:
            throw std::runtime_error("Unexpected send state!");
        }
    }

    sender_connection_sendrecv::return_t
    sender_connection_sendrecv::send_header()
    {
        const auto current_state = connection_state::initialized;
        const auto next_state = connection_state::sent_header;
        HPX_ASSERT(state.load(std::memory_order_acquire) == current_state);
        HPX_UNUSED(current_state);
        LCI_error_t ret;
        if (config_t::enable_sendmc)
        {
            if (completion == nullptr)
            {
                completion =
                    device_p->completion_manager_p->send->alloc_completion();
            }
            state.store(connection_state::locked, std::memory_order_relaxed);
        }
        if (config_t::protocol == config_t::protocol_t::putsendrecv)
        {
            ret = LCI_putmac(device_p->endpoint_new, header_buffer, dst_rank, 0,
                LCI_DEFAULT_COMP_REMOTE,
                config_t::enable_sendmc ? completion : nullptr, sharedPtr_p);
        }
        else
        {
            HPX_ASSERT(config_t::protocol == config_t::protocol_t::sendrecv);
            ret = LCI_sendmc(device_p->endpoint_new, header_buffer, dst_rank, 0,
                config_t::enable_sendmc ? completion : nullptr, sharedPtr_p);
        }
        if (ret == LCI_OK)
        {
            util::lci_environment::log(
                util::lci_environment::log_level_t::debug, "send",
                "%s (%d, %d, %d) length %lu\n",
                config_t::protocol == config_t::protocol_t::putsendrecv ?
                    "LCI_putmna" :
                    "LCI_sendmn",
                LCI_RANK, dst_rank, tag, header_buffer.length);
            if (config_t::enable_sendmc)
            {
                auto ret_comp = completion;
                completion = nullptr;
                state.store(next_state, std::memory_order_release);
                return {return_status_t::wait, ret_comp};
            }
            else
            {
                state.store(next_state, std::memory_order_release);
                return send_transmission_chunks();
            }
        }
        else
        {
            HPX_ASSERT(ret == LCI_ERR_RETRY);
            if (config_t::enable_sendmc)
                state.store(current_state, std::memory_order_release);
            return {return_status_t::retry, nullptr};
        }
    }

    sender_connection_sendrecv::return_t
    sender_connection_sendrecv::unified_followup_send(
        void* address, size_t length)
    {
        if (length <= (size_t) LCI_MEDIUM_SIZE)
        {
            LCI_mbuffer_t buffer;
            buffer.address = address;
            buffer.length = length;
            if (config_t::enable_sendmc && completion == nullptr)
            {
                completion =
                    device_p->completion_manager_p->send->alloc_completion();
            }
            LCI_error_t ret = LCI_sendmc(device_p->endpoint_followup, buffer,
                dst_rank, tag, config_t::enable_sendmc ? completion : nullptr,
                sharedPtr_p);
            if (ret == LCI_OK)
            {
                util::lci_environment::log(
                    util::lci_environment::log_level_t::debug, "send",
                    "sendm (%d, %d, %d) device %d tag %d size %d\n", LCI_RANK,
                    dst_rank, original_tag, device_p->idx, tag, length);
                tag = (tag + 1) % LCI_MAX_TAG;
                if (config_t::enable_sendmc)
                {
                    auto ret_comp = completion;
                    completion = nullptr;
                    return {return_status_t::wait, ret_comp};
                }
                else
                    return {return_status_t::done, nullptr};
            }
            else
            {
                HPX_ASSERT(ret == LCI_ERR_RETRY);
                return {return_status_t::retry, nullptr};
            }
        }
        else
        {
            if (config_t::reg_mem && segment_to_use == LCI_SEGMENT_ALL)
            {
                LCI_memory_register(
                    device_p->device, address, length, &segment_to_use);
            }
            if (completion == nullptr)
            {
                completion =
                    device_p->completion_manager_p->send->alloc_completion();
            }
            LCI_lbuffer_t buffer;
            buffer.segment = segment_to_use;
            buffer.address = address;
            buffer.length = length;
            LCI_error_t ret = LCI_sendl(device_p->endpoint_followup, buffer,
                dst_rank, tag, completion, sharedPtr_p);
            if (ret == LCI_OK)
            {
                util::lci_environment::log(
                    util::lci_environment::log_level_t::debug, "send",
                    "sendl (%d, %d, %d) device %d, tag %d size %d\n", LCI_RANK,
                    dst_rank, original_tag, device_p->idx, tag, length);
                tag = (tag + 1) % LCI_MAX_TAG;
                if (segment_used != LCI_SEGMENT_ALL)
                {
                    LCI_memory_deregister(&segment_used);
                    segment_used = LCI_SEGMENT_ALL;
                }
                segment_used = segment_to_use;
                segment_to_use = LCI_SEGMENT_ALL;
                auto ret_comp = completion;
                completion = nullptr;
                return {return_status_t::wait, ret_comp};
            }
            else
            {
                HPX_ASSERT(ret == LCI_ERR_RETRY);
                return {return_status_t::retry, nullptr};
            }
        }
    }

    sender_connection_sendrecv::return_t
    sender_connection_sendrecv::send_transmission_chunks()
    {
        const auto current_state = connection_state::sent_header;
        const auto next_state = connection_state::sent_transmission_chunks;
        HPX_ASSERT(state.load(std::memory_order_acquire) == current_state);
        if (!need_send_tchunks)
        {
            state.store(next_state, std::memory_order_release);
            return send_data();
        }

        std::vector<typename parcel_buffer_type::transmission_chunk_type>&
            tchunks = buffer_.transmission_chunks_;
        size_t tchunks_size = tchunks.size() *
            sizeof(parcel_buffer_type::transmission_chunk_type);
        state.store(connection_state::locked, std::memory_order_relaxed);
        auto ret = unified_followup_send(tchunks.data(), tchunks_size);
        if (ret.status == return_status_t::done)
        {
            state.store(next_state, std::memory_order_release);
            return send_data();
        }
        else if (ret.status == return_status_t::retry)
        {
            state.store(current_state, std::memory_order_release);
            return ret;
        }
        else
        {
            state.store(next_state, std::memory_order_release);
            return ret;
        }
    }

    sender_connection_sendrecv::return_t sender_connection_sendrecv::send_data()
    {
        const auto current_state = connection_state::sent_transmission_chunks;
        const auto next_state = connection_state::sent_data;
        HPX_ASSERT(state.load(std::memory_order_acquire) == current_state);
        if (!need_send_data)
        {
            state.store(next_state, std::memory_order_release);
            return send_chunks();
        }
        state.store(connection_state::locked, std::memory_order_relaxed);
        auto ret =
            unified_followup_send(buffer_.data_.data(), buffer_.data_.size());
        if (ret.status == return_status_t::done)
        {
            state.store(next_state, std::memory_order_release);
            return send_chunks();
        }
        else if (ret.status == return_status_t::retry)
        {
            state.store(current_state, std::memory_order_release);
            return ret;
        }
        else
        {
            state.store(next_state, std::memory_order_release);
            return ret;
        }
    }

    sender_connection_sendrecv::return_t
    sender_connection_sendrecv::send_chunks()
    {
        const auto current_state = connection_state::sent_data;
        const auto next_state = connection_state::sent_chunks;
        HPX_ASSERT(state.load(std::memory_order_acquire) == current_state);

        while (send_chunks_idx < buffer_.chunks_.size())
        {
            serialization::serialization_chunk& chunk =
                buffer_.chunks_[send_chunks_idx];
            if (chunk.type_ == serialization::chunk_type::chunk_type_pointer ||
                chunk.type_ ==
                    serialization::chunk_type::chunk_type_const_pointer)
            {
                state.store(
                    connection_state::locked, std::memory_order_relaxed);
                auto ret = unified_followup_send(
                    const_cast<void*>(chunk.data_.cpos_), chunk.size_);
                if (ret.status == return_status_t::done)
                {
                    ++send_chunks_idx;
                    state.store(current_state, std::memory_order_release);
                    continue;
                }
                else if (ret.status == return_status_t::retry)
                {
                    state.store(current_state, std::memory_order_release);
                    return ret;
                }
                else
                {
                    ++send_chunks_idx;
                    state.store(current_state, std::memory_order_release);
                    return ret;
                }
            }
            else
            {
                ++send_chunks_idx;
            }
        }

        state.store(next_state, std::memory_order_release);
        return {return_status_t::done, nullptr};
    }

    void sender_connection_sendrecv::done()
    {
        util::lci_environment::log(util::lci_environment::log_level_t::debug,
            "send", "send connection (%d, %d, %d, %d) done!\n", LCI_RANK,
            dst_rank, original_tag, tag - original_tag + 1);
        profile_end_hook();
        error_code ec;
        handler_(ec);
        handler_.reset();
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
        data_point_.time_ = timer_.elapsed_nanoseconds();
        pp_->add_sent_data(data_point_);
#endif
        if (segment_used != LCI_SEGMENT_ALL)
        {
            LCI_memory_deregister(&segment_used);
            segment_used = LCI_SEGMENT_ALL;
        }
        if (config_t::enable_sendmc)
        {
            LCI_mbuffer_free(header_buffer);
        }
        HPX_ASSERT(completion == nullptr);
        HPX_ASSERT(segment_to_use == LCI_SEGMENT_ALL);
        buffer_.clear();
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

    bool sender_connection_sendrecv::tryMerge(
        const std::shared_ptr<sender_connection_base>& other_base)
    {
        // We cannot merge any message here.
        HPX_UNUSED(other_base);
        return false;
    }
}    // namespace hpx::parcelset::policies::lci

#endif
