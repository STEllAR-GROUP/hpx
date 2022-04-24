//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)
#include <hpx/assert.hpp>

#include <hpx/parcelport_lci/header.hpp>
#include <hpx/parcelset/decode_parcels.hpp>
#include <hpx/parcelset/parcel_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lci {
    template <typename Parcelport>
    struct receiver_connection
    {
    private:
        enum connection_state
        {
            initialized,
            rcvd_transmission_chunks,
            rcvd_data,
            rcvd_chunks,
        };

        using data_type = std::vector<char>;
        using buffer_type = parcel_buffer<data_type, data_type>;

    public:
        receiver_connection(int src, header h, Parcelport& pp) noexcept
          : state_(initialized)
          , src_rank(src)
          , tag_(h.tag())
          , header_(h)
          , sync_tchunks(nullptr)
          , sync_others(nullptr)
          , chunks_idx_(0)
          , pp_(pp)
        {
            header_.assert_valid();

#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            parcelset::data_point& data = buffer_.data_point_;
            data.time_ = timer_.elapsed_nanoseconds();
            data.bytes_ = static_cast<std::size_t>(header_.numbytes());
#endif
            buffer_.data_.resize(static_cast<std::size_t>(header_.size()));
            buffer_.num_chunks_ = header_.num_chunks();

            // calculate how many long messages to recv
            int num_zero_copy_chunks =
                static_cast<int>(buffer_.num_chunks_.first);
            if (num_zero_copy_chunks != 0)
                LCI_sync_create(LCI_UR_DEVICE, 1, &sync_tchunks);
            else
                sync_tchunks = nullptr;

            int recv_num = 0;
            // data
            if (!header_.piggy_back())
                ++recv_num;
            // chunks
            recv_num += num_zero_copy_chunks;
            // create synchronizer
            if (recv_num > 0)
                LCI_sync_create(LCI_UR_DEVICE, recv_num, &sync_others);
            else
                sync_others = nullptr;
        }

        bool receive(std::size_t num_thread = -1)
        {
            switch (state_)
            {
            case initialized:
                return receive_transmission_chunks(num_thread);

            case rcvd_transmission_chunks:
                return receive_data(num_thread);

            case rcvd_data:
                return receive_chunks(num_thread);

            case rcvd_chunks:
                return send_release_tag(num_thread);

            default:
                HPX_ASSERT(false);
            }
            return false;
        }

        bool unified_recv(
            void* buffer, int length, int rank, LCI_tag_t tag, LCI_comp_t sync)
        {
            LCI_error_t ret;
            if (length <= LCI_MEDIUM_SIZE)
            {
                LCI_mbuffer_t mbuffer;
                mbuffer.address = buffer;
                mbuffer.length = length;
                ret = LCI_recvm(util::lci_environment::lci_endpoint(), mbuffer,
                    rank, tag, sync, nullptr);
            }
            else
            {
                LCI_lbuffer_t lbuffer;
                lbuffer.address = buffer;
                lbuffer.length = length;
                lbuffer.segment = LCI_SEGMENT_ALL;
                ret = LCI_recvl(util::lci_environment::lci_endpoint(), lbuffer,
                    src_rank, tag_, sync, nullptr);
            }
            return ret == LCI_OK;
        }

        bool receive_transmission_chunks(std::size_t num_thread = -1)
        {
            // determine the size of the chunk buffer
            std::size_t num_zero_copy_chunks = static_cast<std::size_t>(
                static_cast<std::uint32_t>(buffer_.num_chunks_.first));
            std::size_t num_non_zero_copy_chunks = static_cast<std::size_t>(
                static_cast<std::uint32_t>(buffer_.num_chunks_.second));

            auto& tchunks = buffer_.transmission_chunks_;
            tchunks.resize(num_zero_copy_chunks + num_non_zero_copy_chunks);
            if (num_zero_copy_chunks != 0)
            {
                buffer_.chunks_.resize(num_zero_copy_chunks);
                int tchunks_length = static_cast<int>(tchunks.size() *
                    sizeof(buffer_type::transmission_chunk_type));
                bool ret = unified_recv(tchunks.data(), tchunks_length,
                    src_rank, tag_, sync_tchunks);
                if (!ret)
                    return false;
            }
            state_ = rcvd_transmission_chunks;
            return receive_data(num_thread);
        }

        bool receive_data(std::size_t num_thread = -1)
        {
            char* piggy_back = header_.piggy_back();
            if (piggy_back)
            {
                std::memcpy(
                    &buffer_.data_[0], piggy_back, buffer_.data_.size());
            }
            else
            {
                bool ret = unified_recv(buffer_.data_.data(),
                    static_cast<int>(buffer_.data_.size()), src_rank, tag_,
                    sync_others);
                if (!ret)
                    return false;
            }
            state_ = rcvd_data;
            return receive_chunks(num_thread);
        }

        bool receive_chunks(std::size_t num_thread = -1)
        {
            if (sync_tchunks != nullptr)
            {
                LCI_error_t ret = LCI_sync_test(sync_tchunks, nullptr);
                if (ret != LCI_OK)
                    return false;
                LCI_sync_free(&sync_tchunks);
                sync_tchunks = nullptr;
            }
            while (chunks_idx_ < buffer_.chunks_.size())
            {
                std::size_t idx = chunks_idx_;
                std::size_t chunk_size =
                    buffer_.transmission_chunks_[idx].second;

                data_type& c = buffer_.chunks_[idx];
                c.resize(chunk_size);
                {
                    bool ret =
                        unified_recv(c.data(), static_cast<int>(c.size()),
                            src_rank, tag_, sync_others);
                    if (!ret)
                        return false;
                }
                ++chunks_idx_;
            }

            state_ = rcvd_chunks;

            return send_release_tag(num_thread);
        }

        bool send_release_tag(std::size_t num_thread = -1)
        {
            if (sync_others != nullptr)
            {
                LCI_error_t ret = LCI_sync_test(sync_others, nullptr);
                if (ret != LCI_OK)
                    return false;
                LCI_sync_free(&sync_others);
                sync_others = nullptr;
            }
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            parcelset::data_point& data = buffer_.data_point_;
            data.time_ = timer_.elapsed_nanoseconds() - data.time_;
#endif
            {
                LCI_short_t short_rt_;
                *(int*) &short_rt_ = tag_;
                if (LCI_puts(util::lci_environment::rt_endpoint(), short_rt_,
                        src_rank, 1, LCI_DEFAULT_COMP_REMOTE) != LCI_OK)
                {
                    return false;
                }
            }

            decode_parcels(pp_, HPX_MOVE(buffer_), num_thread);

            return true;
        }

#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
        hpx::chrono::high_resolution_timer timer_;
#endif
        connection_state state_;

        int src_rank;
        int tag_;
        header header_;
        buffer_type buffer_;

        LCI_comp_t sync_tchunks;
        LCI_comp_t sync_others;
        std::size_t chunks_idx_;

        Parcelport& pp_;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
