//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)
#include <hpx/assert.hpp>
#include <hpx/modules/mpi_base.hpp>
#include <hpx/parcelport_mpi/header.hpp>
#include <hpx/parcelset/decode_parcels.hpp>
#include <hpx/parcelset/parcel_buffer.hpp>
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
#include <hpx/modules/timing.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::mpi {

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
            sent_release_tag
        };

        using data_type = std::vector<char>;
        using buffer_type =
            parcel_buffer<data_type, serialization::serialization_chunk>;

    public:
        receiver_connection(int src, header const& h, Parcelport& pp) noexcept
          : state_(initialized)
          , src_(src)
          , tag_(h.tag())
          , header_(h)
          , request_(MPI_REQUEST_NULL)
          , request_ptr_(nullptr)
          , chunks_idx_(0)
          , zero_copy_chunks_idx_(0)
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

            case sent_release_tag:
                return done();

            default:
                HPX_ASSERT(false);
            }
            return false;
        }

        bool receive_transmission_chunks(std::size_t num_thread = -1)
        {
            // determine the size of the chunk buffer
            auto const num_zero_copy_chunks = static_cast<std::size_t>(
                static_cast<std::uint32_t>(buffer_.num_chunks_.first));
            auto const num_non_zero_copy_chunks = static_cast<std::size_t>(
                static_cast<std::uint32_t>(buffer_.num_chunks_.second));
            buffer_.transmission_chunks_.resize(
                num_zero_copy_chunks + num_non_zero_copy_chunks);
            if (num_zero_copy_chunks != 0)
            {
                buffer_.chunks_.resize(num_zero_copy_chunks);
                if (!pp_.allow_zero_copy_receive_optimizations())
                {
                    chunk_buffers_.resize(num_zero_copy_chunks);
                }

                {
                    util::mpi_environment::scoped_lock l;

                    [[maybe_unused]] int const ret = MPI_Irecv(
                        buffer_.transmission_chunks_.data(),
                        static_cast<int>(buffer_.transmission_chunks_.size() *
                            sizeof(buffer_type::transmission_chunk_type)),
                        MPI_BYTE, src_, tag_,
                        util::mpi_environment::communicator(), &request_);
                    HPX_ASSERT_LOCKED(l, ret == MPI_SUCCESS);

                    request_ptr_ = &request_;
                }
            }

            state_ = rcvd_transmission_chunks;

            return receive_data(num_thread);
        }

        bool receive_data(std::size_t num_thread = -1)
        {
            if (!request_done())
            {
                return false;
            }

            if (char const* piggy_back = header_.piggy_back())
            {
                std::memcpy(
                    buffer_.data_.data(), piggy_back, buffer_.data_.size());
            }
            else
            {
                util::mpi_environment::scoped_lock l;

                [[maybe_unused]] int const ret = MPI_Irecv(buffer_.data_.data(),
                    static_cast<int>(buffer_.data_.size()), MPI_BYTE, src_,
                    tag_, util::mpi_environment::communicator(), &request_);
                HPX_ASSERT_LOCKED(l, ret == MPI_SUCCESS);

                request_ptr_ = &request_;
            }

            state_ = rcvd_data;

            return receive_chunks(num_thread);
        }

        bool receive_chunks(std::size_t num_thread = -1)
        {
            if (pp_.allow_zero_copy_receive_optimizations())
            {
                if (!request_done())
                {
                    return false;
                }

                // handle zero-copy receive, this should be done on the first entry
                // to receive_chunks only
                if (parcels_.empty())
                {
                    HPX_ASSERT(zero_copy_chunks_idx_ == 0);

                    auto const num_zero_copy_chunks = static_cast<std::size_t>(
                        static_cast<std::uint32_t>(buffer_.num_chunks_.first));
                    if (num_zero_copy_chunks != 0)
                    {
                        HPX_ASSERT(
                            buffer_.chunks_.size() == num_zero_copy_chunks);

                        // De-serialize the parcels such that all data but the
                        // zero-copy chunks are in place. This de-serialization also
                        // allocates all zero-chunk buffers and stores those in the
                        // chunks array for the subsequent networking to place the
                        // received data directly.
                        for (std::size_t i = 0; i != num_zero_copy_chunks; ++i)
                        {
                            auto const chunk_size = static_cast<std::size_t>(
                                buffer_.transmission_chunks_[i].second);
                            buffer_.chunks_[i] =
                                serialization::create_pointer_chunk(
                                    nullptr, chunk_size);
                        }

                        parcels_ =
                            decode_parcels_zero_copy(pp_, buffer_, num_thread);

                        // note that at this point, buffer_.chunks_ will have
                        // entries for all chunks, including the non-zero-copy ones
                    }

                    // we should have received at least one parcel if there are
                    // zero-copy chunks to be received
                    HPX_ASSERT(parcels_.empty() || !buffer_.chunks_.empty());
                }

                while (chunks_idx_ != buffer_.chunks_.size())
                {
                    if (!request_done())
                    {
                        return false;
                    }

                    auto& c = buffer_.chunks_[chunks_idx_++];
                    if (c.type_ == serialization::chunk_type::chunk_type_index)
                    {
                        continue;    // skip non-zero-copy chunks
                    }

                    // the zero-copy chunks come first in the transmission_chunks_
                    // array
                    auto const chunk_size = static_cast<std::size_t>(
                        buffer_.transmission_chunks_[zero_copy_chunks_idx_++]
                            .second);

                    // the parcel de-serialization above should have allocated the
                    // correct amount of memory
                    HPX_ASSERT_MSG(
                        c.data() != nullptr && c.size() == chunk_size,
                        "zero-copy chunk buffers should have been initialized "
                        "during de-serialization");

                    {
                        util::mpi_environment::scoped_lock l;

                        [[maybe_unused]] int const ret = MPI_Irecv(c.data(),
                            static_cast<int>(chunk_size), MPI_BYTE, src_, tag_,
                            util::mpi_environment::communicator(), &request_);
                        HPX_ASSERT_LOCKED(l, ret == MPI_SUCCESS);

                        request_ptr_ = &request_;
                    }
                }
                HPX_ASSERT_MSG(
                    zero_copy_chunks_idx_ == buffer_.num_chunks_.first,
                    "observed: {}, expected {}", zero_copy_chunks_idx_,
                    buffer_.num_chunks_.first);
            }
            else
            {
                HPX_ASSERT(chunk_buffers_.size() == buffer_.chunks_.size());
                while (chunks_idx_ < buffer_.chunks_.size())
                {
                    if (!request_done())
                    {
                        return false;
                    }

                    std::size_t const idx = chunks_idx_++;
                    std::size_t const chunk_size =
                        buffer_.transmission_chunks_[idx].second;

                    auto& c = chunk_buffers_[idx];
                    c.resize(chunk_size);

                    // store buffer for decode_parcels below
                    buffer_.chunks_[idx] = serialization::create_pointer_chunk(
                        c.data(), chunk_size);

                    {
                        util::mpi_environment::scoped_lock l;

                        [[maybe_unused]] int const ret = MPI_Irecv(c.data(),
                            static_cast<int>(c.size()), MPI_BYTE, src_, tag_,
                            util::mpi_environment::communicator(), &request_);
                        HPX_ASSERT_LOCKED(l, ret == MPI_SUCCESS);

                        request_ptr_ = &request_;
                    }
                }
            }
            state_ = rcvd_chunks;

            return send_release_tag(num_thread);
        }

        bool send_release_tag(std::size_t num_thread = -1)
        {
            if (!request_done())
            {
                return false;
            }

#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
            parcelset::data_point& data = buffer_.data_point_;
            data.time_ = timer_.elapsed_nanoseconds() - data.time_;
#endif
            {
                util::mpi_environment::scoped_lock l;

                [[maybe_unused]] int const ret = MPI_Isend(&tag_, 1, MPI_INT,
                    src_, 1, util::mpi_environment::communicator(), &request_);
                HPX_ASSERT_LOCKED(l, ret == MPI_SUCCESS);

                request_ptr_ = &request_;
            }

            if (parcels_.empty())
            {
                // decode and handle received data
                HPX_ASSERT(buffer_.num_chunks_.first == 0 ||
                    !pp_.allow_zero_copy_receive_optimizations());
                handle_received_parcels(
                    decode_parcels(pp_, HPX_MOVE(buffer_), num_thread),
                    num_thread);
                chunk_buffers_.clear();
            }
            else
            {
                // handle the received zero-copy parcels.
                HPX_ASSERT(buffer_.num_chunks_.first != 0 &&
                    pp_.allow_zero_copy_receive_optimizations());
                handle_received_parcels(HPX_MOVE(parcels_));
                buffer_ = buffer_type{};
            }

            state_ = sent_release_tag;

            return done();
        }

        bool done() noexcept
        {
            return request_done();
        }

        bool request_done() noexcept
        {
            if (request_ptr_ == nullptr)
            {
                return true;
            }

            util::mpi_environment::scoped_try_lock l;
            if (!l.locked)
            {
                return false;
            }

            int completed = 0;
            [[maybe_unused]] int const ret =
                MPI_Test(request_ptr_, &completed, MPI_STATUS_IGNORE);
            HPX_ASSERT_LOCKED(l, ret == MPI_SUCCESS);

            if (completed)
            {
                request_ptr_ = nullptr;
                return true;
            }
            return false;
        }

#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
        hpx::chrono::high_resolution_timer timer_;
#endif
        connection_state state_;

        int src_;
        int tag_;
        header header_;
        buffer_type buffer_;

        MPI_Request request_;
        MPI_Request* request_ptr_;
        std::size_t chunks_idx_;
        std::size_t zero_copy_chunks_idx_;

        Parcelport& pp_;

        std::vector<parcelset::parcel> parcels_;
        std::vector<std::vector<char>> chunk_buffers_;
    };
}    // namespace hpx::parcelset::policies::mpi

#endif
