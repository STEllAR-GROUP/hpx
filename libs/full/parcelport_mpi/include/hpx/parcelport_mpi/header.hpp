//  Copyright (c) 2013-2024 Hartmut Kaiser
//  Copyright (c) 2013-2015 Thomas Heller
//  Copyright (c)      2023 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)
#include <hpx/assert.hpp>

#include <hpx/parcelset/parcel_buffer.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace hpx::parcelset::policies::mpi {

    struct header
    {
        using value_type = int;
        enum data_pos
        {
            // signature for assert_valid
            pos_signature = 0,
            // tag
            pos_tag = 1 * sizeof(value_type),
            // enable ack handshakes
            pos_ack_handshakes = 2 * sizeof(value_type),
            // non-zero-copy chunk size
            pos_numbytes_nonzero_copy = 3 * sizeof(value_type),
            // transmission chunk size
            pos_numbytes_tchunk = 4 * sizeof(value_type),
            // how many bytes in total (including zero-copy and non-zero-copy chunks)
            pos_numbytes = 5 * sizeof(value_type),
            // zero-copy chunk number
            pos_numchunks_zero_copy = 6 * sizeof(value_type),
            // non-zero-copy chunk number
            pos_numchunks_nonzero_copy = 7 * sizeof(value_type),
            // whether piggyback data
            pos_piggy_back_flag_data = 8 * sizeof(value_type),
            // whether piggyback transmission chunk
            pos_piggy_back_flag_tchunk = 8 * sizeof(value_type) + 1,
            pos_piggy_back_address = 8 * sizeof(value_type) + 2
        };

        template <typename buffer_type, typename ChunkType>
        static size_t get_header_size(
            parcel_buffer<buffer_type, ChunkType> const& buffer,
            size_t max_header_size) noexcept
        {
            HPX_ASSERT(max_header_size >= pos_piggy_back_address);

            size_t current_header_size = pos_piggy_back_address;
            if (buffer.data_.size() <= (max_header_size - current_header_size))
            {
                current_header_size += buffer.data_.size();
            }

            int const num_zero_copy_chunks = buffer.num_chunks_.first;
            [[maybe_unused]] int const num_non_zero_copy_chunks =
                buffer.num_chunks_.second;
            if (num_zero_copy_chunks != 0)
            {
                HPX_ASSERT(buffer.transmission_chunks_.size() ==
                    static_cast<size_t>(
                        num_zero_copy_chunks + num_non_zero_copy_chunks));
                int const tchunk_size =
                    static_cast<int>(buffer.transmission_chunks_.size() *
                        sizeof(typename parcel_buffer<buffer_type,
                            ChunkType>::transmission_chunk_type));
                if (tchunk_size <=
                    static_cast<int>(max_header_size - current_header_size))
                {
                    current_header_size += tchunk_size;
                }
            }
            return current_header_size;
        }

        template <typename buffer_type, typename ChunkType>
        header(parcel_buffer<buffer_type, ChunkType> const& buffer,
            char* header_buffer, size_t max_header_size) noexcept
        {
            HPX_ASSERT(max_header_size >= pos_piggy_back_address);
            data_ = header_buffer;
            memset(data_, 0, pos_piggy_back_address);

            std::int64_t const size =
                static_cast<std::int64_t>(buffer.data_.size());
            std::int64_t const numbytes =
                static_cast<std::int64_t>(buffer.data_size_);
            HPX_ASSERT(size <= (std::numeric_limits<value_type>::max)());
            HPX_ASSERT(numbytes <= (std::numeric_limits<value_type>::max)());

            int const num_zero_copy_chunks = buffer.num_chunks_.first;
            int const num_non_zero_copy_chunks = buffer.num_chunks_.second;

            set(pos_signature, MAGIC_SIGNATURE);
            set(pos_numbytes_nonzero_copy, static_cast<value_type>(size));
            set(pos_numbytes, static_cast<value_type>(numbytes));
            set(pos_numchunks_zero_copy, num_zero_copy_chunks);
            set(pos_numchunks_nonzero_copy, num_non_zero_copy_chunks);
            data_[pos_piggy_back_flag_data] = 0;
            data_[pos_piggy_back_flag_tchunk] = 0;

            size_t current_header_size = pos_piggy_back_address;
            if (buffer.data_.size() <= (max_header_size - current_header_size))
            {
                data_[pos_piggy_back_flag_data] = 1;
                std::memcpy(
                    &data_[current_header_size], &buffer.data_[0], size);
                current_header_size += size;
            }

            if (num_zero_copy_chunks != 0)
            {
                HPX_ASSERT(buffer.transmission_chunks_.size() ==
                    static_cast<size_t>(
                        num_zero_copy_chunks + num_non_zero_copy_chunks));
                int const tchunk_size =
                    static_cast<int>(buffer.transmission_chunks_.size() *
                        sizeof(typename parcel_buffer<buffer_type,
                            ChunkType>::transmission_chunk_type));
                set(pos_numbytes_tchunk, static_cast<value_type>(tchunk_size));
                if (tchunk_size <=
                    static_cast<int>(max_header_size - current_header_size))
                {
                    data_[pos_piggy_back_flag_tchunk] = 1;
                    std::memcpy(&data_[current_header_size],
                        buffer.transmission_chunks_.data(), tchunk_size);
                }
            }
        }

        header() noexcept
        {
            data_ = nullptr;
        }

        explicit header(char* header_buffer) noexcept
        {
            data_ = header_buffer;
        }

        void reset() noexcept
        {
            data_ = nullptr;
        }

        [[nodiscard]] bool valid() const noexcept
        {
            return data_ != nullptr && signature() == MAGIC_SIGNATURE;
        }

        void assert_valid() const noexcept
        {
            HPX_ASSERT(valid());
        }

        [[nodiscard]] char* data() const noexcept
        {
            return data_;
        }

        [[nodiscard]] size_t size() const noexcept
        {
            return pos_piggy_back_address + piggy_back_size();
        }

        [[nodiscard]] value_type signature() const noexcept
        {
            return get(pos_signature);
        }

        void set_tag(int tag) noexcept
        {
            set(pos_tag, static_cast<value_type>(tag));
        }

        void set_ack_handshakes(bool enable_ack_handshakes) noexcept
        {
            set(pos_ack_handshakes, enable_ack_handshakes ? 1 : 0);
        }

        [[nodiscard]] value_type get_tag() const noexcept
        {
            return get(pos_tag);
        }

        [[nodiscard]] value_type get_ack_handshakes() const noexcept
        {
            return get(pos_ack_handshakes);
        }

        [[nodiscard]] value_type numbytes_nonzero_copy() const noexcept
        {
            return get(pos_numbytes_nonzero_copy);
        }

        [[nodiscard]] value_type numbytes_tchunk() const noexcept
        {
            return get(pos_numbytes_tchunk);
        }

        [[nodiscard]] value_type numbytes() const noexcept
        {
            return get(pos_numbytes);
        }

        [[nodiscard]] value_type num_zero_copy_chunks() const noexcept
        {
            return get(pos_numchunks_zero_copy);
        }

        [[nodiscard]] value_type num_non_zero_copy_chunks() const noexcept
        {
            return get(pos_numchunks_nonzero_copy);
        }

        [[nodiscard]] constexpr char* piggy_back_address() const noexcept
        {
            if (data_[pos_piggy_back_flag_data] ||
                data_[pos_piggy_back_flag_tchunk])
                return &data_[pos_piggy_back_address];
            return nullptr;
        }

        [[nodiscard]] int piggy_back_size() const noexcept
        {
            int result = 0;
            if (data_[pos_piggy_back_flag_data])
                result += numbytes_nonzero_copy();
            if (data_[pos_piggy_back_flag_tchunk])
                result += numbytes_tchunk();
            return result;
        }

        [[nodiscard]] constexpr char* piggy_back_data() const noexcept
        {
            if (data_[pos_piggy_back_flag_data])
                return &data_[pos_piggy_back_address];
            return nullptr;
        }

        [[nodiscard]] constexpr char* piggy_back_tchunk() const noexcept
        {
            size_t current_header_size = pos_piggy_back_address;
            if (!data_[pos_piggy_back_flag_tchunk])
                return nullptr;
            if (data_[pos_piggy_back_flag_data])
                current_header_size += numbytes_nonzero_copy();
            return &data_[current_header_size];
        }

    private:
        // random magic number for assert_valid
        static constexpr int MAGIC_SIGNATURE = 19527;
        char* data_;

        constexpr void set(std::size_t Pos, value_type const& t) noexcept
        {
            *hpx::bit_cast<value_type*>(&data_[Pos]) = t;
        }

        [[nodiscard]] constexpr value_type get(std::size_t Pos) const noexcept
        {
            return *hpx::bit_cast<value_type const*>(&data_[Pos]);
        }
    };
}    // namespace hpx::parcelset::policies::mpi

#endif
