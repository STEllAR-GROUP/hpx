//  Copyright (c) 2013-2023 Hartmut Kaiser
//  Copyright (c) 2013-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)
#include <hpx/assert.hpp>
#include <hpx/modules/type_support.hpp>
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
            pos_tag = 0 * sizeof(value_type),
            pos_size = 1 * sizeof(value_type),
            pos_numbytes = 2 * sizeof(value_type),
            pos_numchunks_first = 3 * sizeof(value_type),
            pos_numchunks_second = 4 * sizeof(value_type),
            pos_piggy_back_flag = 5 * sizeof(value_type),
            pos_piggy_back_data = 5 * sizeof(value_type) + 1
        };

        static constexpr std::int32_t data_size_ = 512;

        template <typename Buffer>
        header(Buffer const& buffer, int tag) noexcept
        {
            init(buffer, tag);
        }

        template <typename Buffer>
        void init(Buffer const& buffer, int tag) noexcept
        {
            auto const size = static_cast<std::int64_t>(buffer.size_);
            auto const numbytes = static_cast<std::int64_t>(buffer.data_size_);

            HPX_ASSERT(size <= (std::numeric_limits<value_type>::max)());
            HPX_ASSERT(numbytes <= (std::numeric_limits<value_type>::max)());

            set(pos_tag, tag);
            set(pos_size, static_cast<value_type>(size));
            set(pos_numbytes, static_cast<value_type>(numbytes));
            set(pos_numchunks_first,
                static_cast<value_type>(buffer.num_chunks_.first));
            set(pos_numchunks_second,
                static_cast<value_type>(buffer.num_chunks_.second));

            if (buffer.data_.size() <= (data_size_ - pos_piggy_back_data))
            {
                data_[pos_piggy_back_flag] = 1;
                std::memcpy(&data_[pos_piggy_back_data], buffer.data_.data(),
                    buffer.data_.size());
            }
            else
            {
                data_[pos_piggy_back_flag] = 0;
            }
        }

        header() noexcept
        {
            reset();
        }

        void reset() noexcept
        {
            std::memset(data_.data(), static_cast<char>(-1), data_size_);
            data_[pos_piggy_back_flag] = 1;
        }

        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return data_[0] != static_cast<char>(-1);
        }

        void assert_valid() const noexcept
        {
            HPX_ASSERT(tag() != -1);
            HPX_ASSERT(size() != -1);
            HPX_ASSERT(numbytes() != -1);
            HPX_ASSERT(num_chunks().first != -1);
            HPX_ASSERT(num_chunks().second != -1);
        }

        constexpr char* data() noexcept
        {
            return data_.data();
        }

        [[nodiscard]] constexpr value_type tag() const noexcept
        {
            return get(pos_tag);
        }

        [[nodiscard]] constexpr value_type size() const noexcept
        {
            return get(pos_size);
        }

        [[nodiscard]] constexpr value_type numbytes() const noexcept
        {
            return get(pos_numbytes);
        }

        [[nodiscard]] constexpr std::pair<value_type, value_type> num_chunks()
            const noexcept
        {
            return std::make_pair(
                get(pos_numchunks_first), get(pos_numchunks_second));
        }

        [[nodiscard]] constexpr char* piggy_back() noexcept
        {
            if (data_[pos_piggy_back_flag])
                return &data_[pos_piggy_back_data];
            return nullptr;
        }

    private:
        std::array<char, data_size_> data_;

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
