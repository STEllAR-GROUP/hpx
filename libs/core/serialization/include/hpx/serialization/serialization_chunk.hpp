//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c)      2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <climits>
#include <cstddef>
#include <cstdint>

#if CHAR_BIT != 8
#error This code assumes an eight-bit byte.
#endif

namespace hpx::serialization {

    ////////////////////////////////////////////////////////////////////////////
    union chunk_data
    {
        std::size_t index_;    // position inside the data buffer //-V117
        void const* cpos_;     // const pointer to external data buffer //-V117
        void* pos_;            // pointer to external data buffer //-V117
    };

    enum class chunk_type : std::uint8_t
    {
        chunk_type_index = 0,
        chunk_type_pointer = 1
    };

    struct serialization_chunk
    {
        chunk_data data_;       // index or pointer
        std::size_t size_;      // size of serialization_chunk starting pos_
        std::uint64_t rkey_;    // optional RDMA remote key for parcelport
                                // operations
        chunk_type type_;       // chunk_type
    };

    ///////////////////////////////////////////////////////////////////////
    inline serialization_chunk create_index_chunk(
        std::size_t index, std::size_t size) noexcept
    {
        serialization_chunk retval = {
            {0}, size, 0, chunk_type::chunk_type_index};
        retval.data_.index_ = index;
        return retval;
    }

    inline serialization_chunk create_pointer_chunk(
        void const* pos, std::size_t size, std::uint64_t rkey = 0) noexcept
    {
        serialization_chunk retval = {
            {0}, size, rkey, chunk_type::chunk_type_pointer};
        retval.data_.cpos_ = pos;
        return retval;
    }
}    // namespace hpx::serialization
