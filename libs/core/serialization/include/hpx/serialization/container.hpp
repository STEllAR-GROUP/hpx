//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c)      2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/basic_archive.hpp>
#include <hpx/serialization/binary_filter.hpp>

#include <cstddef>

namespace hpx::serialization {

    struct erased_output_container
    {
        virtual ~erased_output_container() = default;

        virtual bool is_preprocessing() const noexcept
        {
            return false;
        }
        virtual void set_filter(binary_filter* filter) = 0;
        virtual void save_binary(void const* address, std::size_t count) = 0;
        virtual std::size_t save_binary_chunk(
            void const* address, std::size_t count) = 0;
        virtual void reset() = 0;
        virtual std::size_t get_num_chunks() const noexcept = 0;
        virtual void flush() = 0;
    };

    struct erased_input_container
    {
        virtual ~erased_input_container() = default;

        virtual bool is_preprocessing() const noexcept
        {
            return false;
        }
        virtual void set_filter(binary_filter* filter) = 0;
        virtual void set_zero_copy_serialization_threshold(
            std::size_t zero_copy_serialization_threshold) = 0;
        virtual void load_binary(void* address, std::size_t count) = 0;
        virtual void load_binary_chunk(void* address, std::size_t count) = 0;
    };
}    // namespace hpx::serialization
