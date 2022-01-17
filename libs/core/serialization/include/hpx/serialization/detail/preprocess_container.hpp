//  Copyright (c) 2015-2019 Hartmut Kaiser
//  Copyright (c) 2015-2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/traits/serialization_access_data.hpp>

#include <cstddef>
#include <type_traits>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization { namespace detail {

    // This 'container' is used to gather the required archive size for a given
    // type before it is serialized.
    class preprocess_container
    {
    public:
        constexpr preprocess_container()
          : size_(0)
        {
        }

        std::size_t size() const
        {
            return size_;
        }
        void resize(std::size_t size)
        {
            size_ = size;
        }

        void reset()
        {
            size_ = 0;
        }

        static constexpr bool is_preprocessing()
        {
            return true;
        }

    private:
        std::size_t size_;
    };
}}}    // namespace hpx::serialization::detail

namespace hpx { namespace traits {

    template <>
    struct serialization_access_data<
        serialization::detail::preprocess_container>
      : default_serialization_access_data<
            serialization::detail::preprocess_container>
    {
        using preprocessing_only = std::true_type;

        static constexpr bool is_preprocessing()
        {
            return true;
        }

        static std::size_t size(
            serialization::detail::preprocess_container const& cont)
        {
            return cont.size();
        }

        static void resize(serialization::detail::preprocess_container& cont,
            std::size_t count)
        {
            return cont.resize(cont.size() + count);
        }

        static void reset(serialization::detail::preprocess_container& cont)
        {
            cont.reset();
        }
    };
}}    // namespace hpx::traits
