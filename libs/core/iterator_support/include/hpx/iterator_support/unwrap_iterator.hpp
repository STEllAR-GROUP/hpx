//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/concepts/concepts.hpp>
#include <hpx/type_support/bit_cast.hpp>
#include <hpx/type_support/is_contiguous_iterator.hpp>

#include <iterator>
#include <type_traits>

namespace hpx::util {

    namespace detail {

        template <typename Iter,
            HPX_CONCEPT_REQUIRES_(!std::is_pointer_v<Iter>)>
        HPX_FORCEINLINE constexpr auto get_unwrapped_ptr(Iter it) noexcept
        {
            static_assert(hpx::traits::is_contiguous_iterator_v<Iter>,
                "optimized merge is possible for contiguous-iterators "
                "only");

            using value_t = typename std::iterator_traits<Iter>::value_type;
            return const_cast<value_t*>(
                hpx::bit_cast<value_t const volatile*>(&*it));
        }

        template <typename T, HPX_CONCEPT_REQUIRES_(std::is_pointer_v<T>)>
        HPX_FORCEINLINE constexpr auto get_unwrapped_ptr(T ptr) noexcept
        {
            using value_t = std::remove_pointer_t<T>;
            return const_cast<value_t*>(
                hpx::bit_cast<value_t const volatile*>(ptr));
        }
    }    // namespace detail

    template <typename Iter>
    HPX_FORCEINLINE auto get_unwrapped(Iter it)
    {
        // is_contiguous_iterator_v is true for pointers
        if constexpr (hpx::traits::is_contiguous_iterator_v<Iter>)
        {
            return hpx::util::detail::get_unwrapped_ptr(it);
        }
        else
        {
            return it;
        }
    }
}    // namespace hpx::util
