//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/type_support/is_relocatable.hpp>
#include <hpx/type_support/is_trivially_relocatable.hpp>
#include <hpx/type_support/relocate_at.hpp>

#include <cstring>
#include <type_traits>

#if defined(HPX_HAVE_P1144_STD_RELOCATE_AT)
#include <memory>
#endif

namespace hpx::experimental::util {

#if defined(HPX_HAVE_P1144_STD_RELOCATE_AT)
    using std::uninitialized_relocate;
#else

    namespace detail {
        struct buffer_memcpy_tag
        {
        };

        struct for_loop_nothrow_tag
        {
        };

        struct for_loop_try_catch_tag
        {
        };

        template <typename InIter, typename OutIter,
            bool iterators_are_contiguous_v = false>
        struct relocation_traits
        {
            using in_type = typename std::iterator_traits<InIter>::value_type;
            using out_type = typename std::iterator_traits<OutIter>::value_type;

            constexpr static bool valid_relocation =
                is_relocatable_from_v<out_type, in_type>;
            //     ^^^^ Cannot relocate between unrelated types

            constexpr static bool is_memcpyable =
                is_trivially_relocatable_v<in_type> &&
                // ^^^^ The important check
                !std::is_volatile_v<in_type> && !std::is_volatile_v<out_type>;
            //     ^^^^ Cannot memcpy volatile types

            constexpr static bool is_buffer_memcpyable =
                is_memcpyable && iterators_are_contiguous_v;

            constexpr static bool can_move_construct_nothrow =
                std::is_nothrow_constructible_v<in_type, out_type>;
            // This is to skip the try-catch block
            // type_dst is treated as an rvalue by is_nothrow_constructible

            constexpr static bool is_noexcept_relocatable_v =
                can_move_construct_nothrow || is_memcpyable;
            // If memcpy is not possible, we need to check if the move
            // constructor is noexcept

            // Using a tag to distinguish implementations
            // clang-format off
            using implementation_tag = std::conditional_t<
                is_buffer_memcpyable,
                    buffer_memcpy_tag,
                    std::conditional_t<is_noexcept_relocatable_v,
                        for_loop_nothrow_tag,
                        for_loop_try_catch_tag
                    >
                >;
            // clang-format on
        };

        template <typename InIter, typename Size, typename FwdIter>
        FwdIter uninitialized_relocate_n_primitive_helper(
            InIter first, Size n, FwdIter dst, buffer_memcpy_tag) noexcept
        {
            if (n != 0)
            {
                std::byte const* first_byte =
                    reinterpret_cast<std::byte const*>(std::addressof(*first));

                std::byte* dst_byte = const_cast<std::byte*>(
                    reinterpret_cast<std::byte const*>(std::addressof(*dst)));

                Size n_bytes = n * sizeof(*first);

                // Ideally we would want to convey to the compiler
                // That the new buffer actually contains objects
                // within their lifetime. But this is not possible
                // with current language features.
                std::memmove(dst_byte, first_byte, n_bytes);

                dst += n;
            }

            return dst;
        }

        template <typename InIter, typename Size, typename FwdIter>
        // Either the buffer is not contiguous or the types are no-throw
        // move constructible but not trivially relocatable
        FwdIter uninitialized_relocate_n_primitive_helper(
            InIter first, Size n, FwdIter dst, for_loop_nothrow_tag) noexcept
        {
            for (Size i = 0; i < n; ++first, ++dst, ++i)
            {
                // if the type is trivially relocatable this will be a memcpy
                // otherwise it will be a move + destroy
                hpx::experimental::detail::relocate_at_helper(
                    std::addressof(*first), std::addressof(*dst));
            }

            return dst;
        }

        template <typename InIter, typename Size, typename FwdIter>
        FwdIter uninitialized_relocate_n_primitive_helper(
            InIter first, Size n, FwdIter dst, for_loop_try_catch_tag)
        {
            FwdIter original_dst = dst;

            for (Size i = 0; i < n; ++first, ++dst, ++i)
            {
                try
                {
                    // the move + destroy version will be used
                    hpx::experimental::detail::relocate_at_helper(
                        std::addressof(*first), std::addressof(*dst));
                }
                catch (...)
                {
                    // destroy all objects other that the one
                    // that caused the exception
                    // (relocate_at already destroyed that one)

                    // destroy all objects constructed so far
                    std::destroy(original_dst, dst);
                    // destroy all the objects not relocated yet
                    std::destroy_n(++first, n - i - 1);
                    // Note:
                    // using destroy_n instead of destroy + advance
                    // to avoid calculating the distance

                    throw;
                }
            }

            return dst;
        }

    }    // namespace detail

    template <typename InIter, typename FwdIter, typename Size,
        typename iterators_are_contiguous_t>
    // clang-format off
    FwdIter uninitialized_relocate_n_primitive(InIter first, Size n,
        FwdIter dst, iterators_are_contiguous_t) noexcept(
            detail::relocation_traits<InIter, FwdIter>::is_noexcept_relocatable_v)
    {
        static_assert(
            detail::relocation_traits<InIter, FwdIter>::valid_relocation,
            "uninitialized_move(first, last, dst) must be well-formed");

        using implementation_tag = typename detail::relocation_traits<InIter,
            FwdIter, iterators_are_contiguous_t::value>::implementation_tag;

        return detail::uninitialized_relocate_n_primitive_helper(
            first, n, dst, implementation_tag{});
    }
    // clang-format on

    template <typename InIter, typename Size, typename FwdIter>
    FwdIter uninitialized_relocate_n_primitive(InIter first, Size n,
        FwdIter dst) noexcept(detail::relocation_traits<InIter,
        FwdIter>::is_noexcept_relocatable_v)
    {
        using iterators_are_contiguous_default_t =
            std::bool_constant<std::is_pointer_v<InIter> &&
                std::is_pointer_v<FwdIter>>;

        return uninitialized_relocate_n_primitive(
            first, n, dst, iterators_are_contiguous_default_t{});
    }

#endif    // defined(HPX_HAVE_P1144_STD_RELOCATE_AT)

}    // namespace hpx::experimental::util
