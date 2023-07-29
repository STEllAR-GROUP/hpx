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

namespace hpx {

#if defined(HPX_HAVE_P1144_STD_RELOCATE_AT)
    using std::uninitialized_relocate;
#else

    namespace detail {

        enum struct relocate_strategy
        {
            buffer_memcpy = 0,
            for_loop_nothrow,
            for_loop_try_catch
        };

        template <typename InIter, typename FwdIter>
        struct choose_uninitialized_relocate_helper
        {
            //     using in_type = typename std::remove_reference_t<
            //         decltype(*std::declval<InIter>())>;
            //     using out_type = typename std::remove_reference_t<
            //         decltype(*std::declval<FwdIter>())>;

            using in_type = typename std::iterator_traits<InIter>::value_type;
            using out_type = typename std::iterator_traits<FwdIter>::value_type;

            constexpr static bool valid_relocation =
                hpx::is_relocatable_from_v<out_type, in_type>;

            constexpr static bool is_buffer_memcpyable =
                hpx::is_trivially_relocatable_v<in_type> &&
                //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ The important check
                std::is_same_v<std::decay_t<in_type>, std::decay_t<out_type>> &&
                //  can only relocate between same types
                !std::is_volatile_v<in_type> && !std::is_volatile_v<out_type> &&
                //  volatile types are not memcpyable
                std::is_pointer_v<InIter> && std::is_pointer_v<FwdIter>;
            //  ^^ the best we can do to check for contiguous iterators

            constexpr static bool can_move_construct_nothrow =
                std::is_nothrow_constructible_v<out_type,
                    std::add_rvalue_reference_t<in_type>>;
            // Checks if the move constructor is noexcept to skip
            // the try-catch block

            // Using an enum to distinguish implementations
            constexpr static relocate_strategy value = is_buffer_memcpyable ?
                relocate_strategy::buffer_memcpy :
                can_move_construct_nothrow ?
                relocate_strategy::for_loop_nothrow :
                relocate_strategy::for_loop_try_catch;
        };

        template <typename InIter, typename FwdIter,
            std::enable_if_t<
                choose_uninitialized_relocate_helper<InIter, FwdIter>::value ==
                    relocate_strategy::buffer_memcpy,
                int> = 0>
        FwdIter uninitialized_relocate_helper(
            InIter first, InIter last, FwdIter dst) noexcept
        {
            auto n_objects = std::distance(first, last);

            if (n_objects != 0)
            {
                std::byte* first_byte =
                    reinterpret_cast<std::byte*>(std::addressof(*first));
                std::byte* last_byte =
                    reinterpret_cast<std::byte*>(std::addressof(*last));

                auto n_bytes = std::distance(first_byte, last_byte);

                void* dst_void = static_cast<void*>(std::addressof(*dst));
                void* first_void = static_cast<void*>(std::addressof(*first));

                std::memmove(dst_void, first_void, n_bytes);

                dst += n_objects;
            }

            return dst;
        }

        template <typename InIter, typename FwdIter,
            std::enable_if_t<
                choose_uninitialized_relocate_helper<InIter, FwdIter>::value ==
                    relocate_strategy::for_loop_nothrow,
                int> = 0>
        FwdIter uninitialized_relocate_helper(
            InIter first, InIter last, FwdIter dst) noexcept
        {
            for (; first != last; ++first, ++dst)
            {
                // the move + destroy version will be used
                hpx::relocate_at(std::addressof(*first), std::addressof(*dst));
            }

            return dst;
        }

        template <typename InIter, typename FwdIter,
            std::enable_if_t<
                choose_uninitialized_relocate_helper<InIter, FwdIter>::value ==
                    relocate_strategy::for_loop_try_catch,
                int> = 0>
        FwdIter uninitialized_relocate_helper(
            InIter first, InIter last, FwdIter dst)
        {
            FwdIter original_dst = dst;

            for (; first != last; ++first, ++dst)
            {
                try
                {
                    // the move + destroy version will be used
                    hpx::relocate_at(
                        std::addressof(*first), std::addressof(*dst));
                }
                catch (...)
                {
                    // destroy all objects other that the one
                    // that caused the exception

                    // destroy all objects constructed so far
                    std::destroy(original_dst, dst);
                    // destroy all the objects not relocated yet
                    std::destroy(++first, last);

                    throw;
                }
            }

            return dst;
        }

    }    // namespace detail

    template <typename InIter, typename FwdIter>
    FwdIter
    uninitialized_relocate(InIter first, InIter last, FwdIter dst) noexcept(
        detail::choose_uninitialized_relocate_helper<InIter, FwdIter>::value !=
        detail::relocate_strategy::for_loop_try_catch)
    // maybe noexcept(auto)?
    {
        static_assert(detail::choose_uninitialized_relocate_helper<InIter,
                          FwdIter>::valid_relocation,
            "uninitialized_move(first, last, dst) must be well-formed");
        return detail::uninitialized_relocate_helper(first, last, dst);
    }

#endif    // defined(HPX_HAVE_P1144_STD_RELOCATE_AT)

}    // namespace hpx
