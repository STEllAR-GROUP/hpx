//  Copyright (c) 2016-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/pointer_category.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/type_support/is_contiguous_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>    // for std::memmove
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::parallel::util {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename Category, typename Enable = void>
        struct copy_helper;
        template <typename Category, typename Enable = void>
        struct copy_n_helper;

        template <typename Category, typename Enable = void>
        struct copy_synchronize_helper;

        template <typename Category, typename Enable = void>
        struct move_helper;
        template <typename Category, typename Enable = void>
        struct move_n_helper;

        template <typename Category, typename Enable = void>
        struct uninit_copy_n_helper;

        template <typename Category, typename Enable = void>
        struct uninit_move_n_helper;

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        HPX_FORCEINLINE std::enable_if_t<std::is_pointer_v<T>, char*> to_ptr(
            T ptr) noexcept
        {
            return const_cast<char*>(
                reinterpret_cast<char const volatile*>(ptr));
        }

        template <typename T>
        HPX_FORCEINLINE std::enable_if_t<std::is_pointer_v<T>, char const*>
        to_const_ptr(T ptr) noexcept
        {
            return const_cast<char const*>(
                reinterpret_cast<char const volatile*>(ptr));
        }

        template <typename Iter>
        HPX_FORCEINLINE std::enable_if_t<!std::is_pointer_v<Iter>, char*>
        to_ptr(Iter ptr) noexcept
        {
            static_assert(hpx::traits::is_contiguous_iterator_v<Iter>,
                "optimized copy/move is possible for contiguous-iterators "
                "only");

            return const_cast<char*>(
                reinterpret_cast<char const volatile*>(&*ptr));
        }

        template <typename Iter>
        HPX_FORCEINLINE std::enable_if_t<!std::is_pointer_v<Iter>, char const*>
        to_const_ptr(Iter ptr) noexcept
        {
            static_assert(hpx::traits::is_contiguous_iterator_v<Iter>,
                "optimized copy/move is possible for contiguous-iterators "
                "only");

            return const_cast<char const*>(
                reinterpret_cast<char const volatile*>(&*ptr));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static in_out_result<InIter, OutIter> copy_memmove(
            InIter first, std::size_t count, OutIter dest) noexcept
        {
            using data_type = hpx::traits::iter_value_t<InIter>;

            if (count != 0)
            {
                char const* const first_ch = to_const_ptr(first);
                char* const dest_ch = to_ptr(dest);

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overread"
#endif
                std::memmove(dest_ch, first_ch, count * sizeof(data_type));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif

                std::advance(first, count);
                std::advance(dest, count);
            }
            return in_out_result<InIter, OutIter>{
                HPX_MOVE(first), HPX_MOVE(dest)};
        }

        ///////////////////////////////////////////////////////////////////////
        // Customization point for optimizing copy operations
        template <typename Category, typename Enable>
        struct copy_helper
        {
            template <typename InIter, typename Sent, typename OutIter>
            HPX_HOST_DEVICE
                HPX_FORCEINLINE static constexpr in_out_result<InIter, OutIter>
                call(InIter first, Sent last, OutIter dest)
            {
                while (first != last)
                    *dest++ = *first++;
                return in_out_result<InIter, OutIter>{
                    HPX_MOVE(first), HPX_MOVE(dest)};
            }
        };

        template <typename Dummy>
        struct copy_helper<hpx::traits::trivially_copyable_pointer_tag, Dummy>
        {
            template <typename InIter, typename Sent, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(
                InIter first, Sent last, OutIter dest) noexcept
            {
                return copy_memmove(
                    first, parallel::detail::distance(first, last), dest);
            }
        };
    }    // namespace detail

    template <typename InIter, typename Sent, typename OutIter>
    HPX_FORCEINLINE constexpr in_out_result<InIter, OutIter> copy(
        InIter first, Sent last, OutIter dest)
    {
        using category = hpx::traits::pointer_copy_category_t<
            std::decay_t<
                hpx::traits::remove_const_iterator_value_type_t<InIter>>,
            std::decay_t<OutIter>>;
        return detail::copy_helper<category>::call(first, last, dest);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Customization point for optimizing copy_n operations
        template <typename Category, typename Enable>
        struct copy_n_helper
        {
            template <typename InIter, typename OutIter>
            HPX_HOST_DEVICE
                HPX_FORCEINLINE static constexpr in_out_result<InIter, OutIter>
                call(InIter first, std::size_t num, OutIter dest)
            {
                std::size_t count(
                    num & static_cast<std::size_t>(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++first, ++dest, i += 4)    //-V112
                {
                    *dest = *first;
                    *++dest = *++first;
                    *++dest = *++first;
                    *++dest = *++first;
                }
                for (/**/; count < num; (void) ++first, ++dest, ++count)
                {
                    *dest = *first;
                }
                return in_out_result<InIter, OutIter>{
                    HPX_MOVE(first), HPX_MOVE(dest)};
            }
        };

        template <typename Dummy>
        struct copy_n_helper<hpx::traits::trivially_copyable_pointer_tag, Dummy>
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(
                InIter first, std::size_t count, OutIter dest) noexcept
            {
                return copy_memmove(first, count, dest);
            }
        };
    }    // namespace detail

    template <typename ExPolicy>
    struct copy_n_t final
      : hpx::functional::detail::tag_fallback<copy_n_t<ExPolicy>>
    {
    private:
        template <typename InIter, typename OutIter>
        friend HPX_HOST_DEVICE
            HPX_FORCEINLINE constexpr in_out_result<InIter, OutIter>
            tag_fallback_invoke(hpx::parallel::util::copy_n_t<ExPolicy>,
                InIter first, std::size_t count, OutIter dest)
        {
            using category = hpx::traits::pointer_copy_category_t<
                std::decay_t<
                    hpx::traits::remove_const_iterator_value_type_t<InIter>>,
                std::decay_t<OutIter>>;
            return detail::copy_n_helper<category>::call(first, count, dest);
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr copy_n_t<ExPolicy> copy_n = copy_n_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename InIter, typename OutIter>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr in_out_result<InIter, OutIter>
    copy_n(InIter first, std::size_t count, OutIter dest)
    {
        return hpx::parallel::util::copy_n_t<ExPolicy>{}(first, count, dest);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Customization point for copy-synchronize operations
        template <typename Category, typename Enable>
        struct copy_synchronize_helper
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static constexpr void call(
                InIter const&, OutIter const&) noexcept
            {
                // do nothing by default (std::memmove is already synchronous)
            }
        };
    }    // namespace detail

    template <typename InIter, typename OutIter>
    HPX_FORCEINLINE constexpr void copy_synchronize(
        InIter const& first, OutIter const& dest)
    {
        using category =
            hpx::traits::pointer_copy_category_t<std::decay_t<InIter>,
                std::decay_t<OutIter>>;
        detail::copy_synchronize_helper<category>::call(first, dest);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Customization point for optimizing copy_n operations
        template <typename Category, typename Enable>
        struct move_helper
        {
            template <typename InIter, typename Sent, typename OutIter>
            HPX_FORCEINLINE static constexpr in_out_result<InIter, OutIter>
            call(InIter first, Sent last, OutIter dest)
            {
                while (first != last)
                {
                    // NOLINTNEXTLINE(bugprone-macro-repeated-side-effects)
                    *dest++ = HPX_MOVE(*first++);
                }

                return in_out_result<InIter, OutIter>{
                    HPX_MOVE(first), HPX_MOVE(dest)};
            }
        };

        template <typename Dummy>
        struct move_helper<hpx::traits::trivially_copyable_pointer_tag, Dummy>
        {
            template <typename InIter, typename Sent, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(
                InIter first, Sent last, OutIter dest) noexcept
            {
                return copy_memmove(
                    first, parallel::detail::distance(first, last), dest);
            }
        };
    }    // namespace detail

    template <typename InIter, typename Sent, typename OutIter>
    HPX_FORCEINLINE constexpr in_out_result<InIter, OutIter> move(
        InIter first, Sent last, OutIter dest)
    {
        using category =
            hpx::traits::pointer_move_category_t<std::decay_t<InIter>,
                std::decay_t<OutIter>>;
        return detail::move_helper<category>::call(first, last, dest);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Customization point for optimizing copy_n operations
        template <typename Category, typename Enable>
        struct move_n_helper
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static constexpr in_out_result<InIter, OutIter>
            call(InIter first, std::size_t num, OutIter dest)
            {
                std::size_t count(
                    num & static_cast<std::size_t>(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++first, ++dest, i += 4)    //-V112
                {
                    *dest = HPX_MOVE(*first);
                    // NOLINTNEXTLINE(bugprone-macro-repeated-side-effects)
                    *++dest = HPX_MOVE(*++first);
                    // NOLINTNEXTLINE(bugprone-macro-repeated-side-effects)
                    *++dest = HPX_MOVE(*++first);
                    // NOLINTNEXTLINE(bugprone-macro-repeated-side-effects)
                    *++dest = HPX_MOVE(*++first);
                }
                for (/**/; count < num; (void) ++first, ++dest, ++count)
                {
                    *dest = HPX_MOVE(*first);
                }
                return in_out_result<InIter, OutIter>{
                    HPX_MOVE(first), HPX_MOVE(dest)};
            }
        };

        template <typename Dummy>
        struct move_n_helper<hpx::traits::trivially_copyable_pointer_tag, Dummy>
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(
                InIter first, std::size_t count, OutIter dest) noexcept
            {
                return copy_memmove(first, count, dest);
            }
        };
    }    // namespace detail

    template <typename InIter, typename OutIter>
    HPX_FORCEINLINE constexpr in_out_result<InIter, OutIter> move_n(
        InIter first, std::size_t count, OutIter dest)
    {
        using category =
            hpx::traits::pointer_move_category_t<std::decay_t<InIter>,
                std::decay_t<OutIter>>;
        return detail::move_n_helper<category>::call(first, count, dest);
    }

    // helpers for uninit_copy_n
    namespace detail {
        // Customization point for optimizing copy_n operations
        template <typename Category, typename Dummy>
        struct uninit_copy_n_helper
        {
            template <typename ExPolicy, typename InIter, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(
                ExPolicy&& policy, InIter first, std::size_t num, OutIter dest)
            {
                return in_out_result<InIter, OutIter>{std::next(first, num),
                    ::hpx::parallel::util::loop_with_cleanup_n(
                        HPX_FORWARD(ExPolicy, policy), first, num, dest,
                        [](InIter it, OutIter current) -> void {
                            hpx::construct_at(std::addressof(*current), *it);
                        },
                        [](OutIter it) -> void {
                            std::destroy_at(std::addressof(*it));
                        })};
            }
        };

        template <typename Dummy>
        struct uninit_copy_n_helper<hpx::traits::trivially_copyable_pointer_tag,
            Dummy>
        {
            template <typename ExPolicy, typename InIter, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(ExPolicy,
                InIter first, std::size_t count, OutIter dest) noexcept
            {
                return copy_memmove(first, count, dest);
            }
        };
    }    // namespace detail

    template <typename InIter, typename OutIter>
    HPX_FORCEINLINE constexpr in_out_result<InIter, OutIter> uninit_copy_n(
        InIter first, std::size_t count, OutIter dest)
    {
        using category =
            hpx::traits::pointer_copy_category_t<std::decay_t<InIter>,
                std::decay_t<OutIter>>;
        return detail::uninit_copy_n_helper<category>::call(
            hpx::execution::seq, first, count, dest);
    }

    template <typename ExPolicy, typename InIter, typename OutIter>
    HPX_FORCEINLINE constexpr in_out_result<InIter, OutIter> uninit_copy_n(
        ExPolicy&& policy, InIter first, std::size_t count, OutIter dest)
    {
        using category =
            hpx::traits::pointer_copy_category_t<std::decay_t<InIter>,
                std::decay_t<OutIter>>;
        return detail::uninit_copy_n_helper<category>::call(
            HPX_FORWARD(ExPolicy, policy), first, count, dest);
    }

    // helpers for uninit_move_n
    namespace detail {
        // Customization point for optimizing move_n operations
        template <typename Category, typename Enable>
        struct uninit_move_n_helper
        {
            template <typename ExPolicy, typename InIter, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(
                ExPolicy&& policy, InIter first, std::size_t num, OutIter dest)
            {
                return in_out_result<InIter, OutIter>{std::next(first, num),
                    ::hpx::parallel::util::loop_with_cleanup_n(
                        HPX_FORWARD(ExPolicy, policy), first, num, dest,
                        [](InIter it, OutIter current) -> void {
                            hpx::construct_at(
                                std::addressof(*current), HPX_MOVE(*it));
                        },
                        [](OutIter it) -> void {
                            std::destroy_at(std::addressof(*it));
                        })};
            }
        };

        template <typename Dummy>
        struct uninit_move_n_helper<hpx::traits::trivially_copyable_pointer_tag,
            Dummy>
        {
            template <typename ExPolicy, typename InIter, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(ExPolicy,
                InIter first, std::size_t count, OutIter dest) noexcept
            {
                return copy_memmove(first, count, dest);
            }
        };
    }    // namespace detail

    template <typename InIter, typename OutIter>
    HPX_FORCEINLINE constexpr in_out_result<InIter, OutIter> uninit_move_n(
        InIter first, std::size_t count, OutIter dest)
    {
        using category =
            hpx::traits::pointer_copy_category_t<std::decay_t<InIter>,
                std::decay_t<OutIter>>;
        return detail::uninit_move_n_helper<category>::call(
            hpx::execution::seq, first, count, dest);
    }

    template <typename ExPolicy, typename InIter, typename OutIter>
    HPX_FORCEINLINE constexpr in_out_result<InIter, OutIter> uninit_move_n(
        ExPolicy&& policy, InIter first, std::size_t count, OutIter dest)
    {
        using category =
            hpx::traits::pointer_copy_category_t<std::decay_t<InIter>,
                std::decay_t<OutIter>>;
        return detail::uninit_move_n_helper<category>::call(
            HPX_FORWARD(ExPolicy, policy), first, count, dest);
    }
}    // namespace hpx::parallel::util
