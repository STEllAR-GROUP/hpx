//  Copyright (c) 2016-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/algorithms/traits/pointer_category.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>    // for std::memmove
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util {

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

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        HPX_FORCEINLINE std::enable_if_t<std::is_pointer<T>::value, char*>
        to_ptr(T ptr) noexcept
        {
            return const_cast<char*>(
                reinterpret_cast<char volatile const*>(ptr));
        }

        template <typename T>
        HPX_FORCEINLINE std::enable_if_t<std::is_pointer<T>::value, char const*>
        to_const_ptr(T ptr) noexcept
        {
            return const_cast<char const*>(
                reinterpret_cast<char volatile const*>(ptr));
        }

        template <typename Iter>
        HPX_FORCEINLINE std::enable_if_t<!std::is_pointer<Iter>::value, char*>
        to_ptr(Iter ptr) noexcept
        {
            static_assert(hpx::traits::is_contiguous_iterator_v<Iter>,
                "optimized copy/move is possible for contiguous-iterators "
                "only");

            return const_cast<char*>(
                reinterpret_cast<char volatile const*>(&*ptr));
        }

        template <typename Iter>
        HPX_FORCEINLINE
            std::enable_if_t<!std::is_pointer<Iter>::value, char const*>
            to_const_ptr(Iter ptr) noexcept
        {
            static_assert(hpx::traits::is_contiguous_iterator_v<Iter>,
                "optimized copy/move is possible for contiguous-iterators "
                "only");

            return const_cast<char const*>(
                reinterpret_cast<char volatile const*>(&*ptr));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename InIter, typename OutIter>
        HPX_FORCEINLINE static in_out_result<InIter, OutIter> copy_memmove(
            InIter first, std::size_t count, OutIter dest)
        {
            using data_type = typename std::iterator_traits<InIter>::value_type;

            char const* const first_ch = to_const_ptr(first);
            char* const dest_ch = to_ptr(dest);

            std::memmove(dest_ch, first_ch, count * sizeof(data_type));

            std::advance(first, count);
            std::advance(dest, count);
            return in_out_result<InIter, OutIter>{
                std::move(first), std::move(dest)};
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
                    std::move(first), std::move(dest)};
            }
        };

        template <typename Dummy>
        struct copy_helper<hpx::traits::trivially_copyable_pointer_tag, Dummy>
        {
            template <typename InIter, typename Sent, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(
                InIter first, Sent last, OutIter dest)
            {
                return copy_memmove(
                    first, parallel::v1::detail::distance(first, last), dest);
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
                std::size_t count(num & std::size_t(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++first, ++dest, i += 4)
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
                    std::move(first), std::move(dest)};
            }
        };

        template <typename Dummy>
        struct copy_n_helper<hpx::traits::trivially_copyable_pointer_tag, Dummy>
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(
                InIter first, std::size_t count, OutIter dest)
            {
                return copy_memmove(first, count, dest);
            }
        };
    }    // namespace detail

    template <typename ExPolicy>
    struct copy_n_t final : hpx::functional::tag_fallback<copy_n_t<ExPolicy>>
    {
    private:
        template <typename InIter, typename OutIter>
        friend HPX_HOST_DEVICE
            HPX_FORCEINLINE constexpr in_out_result<InIter, OutIter>
            tag_fallback_dispatch(hpx::parallel::util::copy_n_t<ExPolicy>,
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
    HPX_INLINE_CONSTEXPR_VARIABLE copy_n_t<ExPolicy> copy_n =
        copy_n_t<ExPolicy>{};
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
                    *dest++ = std::move(*first++);
                return in_out_result<InIter, OutIter>{
                    std::move(first), std::move(dest)};
            }
        };

        template <typename Dummy>
        struct move_helper<hpx::traits::trivially_copyable_pointer_tag, Dummy>
        {
            template <typename InIter, typename Sent, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(
                InIter first, Sent last, OutIter dest)
            {
                return copy_memmove(
                    first, parallel::v1::detail::distance(first, last), dest);
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
                std::size_t count(num & std::size_t(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++first, ++dest, i += 4)
                {
                    *dest = std::move(*first);
                    *++dest = std::move(*++first);
                    *++dest = std::move(*++first);
                    *++dest = std::move(*++first);
                }
                for (/**/; count < num; (void) ++first, ++dest, ++count)
                {
                    *dest = std::move(*first);
                }
                return in_out_result<InIter, OutIter>{
                    std::move(first), std::move(dest)};
            }
        };

        template <typename Dummy>
        struct move_n_helper<hpx::traits::trivially_copyable_pointer_tag, Dummy>
        {
            template <typename InIter, typename OutIter>
            HPX_FORCEINLINE static in_out_result<InIter, OutIter> call(
                InIter first, std::size_t count, OutIter dest)
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
}}}    // namespace hpx::parallel::util
