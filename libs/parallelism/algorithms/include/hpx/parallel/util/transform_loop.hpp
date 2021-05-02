//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename Iter>
        struct transform_loop
        {
            template <typename InIterB, typename InIterE, typename OutIter,
                typename F>
            HPX_HOST_DEVICE
                HPX_FORCEINLINE static constexpr util::in_out_result<InIterB,
                    OutIter>
                call(InIterB first, InIterE last, OutIter dest, F&& f)
            {
                for (/* */; first != last; (void) ++first, ++dest)
                {
                    *dest = HPX_INVOKE(f, first);
                }

                return util::in_out_result<InIterB, OutIter>{
                    std::move(first), std::move(dest)};
            }
        };
    }    // namespace detail

    struct transform_loop_t final
      : hpx::functional::tag_fallback<transform_loop_t>
    {
    private:
        template <typename ExPolicy, typename IterB, typename IterE,
            typename OutIter, typename F>
        friend HPX_HOST_DEVICE
            HPX_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
            tag_fallback_invoke(hpx::parallel::util::transform_loop_t,
                ExPolicy&&, IterB it, IterE end, OutIter dest, F&& f)
        {
            return detail::transform_loop<IterB>::call(
                it, end, dest, std::forward<F>(f));
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    HPX_INLINE_CONSTEXPR_VARIABLE transform_loop_t transform_loop =
        transform_loop_t{};
#else
    template <typename ExPolicy, typename IterB, typename IterE,
        typename OutIter, typename F>
    HPX_HOST_DEVICE
        HPX_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
        transform_loop(ExPolicy&&, IterB it, IterE end, OutIter dest, F&& f)
    {
        return hpx::parallel::util::transform_loop_t{}(
            ExPolicy&&, it, end, dest, std::forward<F>(f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename Iter>
        struct transform_loop_ind
        {
            template <typename InIterB, typename InIterE, typename OutIter,
                typename F>
            HPX_HOST_DEVICE
                HPX_FORCEINLINE static constexpr util::in_out_result<InIterB,
                    OutIter>
                call(InIterB first, InIterE last, OutIter dest, F&& f)
            {
                for (/* */; first != last; (void) ++first, ++dest)
                {
                    *dest = HPX_INVOKE(f, *first);
                }

                return util::in_out_result<InIterB, OutIter>{
                    std::move(first), std::move(dest)};
            }
        };
    }    // namespace detail

    struct transform_loop_ind_t final
      : hpx::functional::tag_fallback<transform_loop_ind_t>
    {
    private:
        template <typename ExPolicy, typename IterB, typename IterE,
            typename OutIter, typename F>
        friend HPX_HOST_DEVICE
            HPX_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
            tag_fallback_invoke(hpx::parallel::util::transform_loop_ind_t,
                ExPolicy&&, IterB it, IterE end, OutIter dest, F&& f)
        {
            return detail::transform_loop_ind<IterB>::call(
                it, end, dest, std::forward<F>(f));
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    HPX_INLINE_CONSTEXPR_VARIABLE transform_loop_ind_t transform_loop_ind =
        transform_loop_ind_t{};
#else
    template <typename ExPolicy, typename IterB, typename IterE,
        typename OutIter, typename F>
    HPX_HOST_DEVICE
        HPX_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
        transform_loop_ind(ExPolicy&&, IterB it, IterE end, OutIter dest, F&& f)
    {
        return hpx::parallel::util::transform_loop_ind_t{}(
            ExPolicy&&, it, end, dest, std::forward<F>(f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename Iter1, typename Iter2>
        struct transform_binary_loop
        {
            template <typename InIter1B, typename InIter1E, typename InIter2,
                typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr util::
                in_in_out_result<InIter1B, InIter2, OutIter>
                call(InIter1B first1, InIter1E last1, InIter2 first2,
                    OutIter dest, F&& f)
            {
                for (/* */; first1 != last1; (void) ++first1, ++first2, ++dest)
                {
                    *dest = HPX_INVOKE(f, first1, first2);
                }

                return util::in_in_out_result<InIter1B, InIter2, OutIter>{
                    std::move(first1), std::move(first2), std::move(dest)};
            }

            template <typename InIter1B, typename InIter1E, typename InIter2,
                typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr util::
                in_in_out_result<InIter1B, InIter2, OutIter>
                call(InIter1B first1, InIter1E last1, InIter2 first2,
                    InIter2 last2, OutIter dest, F&& f)
            {
                for (/* */; first1 != last1 && first2 != last2;
                     (void) ++first1, ++first2, ++dest)
                {
                    *dest = HPX_INVOKE(f, first1, first2);
                }

                return util::in_in_out_result<InIter1B, InIter2, OutIter>{
                    first1, first2, dest};
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter1B, typename InIter1E,
        typename InIter2, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        !hpx::is_vectorpack_execution_policy<ExPolicy>::value,
        util::in_in_out_result<InIter1B, InIter2, OutIter>>::type
    transform_binary_loop(
        InIter1B first1, InIter1E last1, InIter2 first2, OutIter dest, F&& f)
    {
        return detail::transform_binary_loop<InIter1B, InIter2>::call(
            first1, last1, first2, dest, std::forward<F>(f));
    }

    template <typename ExPolicy, typename InIter1B, typename InIter1E,
        typename InIter2B, typename InIter2E, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        !hpx::is_vectorpack_execution_policy<ExPolicy>::value,
        util::in_in_out_result<InIter1B, InIter2B, OutIter>>::type
    transform_binary_loop(InIter1B first1, InIter1E last1, InIter2B first2,
        InIter2E last2, OutIter dest, F&& f)
    {
        return detail::transform_binary_loop<InIter1B, InIter2B>::call(
            first1, last1, first2, last2, dest, std::forward<F>(f));
    }

    namespace detail {

        template <typename Iter>
        struct transform_loop_n
        {
            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE
                HPX_FORCEINLINE static constexpr std::pair<InIter, OutIter>
                call(InIter it, std::size_t num, OutIter dest, F&& f,
                    std::false_type)
            {
                std::size_t count(num & std::size_t(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++it, i += 4)    // -V112
                {
                    *dest++ = HPX_INVOKE(f, it);
                    *dest++ = HPX_INVOKE(f, ++it);
                    *dest++ = HPX_INVOKE(f, ++it);
                    *dest++ = HPX_INVOKE(f, ++it);
                }
                for (/**/; count < num; (void) ++count, ++it, ++dest)
                {
                    *dest = HPX_INVOKE(f, it);
                }

                return std::make_pair(std::move(it), std::move(dest));
            }

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE
                HPX_FORCEINLINE static constexpr std::pair<InIter, OutIter>
                call(InIter it, std::size_t num, OutIter dest, F&& f,
                    std::true_type)
            {
                while (num >= 4)
                {
                    *dest++ = HPX_INVOKE(f, it);
                    *dest++ = HPX_INVOKE(f, it + 1);
                    *dest++ = HPX_INVOKE(f, it + 2);
                    *dest++ = HPX_INVOKE(f, it + 3);

                    it += 4;
                    num -= 4;
                }

                switch (num)
                {
                case 3:
                    *dest++ = HPX_INVOKE(f, it);
                    *dest++ = HPX_INVOKE(f, it + 1);
                    *dest++ = HPX_INVOKE(f, it + 2);
                    break;

                case 2:
                    *dest++ = HPX_INVOKE(f, it);
                    *dest++ = HPX_INVOKE(f, it + 1);
                    break;

                case 1:
                    *dest++ = HPX_INVOKE(f, it);
                    break;

                default:
                    break;
                }

                return std::make_pair(it + num, std::move(dest));
            }
        };
    }    // namespace detail

    template <typename ExPolicy>
    struct transform_loop_n_t final
      : hpx::functional::tag_fallback<transform_loop_n_t<ExPolicy>>
    {
    private:
        template <typename Iter, typename OutIter, typename F>
        friend HPX_HOST_DEVICE
            HPX_FORCEINLINE constexpr std::pair<Iter, OutIter>
            tag_fallback_invoke(
                hpx::parallel::util::transform_loop_n_t<ExPolicy>, Iter it,
                std::size_t count, OutIter dest, F&& f)
        {
            using pred = hpx::traits::is_random_access_iterator<Iter>;

            return detail::transform_loop_n<Iter>::call(
                it, count, dest, std::forward<F>(f), pred());
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    HPX_INLINE_CONSTEXPR_VARIABLE transform_loop_n_t<ExPolicy>
        transform_loop_n = transform_loop_n_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr std::pair<Iter, OutIter>
    transform_loop_n(Iter it, std::size_t count, OutIter dest, F&& f)
    {
        return hpx::parallel::util::transform_loop_n_t<ExPolicy>{}(
            it, count, std::forward<F>(f));
    }
#endif

    namespace detail {

        template <typename Iter>
        struct transform_loop_n_ind
        {
            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE
                HPX_FORCEINLINE static constexpr std::pair<InIter, OutIter>
                call(InIter it, std::size_t num, OutIter dest, F&& f,
                    std::false_type)
            {
                std::size_t count(num & std::size_t(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++it, i += 4)    // -V112
                {
                    *dest++ = HPX_INVOKE(f, *it);
                    *dest++ = HPX_INVOKE(f, *(++it));
                    *dest++ = HPX_INVOKE(f, *(++it));
                    *dest++ = HPX_INVOKE(f, *(++it));
                }
                for (/**/; count < num; (void) ++count, ++it, ++dest)
                {
                    *dest = HPX_INVOKE(f, *it);
                }

                return std::make_pair(std::move(it), std::move(dest));
            }

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE
                HPX_FORCEINLINE static constexpr std::pair<InIter, OutIter>
                call(InIter it, std::size_t num, OutIter dest, F&& f,
                    std::true_type)
            {
                while (num >= 4)
                {
                    *dest++ = HPX_INVOKE(f, *it);
                    *dest++ = HPX_INVOKE(f, *(it + 1));
                    *dest++ = HPX_INVOKE(f, *(it + 2));
                    *dest++ = HPX_INVOKE(f, *(it + 3));

                    it += 4;
                    num -= 4;
                }

                switch (num)
                {
                case 3:
                    *dest++ = HPX_INVOKE(f, *it);
                    *dest++ = HPX_INVOKE(f, *(it + 1));
                    *dest++ = HPX_INVOKE(f, *(it + 2));
                    break;

                case 2:
                    *dest++ = HPX_INVOKE(f, *it);
                    *dest++ = HPX_INVOKE(f, *(it + 1));
                    break;

                case 1:
                    *dest++ = HPX_INVOKE(f, *it);
                    break;

                default:
                    break;
                }

                return std::make_pair(it + num, std::move(dest));
            }
        };
    }    // namespace detail

    template <typename ExPolicy>
    struct transform_loop_n_ind_t final
      : hpx::functional::tag_fallback<transform_loop_n_ind_t<ExPolicy>>
    {
    private:
        template <typename Iter, typename OutIter, typename F>
        friend HPX_HOST_DEVICE
            HPX_FORCEINLINE constexpr std::pair<Iter, OutIter>
            tag_fallback_invoke(
                hpx::parallel::util::transform_loop_n_ind_t<ExPolicy>, Iter it,
                std::size_t count, OutIter dest, F&& f)
        {
            using pred = hpx::traits::is_random_access_iterator<Iter>;

            return detail::transform_loop_n_ind<Iter>::call(
                it, count, dest, std::forward<F>(f), pred());
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    HPX_INLINE_CONSTEXPR_VARIABLE transform_loop_n_ind_t<ExPolicy>
        transform_loop_n_ind = transform_loop_n_ind_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr std::pair<Iter, OutIter>
    transform_loop_n_ind(Iter it, std::size_t count, OutIter dest, F&& f)
    {
        return hpx::parallel::util::transform_loop_n_ind_t<ExPolicy>{}(
            it, count, std::forward<F>(f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename Iter1, typename Inter2>
        struct transform_binary_loop_n
        {
            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr hpx::tuple<InIter1,
                InIter2, OutIter>
            call(InIter1 first1, std::size_t count, InIter2 first2,
                OutIter dest, F&& f)
            {
                for (/* */; count != 0;
                     (void) --count, ++first1, first2++, ++dest)
                {
                    *dest = HPX_INVOKE(f, first1, first2);
                }

                return hpx::make_tuple(
                    std::move(first1), std::move(first2), std::move(dest));
            }
        };
    }    // namespace detail

    template <typename ExPolicy>
    struct transform_binary_loop_n_t final
      : hpx::functional::tag_fallback<transform_binary_loop_n_t<ExPolicy>>
    {
    private:
        template <typename InIter1, typename InIter2, typename OutIter,
            typename F>
        friend HPX_HOST_DEVICE
            HPX_FORCEINLINE constexpr hpx::tuple<InIter1, InIter2, OutIter>
            tag_fallback_invoke(
                hpx::parallel::util::transform_binary_loop_n_t<ExPolicy>,
                InIter1 first1, std::size_t count, InIter2 first2, OutIter dest,
                F&& f)
        {
            return detail::transform_binary_loop_n<InIter1, InIter2>::call(
                first1, count, first2, dest, std::forward<F>(f));
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    HPX_INLINE_CONSTEXPR_VARIABLE transform_binary_loop_n_t<ExPolicy>
        transform_binary_loop_n = transform_binary_loop_n_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr std::pair<Iter, OutIter>
    transform_binary_loop_n(
        InIter1 first1, std::size_t count, InIter2 first2, OutIter dest, F&& f)
    {
        return hpx::parallel::util::transform_binary_loop_n_t<ExPolicy>{}(
            first, count, first2, dest, std::forward<F>(f));
    }
#endif

}}}    // namespace hpx::parallel::util

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/parallel/datapar/transform_loop.hpp>
#endif
