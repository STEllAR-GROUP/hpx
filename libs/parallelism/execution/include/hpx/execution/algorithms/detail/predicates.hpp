//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/type_support/equality.hpp>

#include <hpx/execution/algorithms/detail/is_negative.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {
    template <typename InputIterator, typename Distance>
    HPX_HOST_DEVICE constexpr void advance_impl(
        InputIterator& i, Distance n, std::random_access_iterator_tag)
    {
        i += n;
    }

    template <typename InputIterator, typename Distance>
    HPX_HOST_DEVICE constexpr void advance_impl(
        InputIterator& i, Distance n, std::bidirectional_iterator_tag)
    {
        if (n < 0)
        {
            while (n++)
                --i;
        }
        else
        {
            while (n--)
                ++i;
        }
    }

    template <typename InputIterator, typename Distance>
    HPX_HOST_DEVICE constexpr void advance_impl(
        InputIterator& i, Distance n, std::input_iterator_tag)
    {
        HPX_ASSERT(n >= 0);
        while (n--)
            ++i;
    }

    template <typename InputIterator, typename Distance>
    HPX_HOST_DEVICE constexpr void advance(InputIterator& i, Distance n)
    {
        advance_impl(i, n,
            typename std::iterator_traits<InputIterator>::iterator_category());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterable, typename Enable = void>
    struct calculate_distance
    {
        template <typename T1, typename T2>
        HPX_FORCEINLINE constexpr static std::size_t call(T1 t1, T2 t2)
        {
            return std::size_t(t2 - t1);
        }
    };

    template <typename Iterable>
    struct calculate_distance<Iterable,
        typename std::enable_if<
            hpx::traits::is_iterator<Iterable>::value>::type>
    {
        template <typename Iter1, typename Iter2>
        HPX_FORCEINLINE constexpr static std::size_t call(
            Iter1 iter1, Iter2 iter2)
        {
            return std::distance(iter1, iter2);
        }
    };

    template <typename Iterable>
    HPX_FORCEINLINE constexpr std::size_t distance(
        Iterable iter1, Iterable iter2)
    {
        return calculate_distance<typename std::decay<Iterable>::type>::call(
            iter1, iter2);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterable, typename Enable = void>
    struct calculate_next
    {
        template <typename T, typename Stride>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr static T call(
            T t1, Stride offset)
        {
            return T(t1 + offset);
        }

        template <typename T, typename Stride>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr static T call(
            T t, std::size_t max_count, Stride& offset, std::true_type)
        {
            if (is_negative(offset))
            {
                offset = Stride(negate(
                    // NVCC seems to have a bug with std::min...
                    max_count < std::size_t(negate(offset)) ? max_count :
                                                              negate(offset)));
                return T(t + offset);
            }

            // NVCC seems to have a bug with std::min...
            offset =
                Stride(max_count < std::size_t(offset) ? max_count : offset);
            return T(t + offset);
        }

        template <typename T, typename Stride>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr static T call(
            T t, std::size_t max_count, Stride& offset, std::false_type)
        {
            // NVCC seems to have a bug with std::min...
            offset = max_count < offset ? max_count : offset;
            return T(t + offset);
        }

        template <typename T, typename Stride>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr static T call(
            T t, std::size_t max_count, Stride& offset)
        {
            return call(
                t, max_count, offset, typename std::is_signed<Stride>::type());
        }
    };

    template <typename Iterable>
    struct calculate_next<Iterable,
        typename std::enable_if<hpx::traits::is_iterator<Iterable>::value &&
            !hpx::traits::is_bidirectional_iterator<Iterable>::value>::type>
    {
        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr static Iter call(
            Iter iter, Stride offset)
        {
#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
            hpx::parallel::v1::detail::advance(iter, offset);
#else
            std::advance(iter, offset);
#endif
            return iter;
        }

        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr static Iter call(
            Iter iter, std::size_t max_count, Stride& offset)
        {
            // anything less than a bidirectional iterator does not support
            // negative offsets
            HPX_ASSERT(!std::is_signed<Stride>::value || !is_negative(offset));

            // NVCC seems to have a bug with std::min...
            Stride count =
                Stride(max_count < std::size_t(offset) ? max_count : offset);

            // advance through the end or max number of elements
            for (/**/; count != 0; (void) ++iter, --count)
                /**/;

            offset -= count;
            return iter;
        }
    };

    template <typename Iterable>
    struct calculate_next<Iterable,
        typename std::enable_if<
            hpx::traits::is_bidirectional_iterator<Iterable>::value>::type>
    {
        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr static Iter call(
            Iter iter, Stride offset)
        {
#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
            hpx::parallel::v1::detail::advance(iter, offset);
#else
            std::advance(iter, offset);
#endif
            return iter;
        }

        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr static Iter call(
            Iter iter, std::size_t max_count, Stride& offset, std::true_type)
        {
            // advance through the end or max number of elements
            if (!is_negative(offset))
            {
                // NVCC seems to have a bug with std::min...
                offset = Stride(
                    max_count < std::size_t(offset) ? max_count : offset);
#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
                hpx::parallel::v1::detail::advance(iter, offset);
#else
                std::advance(iter, offset);
#endif
            }
            else
            {
                offset = negate(Stride(
                    // NVCC seems to have a bug with std::min...
                    max_count < negate(offset) ? max_count : negate(offset)));
#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
                hpx::parallel::v1::detail::advance(iter, offset);
#else
                std::advance(iter, offset);
#endif
            }
            return iter;
        }

        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr static Iter call(
            Iter iter, std::size_t max_count, Stride& offset, std::false_type)
        {
            // advance through the end or max number of elements
            // NVCC seems to have a bug with std::min...
            offset =
                Stride(max_count < std::size_t(offset) ? max_count : offset);
#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
            hpx::parallel::v1::detail::advance(iter, offset);
#else
            std::advance(iter, offset);
#endif
            return iter;
        }

        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr static Iter call(
            Iter iter, std::size_t max_count, Stride& offset)
        {
            return call(iter, max_count, offset,
                typename std::is_signed<Stride>::type());
        }
    };

    template <typename Iterable>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iterable next(
        Iterable iter, std::size_t offset)
    {
        return calculate_next<typename std::decay<Iterable>::type>::call(
            iter, offset);
    }

    template <typename Iterable, typename Stride>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iterable next(
        Iterable iter, std::size_t max_count, Stride& offset)
    {
        return calculate_next<typename std::decay<Iterable>::type>::call(
            iter, max_count, offset);
    }

    ///////////////////////////////////////////////////////////////////////////
    struct equal_to
    {
        template <typename T1, typename T2,
            typename Enable = typename std::enable_if<
                hpx::traits::is_equality_comparable_with<T1, T2>::value>::type>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr bool operator()(
            T1&& t1, T2&& t2) const
        {
            return t1 == t2;
        }
    };

    struct not_equal_to
    {
        template <typename T1, typename T2,
            typename Enable = typename std::enable_if<
                hpx::traits::is_equality_comparable_with<T1, T2>::value>::type>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr bool operator()(
            T1&& t1, T2&& t2) const
        {
            return t1 != t2;
        }
    };

    template <typename Value>
    struct compare_to
    {
        HPX_HOST_DEVICE HPX_FORCEINLINE compare_to(Value&& val)
          : value_(std::move(val))
        {
        }
        HPX_HOST_DEVICE HPX_FORCEINLINE compare_to(Value const& val)
          : value_(val)
        {
        }

        template <typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr auto operator()(
            T const& t) const -> decltype(std::declval<Value>() == t)
        {
            return value_ == t;
        }

        Value value_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct less
    {
        template <typename T1, typename T2>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr bool operator()(
            T1&& t1, T2&& t2) const
        {
            return std::forward<T1>(t1) < std::forward<T2>(t2);
        }
    };

    struct greater
    {
        template <typename T1, typename T2>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr bool operator()(
            T1&& t1, T2&& t2) const
        {
            return std::forward<T1>(t1) > std::forward<T2>(t2);
        }
    };

    struct greater_equal
    {
        template <typename T1, typename T2>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr bool operator()(
            T1&& t1, T2&& t2) const
        {
            return std::forward<T1>(t1) >= std::forward<T2>(t2);
        }
    };

    struct less_equal
    {
        template <typename T1, typename T2>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr bool operator()(
            T1&& t1, T2&& t2) const
        {
            return std::forward<T1>(t1) <= std::forward<T2>(t2);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct min_of
    {
        constexpr T operator()(T const& t1, T const& t2) const
        {
            // NVCC seems to have a bug with std::min...
            return t1 < t2 ? t1 : t2;
        }
    };

    template <typename T>
    struct max_of
    {
        constexpr T operator()(T const& t1, T const& t2) const
        {
            return (std::max)(t1, t2);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct plus
    {
        template <typename T1, typename T2>
        constexpr auto operator()(T1 const& t1, T2 const& t2) const
            -> decltype(t1 + t2)
        {
            return t1 + t2;
        }
    };

    struct minus
    {
        template <typename T1, typename T2>
        constexpr auto operator()(T1 const& t1, T2 const& t2) const
            -> decltype(t1 - t2)
        {
            return t1 - t2;
        }
    };

    struct multiplies
    {
        template <typename T1, typename T2>
        constexpr auto operator()(T1 const& t1, T2 const& t2) const
            -> decltype(t1 * t2)
        {
            return t1 * t2;
        }
    };

    struct divides
    {
        template <typename T1, typename T2>
        constexpr auto operator()(T1 const& t1, T2 const& t2) const
            -> decltype(t1 / t2)
        {
            return t1 / t2;
        }
    };
}}}}    // namespace hpx::parallel::v1::detail

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    using equal_to = hpx::parallel::v1::detail::equal_to;
    using not_equal_to = hpx::parallel::v1::detail::not_equal_to;
    using less = hpx::parallel::v1::detail::less;
    using greater = hpx::parallel::v1::detail::greater;
    using greater_equal = hpx::parallel::v1::detail::greater_equal;
    using less_equal = hpx::parallel::v1::detail::less_equal;
}}    // namespace hpx::ranges
