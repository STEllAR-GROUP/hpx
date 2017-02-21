//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_PREDICATES_JUL_13_2014_0824PM)
#define HPX_PARALLEL_DETAIL_PREDICATES_JUL_13_2014_0824PM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <utility>


namespace hpx { namespace detail
{
    template<class InputIterator, class Distance>
    HPX_HOST_DEVICE void advance_impl(InputIterator& i, Distance n,
        std::random_access_iterator_tag)
    {
        i += n;
    }

    template<class InputIterator, class Distance>
    HPX_HOST_DEVICE void advance_impl(InputIterator& i, Distance n,
        std::bidirectional_iterator_tag)
    {
        if (n < 0) {
        while (n++) --i;
        } else {
        while (n--) ++i;
        }
    }

    template<class InputIterator, class Distance>
    HPX_HOST_DEVICE void advance_impl(InputIterator& i, Distance n,
        std::input_iterator_tag)
    {
        HPX_ASSERT(n >= 0);
        while (n--) ++i;
    }

    template<class InputIterator, class Distance>
    HPX_HOST_DEVICE void advance (InputIterator& i, Distance n)
    {
        advance_impl(i, n,
        typename std::iterator_traits<InputIterator>::iterator_category());
    }

}}


namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterable, typename Enable = void>
    struct calculate_distance
    {
        template <typename T1, typename T2>
        HPX_FORCEINLINE static std::size_t call(T1 t1, T2 t2)
        {
            return std::size_t(t2 - t1);
        }
    };

    template <typename Iterable>
    struct calculate_distance<Iterable,
        typename std::enable_if<
            hpx::traits::is_iterator<Iterable>::value
        >::type>
    {
        template <typename Iter1, typename Iter2>
        HPX_FORCEINLINE static std::size_t call(Iter1 iter1, Iter2 iter2)
        {
            return std::distance(iter1, iter2);
        }
    };

    template <typename Iterable>
    HPX_FORCEINLINE std::size_t distance(Iterable iter1, Iterable iter2)
    {
        return calculate_distance<
                typename std::decay<Iterable>::type
            >::call(iter1, iter2);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterable, typename Enable = void>
    struct calculate_next
    {
        template <typename T, typename Stride>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE static T call(T t1, Stride offset)
        {
            return T(t1 + offset);
        }

        template <typename T, typename Stride>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE static
        T call(T t, std::size_t max_count, Stride& offset, std::true_type)
        {
            if (is_negative(offset))
            {
                offset = Stride(negate(
                        // NVCC seems to have a bug with std::min...
                        max_count < std::size_t(negate(offset)) ?
                            max_count : negate(offset)
                    ));
                return T(t + offset);
            }

            // NVCC seems to have a bug with std::min...
            offset = Stride(max_count < std::size_t(offset) ? max_count : offset);
            return T(t + offset);
        }

        template <typename T, typename Stride>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE static
        T call(T t, std::size_t max_count, Stride& offset, std::false_type)
        {
            // NVCC seems to have a bug with std::min...
            offset = max_count < offset ? max_count : offset;
            return T(t + offset);
        }

        template <typename T, typename Stride>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE static
        T call(T t, std::size_t max_count, Stride& offset)
        {
            return call(t, max_count, offset,
                typename std::is_signed<Stride>::type());
        }
    };

    template <typename Iterable>
    struct calculate_next<Iterable,
        typename std::enable_if<
            hpx::traits::is_iterator<Iterable>::value &&
           !hpx::traits::is_bidirectional_iterator<Iterable>::value
        >::type>
    {
        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE static Iter call(Iter iter, Stride offset)
        {
#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
            hpx::detail::advance(iter, offset);
#else
            std::advance(iter, offset);
#endif
            return iter;
        }

        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE static
        Iter call(Iter iter, std::size_t max_count, Stride& offset)
        {
            // anything less than a bidirectional iterator does not support
            // negative offsets
            HPX_ASSERT(!std::is_signed<Stride>::value || !is_negative(offset));

            // NVCC seems to have a bug with std::min...
            Stride count =
                Stride(max_count < std::size_t(offset) ? max_count : offset);

            // advance through the end or max number of elements
            for (/**/; count != 0; (void)++iter, --count)
                /**/;

            offset -= count;
            return iter;
        }
    };

    template <typename Iterable>
    struct calculate_next<Iterable,
        typename std::enable_if<
            hpx::traits::is_bidirectional_iterator<Iterable>::value
        >::type>
    {
        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE static Iter call(Iter iter, Stride offset)
        {
#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
            hpx::detail::advance(iter, offset);
#else
            std::advance(iter, offset);
#endif
            return iter;
        }

        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE static
        Iter call(Iter iter, std::size_t max_count, Stride& offset, std::true_type)
        {
            // advance through the end or max number of elements
            if (!is_negative(offset))
            {
                // NVCC seems to have a bug with std::min...
                offset = Stride(max_count < std::size_t(offset) ? max_count : offset);
#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
                hpx::detail::advance(iter, offset);
#else
                std::advance(iter, offset);
#endif
            }
            else
            {
                offset = negate(Stride(
                        // NVCC seems to have a bug with std::min...
                        max_count < negate(offset) ? max_count : negate(offset)
                    ));
#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
                hpx::detail::advance(iter, offset);
#else
                std::advance(iter, offset);
#endif
            }
            return iter;
        }

        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE static
        Iter call(Iter iter, std::size_t max_count, Stride& offset, std::false_type)
        {
            // advance through the end or max number of elements
            // NVCC seems to have a bug with std::min...
            offset = Stride(max_count < std::size_t(offset) ? max_count : offset);
#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
            hpx::detail::advance(iter, offset);
#else
            std::advance(iter, offset);
#endif
            return iter;
        }

        template <typename Iter, typename Stride>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE static
        Iter call(Iter iter, std::size_t max_count, Stride& offset)
        {
            return call(iter, max_count, offset,
                typename std::is_signed<Stride>::type());
        }
    };

    template <typename Iterable>
    HPX_HOST_DEVICE
    HPX_FORCEINLINE Iterable next(Iterable iter, std::size_t offset)
    {
        return calculate_next<
                typename std::decay<Iterable>::type
            >::call(iter, offset);
    }

    template <typename Iterable, typename Stride>
    HPX_HOST_DEVICE
    HPX_FORCEINLINE
    Iterable next(Iterable iter, std::size_t max_count, Stride& offset)
    {
        return calculate_next<
                typename std::decay<Iterable>::type
            >::call(iter, max_count, offset);
    }

    ///////////////////////////////////////////////////////////////////////////
    struct equal_to
    {
        template <typename T1, typename T2>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        auto operator()(T1 const& t1, T2 const& t2) const
        ->  decltype(t1 == t2)
        {
            return t1 == t2;
        }
    };

    template <typename Value>
    struct compare_to
    {
        HPX_HOST_DEVICE HPX_FORCEINLINE
        compare_to(Value && val)
          : value_(std::move(val))
        {}
        HPX_HOST_DEVICE HPX_FORCEINLINE
        compare_to(Value const& val)
          : value_(val)
        {}

        template <typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        auto operator()(T const& t) const
        ->  decltype(std::declval<Value>() == t)
        {
            return value_ == t;
        }

        Value value_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct less
    {
        template <typename T1, typename T2>
        auto operator()(T1 const& t1, T2 const& t2) const
        ->  decltype(t1 < t2)
        {
            return t1 < t2;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct min_of
    {
        T operator()(T const& t1, T const& t2) const
        {
            // NVCC seems to have a bug with std::min...
            return t1 < t2 ? t1 : t2;
        }
    };

    template <typename T>
    struct max_of
    {
        T operator()(T const& t1, T const& t2) const
        {
            return (std::max)(t1, t2);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct plus
    {
        template <typename T1, typename T2>
        auto operator()(T1 const& t1, T2 const& t2) const
        ->  decltype(t1 + t2)
        {
            return t1 + t2;
        }
    };

    struct minus
    {
        template <typename T1, typename T2>
        auto operator()(T1 const& t1, T2 const& t2) const
        ->  decltype(t1 - t2)
        {
            return t1 - t2;
        }
    };

    struct multiplies
    {
        template <typename T1, typename T2>
        auto operator()(T1 const& t1, T2 const& t2) const
        ->  decltype(t1 * t2)
        {
            return t1 * t2;
        }
    };

    struct divides
    {
        template <typename T1, typename T2>
        auto operator()(T1 const& t1, T2 const& t2) const
        ->  decltype(t1 / t2)
        {
            return t1 / t2;
        }
    };
}}}}

#endif
