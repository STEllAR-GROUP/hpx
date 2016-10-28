//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_BOOST_SIMD_ITERATOR_HELPERS_SEP_22_2016_0228PM)
#define HPX_PARALLEL_DATAPAR_BOOST_SIMD_ITERATOR_HELPERS_SEP_22_2016_0228PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_BOOST_SIMD)
#include <hpx/util/decay.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include <boost/simd.hpp>
#include <boost/simd/function/aligned_store.hpp>

namespace hpx { namespace parallel { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter>
    HPX_FORCEINLINE std::size_t data_alignment(Iter it)
    {
        typedef typename std::iterator_traits<Iter>::value_type value_type;
        return reinterpret_cast<std::uintptr_t>(std::addressof(*it)) &
            (boost::simd::pack<value_type>::alignment - 1);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter1, typename Iter2>
    struct iterators_datapar_compatible_impl
    {
        typedef typename hpx::util::decay<Iter1>::type iterator1_type;
        typedef typename hpx::util::decay<Iter2>::type iterator2_type;

        typedef boost::simd::pack<
                typename std::iterator_traits<iterator1_type>::value_type
            > V1;
        typedef boost::simd::pack<
                typename std::iterator_traits<iterator2_type>::value_type
            > V2;

        typedef std::integral_constant<bool,
                V1::static_size == V2::static_size &&
                    V1::alignment == V2::alignment
            > type;
    };

    template <typename Iter1, typename Iter2>
    struct iterators_datapar_compatible
        : iterators_datapar_compatible_impl<Iter1, Iter2>::type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Enable = void>
    struct iterator_datapar_compatible_impl
        : std::false_type
    {};

    template <typename Iter>
    struct iterator_datapar_compatible_impl<Iter,
            typename std::enable_if<
                hpx::traits::is_random_access_iterator<Iter>::value
            >::type>
        : std::is_arithmetic<typename std::iterator_traits<Iter>::value_type>
    {};

    template <typename Iter>
    struct iterator_datapar_compatible
        : iterator_datapar_compatible_impl<
            typename hpx::util::decay<Iter>::type
        >::type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename V, typename Enable = void>
    struct store_on_exit
    {
        store_on_exit(Iter const& iter)
          : value_(std::addressof(*iter)),
            iter_(iter)
        {
        }
        ~store_on_exit()
        {
            boost::simd::aligned_store(value_, std::addressof(*iter_));
        }

        V* operator&() { return &value_; }
        V const* operator&() const { return &value_; }

        V value_;
        Iter iter_;
    };

    template <typename Iter, typename V>
    struct store_on_exit<Iter, V,
        typename std::enable_if<
            std::is_const<
                typename std::iterator_traits<Iter>::value_type
            >::value
        >::type>
    {
        store_on_exit(Iter const& iter)
          : value_(std::addressof(*iter)),
        {
        }

        V* operator&() { return &value_; }
        V const* operator&() const { return &value_; }

        V value_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Enable = void>
    struct datapar_loop_step
    {
        typedef typename std::iterator_traits<Iter>::value_type value_type;

        typedef boost::simd::pack<value_type, 1> V1;
        typedef boost::simd::pack<value_type> V;

        template <typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::result_of<F&&(V1*)>::type
        call1(F && f, Iter& it)
        {
            store_on_exit<Iter, V1> tmp(it);
            std::advance(it, std::size_t(V1::static_size));
            return hpx::util::invoke(f, &tmp);
        }

        template <typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::result_of<F&&(V*)>::type
        callv(F && f, Iter& it)
        {
            store_on_exit<Iter, V> tmp(it);
            std::advance(it, std::size_t(V::static_size));
            return hpx::util::invoke(f, &tmp);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V1, typename V2>
    struct invoke_vectorized_in2
    {
        static_assert(V1::static_size == V2::static_size,
            "the sizes of the vector-packs should be equal");

        template <typename F, typename Iter1, typename Iter2>
        static typename std::result_of<F&&(V1*, V2*)>::type
        call_unaligned(F && f, Iter1& it1, Iter2& it2)
        {
            V1 tmp1(boost::simd::load<V1>(std::addressof(*it1)));
            V2 tmp2(boost::simd::load<V2>(std::addressof(*it2)));

            std::advance(it1, std::size_t(V1::static_size));
            std::advance(it2, std::size_t(V2::static_size));

            return hpx::util::invoke(std::forward<F>(f), &tmp1, &tmp2);
        }

        template <typename F, typename Iter1, typename Iter2>
        static typename std::result_of<F&&(V1*, V2*)>::type
        call_aligned(F && f, Iter1& it1, Iter2& it2)
        {
            V1 tmp1(std::addressof(*it1));
            V2 tmp2(std::addressof(*it2));

            std::advance(it1, std::size_t(V1::static_size));
            std::advance(it2, std::size_t(V2::static_size));

            return hpx::util::invoke(std::forward<F>(f), &tmp1, &tmp2);
        }
    };

    template <typename Iter1, typename Iter2>
    struct datapar_loop_step2
    {
        typedef typename std::iterator_traits<Iter1>::value_type value1_type;
        typedef typename std::iterator_traits<Iter2>::value_type value2_type;

        typedef boost::simd::pack<value1_type, 1> V11;
        typedef boost::simd::pack<value2_type, 1> V12;

        typedef boost::simd::pack<value1_type> V1;
        typedef boost::simd::pack<value2_type> V2;

        template <typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::result_of<F&&(V11*, V12*)>::type
        call1(F && f, Iter1& it1, Iter2& it2)
        {
            return invoke_vectorized_in2<V11, V12>::call_aligned(
                std::forward<F>(f), it1, it2);
        }

        template <typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::result_of<F&&(V1*, V2*)>::type
        callv(F && f, Iter1& it1, Iter2& it2)
        {
            if (data_alignment(it1) || data_alignment(it2))
            {
                return invoke_vectorized_in2<V1, V2>::call_unaligned(
                    std::forward<F>(f), it1, it2);
            }

            return invoke_vectorized_in2<V1, V2>::call_aligned(
                std::forward<F>(f), it1, it2);
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename V>
    struct invoke_vectorized_inout1
    {
        template <typename F, typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void call_unaligned(F && f, InIter& it, OutIter& dest)
        {
            V tmp(boost::simd::load<V>(std::addressof(*it)));
            auto ret = hpx::util::invoke(f, &tmp);
            boost::simd::store(ret, std::addressof(*dest));
            std::advance(it, std::size_t(V::static_size));
            std::advance(dest, ret.size());
        }

        template <typename F, typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void call_aligned(F && f, InIter& it, OutIter& dest)
        {
            V tmp(std::addressof(*it));
            auto ret = hpx::util::invoke(f, &tmp);
            boost::simd::aligned_store(ret, std::addressof(*dest));
            std::advance(it, std::size_t(V::static_size));
            std::advance(dest, ret.size());
        }
    };

    template <typename V1, typename V2>
    struct invoke_vectorized_inout2
    {
        static_assert(V1::static_size == V2::static_size,
            "the sizes of the vector-packs should be equal");

        template <typename F, typename InIter1, typename InIter2, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE static void
        call_unaligned(F && f, InIter1& it1, InIter2& it2, OutIter& dest)
        {
            V1 tmp1(boost::simd::load<V1>(std::addressof(*it1)));
            V1 tmp2(boost::simd::load<V2>(std::addressof(*it2)));

            auto ret = hpx::util::invoke(f, &tmp1, &tmp2);
            boost::simd::store(ret, std::addressof(*dest));

            std::advance(it1, std::size_t(V1::static_size));
            std::advance(it2, std::size_t(V2::static_size));
            std::advance(dest, ret.size());
        }

        template <typename F, typename InIter1, typename InIter2, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE static void
        call_aligned(F && f, InIter1& it1, InIter2& it2, OutIter& dest)
        {
            V1 tmp1(std::addressof(*it1));
            V1 tmp2(std::addressof(*it2));

            auto ret = hpx::util::invoke(f, &tmp1, &tmp2);
            boost::simd::aligned_store(ret, std::addressof(*dest));

            std::advance(it1, std::size_t(V1::static_size));
            std::advance(it2, std::size_t(V2::static_size));
            std::advance(dest, ret.size());
        }
    };

    struct datapar_transform_loop_step
    {
        template <typename F, typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void call1(F && f, InIter& it, OutIter& dest)
        {
            typedef typename std::iterator_traits<InIter>::value_type
                value_type;

            typedef boost::simd::pack<value_type, 1> V1;
            invoke_vectorized_inout1<V1>::call_aligned(
                std::forward<F>(f), it, dest);
        }

        template <typename F, typename InIter1, typename InIter2,
            typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void call1(F && f, InIter1& it1, InIter2& it2, OutIter& dest)
        {
            typedef typename std::iterator_traits<InIter1>::value_type
                value1_type;
            typedef typename std::iterator_traits<InIter2>::value_type
                value2_type;

            typedef boost::simd::pack<value1_type, 1> V1;
            typedef boost::simd::pack<value2_type, 1> V2;

            invoke_vectorized_inout2<V1, V2>::call_aligned(
                std::forward<F>(f), it1, it2, dest);
        }

        ///////////////////////////////////////////////////////////////////
        template <typename F, typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void callv(F && f, InIter& it, OutIter& dest)
        {
            typedef typename std::iterator_traits<InIter>::value_type
                value_type;

            typedef boost::simd::pack<value_type> V;

            if (data_alignment(it) || data_alignment(dest))
            {
                invoke_vectorized_inout1<V>::call_unaligned(
                    std::forward<F>(f), it, dest);
            }
            else
            {
                invoke_vectorized_inout1<V>::call_aligned(
                    std::forward<F>(f), it, dest);
            }
        }

        template <typename F, typename InIter1, typename InIter2,
            typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void callv(F && f, InIter1& it1, InIter2& it2, OutIter& dest)
        {
            typedef typename std::iterator_traits<InIter1>::value_type
                value1_type;
            typedef typename std::iterator_traits<InIter2>::value_type
                value2_type;

            typedef boost::simd::pack<value1_type> V1;
            typedef boost::simd::pack<value2_type> V2;

            if (data_alignment(it1) || data_alignment(it2) ||
                data_alignment(dest))
            {
                invoke_vectorized_inout2<V1, V2>::call_unaligned(
                    std::forward<F>(f), it1, it2, dest);
            }
            else
            {
                invoke_vectorized_inout2<V1, V2>::call_aligned(
                    std::forward<F>(f), it1, it2, dest);
            }
        }
    };
}}}}

#endif
#endif

