//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_VC_ITERATOR_HELPERS_SEP_09_2016_0143PM)
#define HPX_PARALLEL_DATAPAR_VC_ITERATOR_HELPERS_SEP_09_2016_0143PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/parallel/traits/vector_pack_load_store.hpp>

#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include <Vc/Vc>

namespace hpx { namespace parallel { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Enable = void>
    struct datapar_loop_step
    {
        typedef typename std::iterator_traits<Iter>::value_type value_type;

        typedef Vc::Scalar::Vector<value_type> V1;
        typedef Vc::Vector<value_type> V;

        template <typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::result_of<F&&(V1*)>::type
        call1(F && f, Iter& it)
        {
            store_on_exit<Iter, V1> tmp(it);
            std::advance(it, V1::Size);
            return hpx::util::invoke(f, &tmp);
        }

        template <typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::result_of<F&&(V*)>::type
        callv(F && f, Iter& it)
        {
            store_on_exit<Iter, V> tmp(it);
            std::advance(it, V::Size);
            return hpx::util::invoke(f, &tmp);
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename V1, typename V2>
    struct invoke_vectorized_in2
    {
        static_assert(V1::Size == V2::Size,
            "the sizes of the vector-packs should be equal");

        template <typename F, typename Iter1, typename Iter2>
        static typename std::result_of<F&&(V1*, V2*)>::type
        call_unaligned(F && f, Iter1& it1, Iter2& it2)
        {
            V1 tmp1(traits::vector_pack_load<Iter1, V1>::aligned(it1));
            V2 tmp2(traits::vector_pack_load<Iter2, V2>::aligned(it2));

            std::advance(it1, std::size_t(V1::static_size));
            std::advance(it2, std::size_t(V2::static_size));

            return hpx::util::invoke(std::forward<F>(f), &tmp1, &tmp2);
        }

        template <typename F, typename Iter1, typename Iter2>
        static typename std::result_of<F&&(V1*, V2*)>::type
        call_aligned(F && f, Iter1& it1, Iter2& it2)
        {
            V1 tmp1(traits::vector_pack_load<Iter1, V1>::unaligned(it1));
            V2 tmp2(traits::vector_pack_load<Iter2, V2>::unaligned(it2));

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

        typedef Vc::Scalar::Vector<value1_type> V11;
        typedef Vc::Scalar::Vector<value2_type> V12;

        typedef Vc::Vector<value1_type> V1;
        typedef Vc::Vector<value2_type> V2;

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

    ///////////////////////////////////////////////////////////////////////////
    template <typename V>
    struct invoke_vectorized_inout1
    {
        template <typename F, typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void call_aligned(F && f, InIter& it, OutIter& dest)
        {
            V tmp(traits::vector_pack_load<InIter, V>::aligned(it));

            auto ret = hpx::util::invoke(f, &tmp);
            traits::vector_pack_store<InIter>::aligned(ret, dest);

            std::advance(it, V::Size);
            std::advance(dest, ret.size());
        }

        template <typename F, typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void call_unaligned(F && f, InIter& it, OutIter& dest)
        {
            V tmp(traits::vector_pack_load<InIter, V>::unaligned(it));

            auto ret = hpx::util::invoke(f, &tmp);
            traits::vector_pack_store<OutIter>::unaligned(ret, dest);

            std::advance(it, V::Size);
            std::advance(dest, ret.size());
        }
    };

    template <typename V1, typename V2>
    struct invoke_vectorized_inout2
    {
        static_assert(V1::Size == V2::Size,
            "the sizes of the vector-packs should be equal");

        template <typename F, typename InIter1, typename InIter2, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void call_aligned(F && f, InIter1& it1, InIter2& it2, OutIter& dest)
        {
            V1 tmp1(traits::vector_pack_load<InIter1, V1>::aligned(it1));
            V2 tmp2(traits::vector_pack_load<InIter2, V2>::aligned(it2));

            auto ret = hpx::util::invoke(f, &tmp1, &tmp2);
            traits::vector_pack_store<OutIter>::aligned(ret, dest);

            std::advance(it1, V1::Size);
            std::advance(it2, V2::Size);
            std::advance(dest, ret.size());
        }

        template <typename F, typename InIter1, typename InIter2, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void call_unaligned(F && f, InIter1& it1, InIter2& it2, OutIter& dest)
        {
            V1 tmp1(traits::vector_pack_load<InIter1, V1>::unaligned(it1));
            V2 tmp2(traits::vector_pack_load<InIter2, V2>::unaligned(it2));

            auto ret = hpx::util::invoke(f, &tmp1, &tmp2);
            traits::vector_pack_store<OutIter>::unaligned(ret, dest);

            std::advance(it1, V1::Size);
            std::advance(it2, V2::Size);
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

            typedef Vc::Scalar::Vector<value_type> V1;
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

            typedef Vc::Scalar::Vector<value1_type> V1;
            typedef Vc::Scalar::Vector<value2_type> V2;

            invoke_vectorized_inout2<V1, V2>::call_aligned(
                std::forward<F>(f), it1, it2, dest);
        }

        template <typename F, typename InIter, typename OutIter>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void callv(F && f, InIter& it, OutIter& dest)
        {
            typedef typename std::iterator_traits<InIter>::value_type
                value_type;

            typedef Vc::Vector<value_type> V;

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

            typedef Vc::Vector<value1_type> V1;
            typedef Vc::Vector<value2_type> V2;

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

namespace hpx { namespace parallel { namespace traits { namespace detail
{
//     template <typename T, typename ... Ts>
//     struct is_same_size
//       : hpx::util::detail::all_of<
//             std::condition_variable<bool, sizeof(T) == sizeof(Ts)>...
//         >
//     {};
//
//     template <typename T>
//     struct is_same_size<T>
//       : std::true_type
//     {};

    template <typename Iter>
    struct vector_pack_size_zip_iterator;

    template <typename ... Iter>
    struct vector_pack_size_zip_iterator<hpx::util::zip_iterator<Iter...> >
    {
//         static_assert(
//             is_same_size<
//                 typename std::iterator_traits<Iter>::value_type...
//             >::value,
//             "all iterator value types of the zip_iterator must have the same "
//             "size");

        static std::size_t const value =
            Vc::Vector<typename hpx::util::detail::at_index<
                    0, typename std::iterator_traits<Iter>::value_type...
                >::type>::Size;
    };
}}}}

#endif
#endif

