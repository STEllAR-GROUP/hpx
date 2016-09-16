//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_ITERATOR_HELPERS_SEP_09_2016_0143PM)
#define HPX_PARALLEL_DATAPAR_ITERATOR_HELPERS_SEP_09_2016_0143PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_VC_DATAPAR)
#include <hpx/util/decay.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include <Vc/Vc>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iter>
        HPX_FORCEINLINE std::size_t data_alignment(Iter it)
        {
            typedef typename std::iterator_traits<Iter>::value_type value_type;
            return reinterpret_cast<std::uintptr_t>(std::addressof(*it)) &
                (Vc::Vector<value_type>::MemoryAlignment - 1);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter1, typename Iter2>
        struct iterators_datapar_compatible_impl
        {
            typedef typename hpx::util::decay<Iter1>::type iterator1_type;
            typedef typename hpx::util::decay<Iter2>::type iterator2_type;

            typedef Vc::Vector<
                    typename std::iterator_traits<iterator1_type>::value_type
                > V1;
            typedef Vc::Vector<
                    typename std::iterator_traits<iterator2_type>::value_type
                > V2;

            typedef std::integral_constant<bool,
                    V1::size == V2::size &&
                        V1::MemoryAlignment == V2::MemoryAlignment
                > type;
        };

        template <typename Iter1, typename Iter2>
        struct iterators_datapar_compatible
          : iterators_datapar_compatible_impl<Iter1, Iter2>::type
        {};

        ///////////////////////////////////////////////////////////////////////
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
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Iter, typename V, typename Enable = void>
        struct store_on_exit
        {
            store_on_exit(Iter const& iter)
              : value_(std::addressof(*iter), Vc::Aligned),
                iter_(iter)
            {
            }
            ~store_on_exit()
            {
                value_.store(std::addressof(*iter_), Vc::Aligned);
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
              : value_(std::addressof(*iter), Vc::Aligned)
            {
            }

            V* operator&() { return &value_; }
            V const* operator&() const { return &value_; }

            V value_;
        };

        ///////////////////////////////////////////////////////////////////////
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
                std::advance(it, V1::Size);
                return hpx::util::invoke(f, &tmp);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename V1, typename V2>
        struct invoke_vectorized_in2
        {
            static_assert(V1::Size == V2::Size,
                "the sizes of the vector-packs should be equal");

            template <typename F, typename Iter1, typename Iter2, typename AlignTag>
            static typename std::result_of<F&&(V1*, V2*)>::type
            call(F && f, Iter1& it1, Iter2& it2, AlignTag align)
            {
                V1 tmp1(std::addressof(*it1), align);
                V2 tmp2(std::addressof(*it2), align);
                std::advance(it1, V1::Size);
                std::advance(it2, V2::Size);
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
                return invoke_vectorized_in2<V11, V12>::call(
                    std::forward<F>(f), it1, it2, Vc::Aligned);
            }

            template <typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::result_of<F&&(V1*, V2*)>::type
            callv(F && f, Iter1& it1, Iter2& it2)
            {
                if (data_alignment(it1) || data_alignment(it2))
                {
                    return invoke_vectorized_in2<V1, V2>::call(
                        std::forward<F>(f), it1, it2, Vc::Unaligned);
                }

                return invoke_vectorized_in2<V1, V2>::call(
                    std::forward<F>(f), it1, it2, Vc::Aligned);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename V>
        struct invoke_vectorized_inout1
        {
            template <typename F, typename InIter, typename OutIter,
                typename AlignTag>
            static void call(F && f, InIter& it, OutIter& dest, AlignTag align)
            {
                V tmp(std::addressof(*it), align);
                auto ret = hpx::util::invoke(f, &tmp);
                ret.store(std::addressof(*dest), align);
                std::advance(it, V::Size);
                std::advance(dest, ret.size());
            }
        };

        template <typename V1, typename V2>
        struct invoke_vectorized_inout2
        {
            static_assert(V1::Size == V2::Size,
                "the sizes of the vector-packs should be equal");

            template <typename F, typename InIter1, typename InIter2,
                typename OutIter, typename AlignTag>
            static void call(F && f, InIter1& it1, InIter2& it2,
                OutIter& dest, AlignTag align)
            {
                V1 tmp1(std::addressof(*it1), align);
                V2 tmp2(std::addressof(*it2), align);

                auto ret = hpx::util::invoke(f, &tmp1, &tmp2);
                ret.store(std::addressof(*dest), align);

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
                invoke_vectorized_inout1<V1>::call(
                    std::forward<F>(f), it, dest, Vc::Aligned);
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

                invoke_vectorized_inout2<V1, V2>::call(
                    std::forward<F>(f), it1, it2, dest, Vc::Aligned);
            }

            ///////////////////////////////////////////////////////////////////
            template <typename F, typename InIter, typename OutIter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static void callv(F && f, InIter& it, OutIter& dest)
            {
                typedef typename std::iterator_traits<InIter>::value_type
                    value_type;

                typedef Vc::Vector<value_type> V;

                if (data_alignment(it) || data_alignment(dest))
                {
                    invoke_vectorized_inout1<V>::call(
                        std::forward<F>(f), it, dest, Vc::Unaligned);
                }
                else
                {
                    invoke_vectorized_inout1<V>::call(
                        std::forward<F>(f), it, dest, Vc::Aligned);
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
                    invoke_vectorized_inout2<V1, V2>::call(
                        std::forward<F>(f), it1, it2, dest, Vc::Unaligned);
                }
                else
                {
                    invoke_vectorized_inout2<V1, V2>::call(
                        std::forward<F>(f), it1, it2, dest, Vc::Aligned);
                }
            }
        };
    }
}}}

#endif
#endif

