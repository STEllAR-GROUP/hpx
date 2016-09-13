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
            return reinterpret_cast<std::uintptr_t>(std::addressof(*it)) &
                (Vc::Vector<typename Iter::value_type>::MemoryAlignment - 1);
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
        template <typename Iter>
        struct iterator_datapar_compatible_impl
        {
            typedef typename hpx::util::decay<Iter>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;

            typedef std::integral_constant<bool,
                    hpx::traits::is_random_access_iterator<iterator_type>::value &&
                        std::is_arithmetic<value_type>::value
                > type;
        };

        template <typename Iter>
        struct iterator_datapar_compatible
          : iterator_datapar_compatible_impl<Iter>::type
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
                it += V1::Size;
                return hpx::util::invoke(f, &tmp);
            }

            template <typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::result_of<F&&(V*)>::type
            callv(F && f, Iter& it)
            {
                store_on_exit<Iter, V> tmp(it);
                it += V::Size;
                return hpx::util::invoke(f, &tmp);
            }
        };

        ///////////////////////////////////////////////////////////////////////
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
                V11 tmp1(std::addressof(*it1), Vc::Aligned);
                V12 tmp2(std::addressof(*it2), Vc::Aligned);
                std::advance(it1, V11::Size);
                std::advance(it2, V12::Size);
                return hpx::util::invoke(f, &tmp1, &tmp2);
            }

            template <typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::result_of<F&&(V1*, V2*)>::type
            callv(F && f, Iter1& it1, Iter2& it2)
            {
                if (data_alignment(it1) || data_alignment(it2))
                {
                    V1 tmp1(std::addressof(*it1), Vc::Unaligned);
                    V2 tmp2(std::addressof(*it2), Vc::Unaligned);
                    std::advance(it1, V1::Size);
                    std::advance(it2, V2::Size);
                    return hpx::util::invoke(f, &tmp1, &tmp2);
                }

                V1 tmp1(std::addressof(*it1), Vc::Aligned);
                V2 tmp2(std::addressof(*it2), Vc::Aligned);
                std::advance(it1, V1::Size);
                std::advance(it2, V2::Size);
                return hpx::util::invoke(f, &tmp1, &tmp2);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct datapar_transform_loop_step
        {
            template <typename F, typename InIter, typename OutIter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static void call1(F && f, InIter& it, OutIter& dest)
            {
                typedef typename std::iterator_traits<InIter>::value_type
                    value_type;

                typedef Vc::Scalar::Vector<value_type> V1;

                V1 tmp(std::addressof(*it), Vc::Aligned);
                auto ret = hpx::util::invoke(f, &tmp);
                ret.store(std::addressof(*dest), Vc::Aligned);

                std::advance(it, V1::Size);
                std::advance(dest, ret.size());
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

                V1 tmp1(std::addressof(*it1), Vc::Aligned);
                V2 tmp2(std::addressof(*it2), Vc::Aligned);
                auto ret = hpx::util::invoke(f, &tmp1, &tmp2);
                ret.store(std::addressof(*dest), Vc::Aligned);

                std::advance(it1, V1::Size);
                std::advance(it2, V2::Size);
                std::advance(dest, ret.size());
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
                    V tmp(std::addressof(*it), Vc::Unaligned);
                    auto ret = hpx::util::invoke(f, &tmp);
                    ret.store(std::addressof(*dest), Vc::Unaligned);
                    std::advance(dest, ret.size());
                }
                else
                {
                    V tmp(std::addressof(*it), Vc::Aligned);
                    auto ret = hpx::util::invoke(f, &tmp);
                    ret.store(std::addressof(*dest), Vc::Aligned);
                    std::advance(dest, ret.size());
                }

                std::advance(it, V::Size);
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
                    V1 tmp1(std::addressof(*it1), Vc::Unaligned);
                    V2 tmp2(std::addressof(*it2), Vc::Unaligned);
                    auto ret = hpx::util::invoke(f, &tmp1, &tmp2);
                    ret.store(std::addressof(*dest), Vc::Unaligned);
                }

                V1 tmp1(std::addressof(*it1), Vc::Aligned);
                V2 tmp2(std::addressof(*it2), Vc::Aligned);
                auto ret = hpx::util::invoke(f, &tmp1, &tmp2);
                ret.store(std::addressof(*dest), Vc::Aligned);

                std::advance(it1, V1::Size);
                std::advance(it2, V2::Size);
                std::advance(dest, ret.size());
            }
        };
    }
}}}

#endif
#endif

