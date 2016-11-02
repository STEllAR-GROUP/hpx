//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_VC_SEP_26_2016_0719PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_VC_SEP_26_2016_0719PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)

#include <iterator>
#include <memory>

#include <Vc/Vc>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename NewT>
    struct rebind_pack
    {
        typedef Vc::Vector<T> type;
    };

    // handle non-tuple values
    template <typename T, typename Abi, typename NewT>
    struct rebind_pack<Vc::Vector<T, Abi>, NewT>
    {
        typedef Vc::Vector<NewT, Abi> type;
    };

    template <typename T, typename NewT>
    struct rebind_pack<Vc::Scalar::Vector<T>, NewT>
    {
        typedef Vc::Scalar::Vector<NewT> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename Enable>
    struct vector_pack_load
    {
        template <typename Iter>
                static typename rebind_pack<
            V, typename std::iterator_traits<Iter>::value_type
        >::type
        aligned(Iter const& iter)
        {
            typedef typename rebind_pack<
                    V, typename std::iterator_traits<Iter>::value_type
                >::type vector_pack_type;

            return vector_pack_type(std::addressof(*iter), Vc::Aligned);
        }

        template <typename Iter>
        static typename rebind_pack<
            V, typename std::iterator_traits<Iter>::value_type
        >::type
        unaligned(Iter const& iter)
        {
            typedef typename rebind_pack<
                    V, typename std::iterator_traits<Iter>::value_type
                >::type vector_pack_type;

            return vector_pack_type(std::addressof(*iter), Vc::Unaligned);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename Enable>
    struct vector_pack_store
    {
        template <typename Iter_>
        static void aligned(V const& value, Iter_ const& iter)
        {
            value.store(std::addressof(*iter), Vc::Aligned);
        }

        template <typename Iter_>
        static void unaligned(V const& value, Iter_ const& iter)
        {
            value.store(std::addressof(*iter), Vc::Unaligned);
        }
    };
}}}

#endif
#endif
