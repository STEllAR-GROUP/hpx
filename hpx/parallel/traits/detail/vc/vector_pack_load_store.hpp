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

    // handle packs of tuples (value_types of zip_iterators)
    template <typename ... T, typename Abi, typename NewT>
    struct rebind_pack<Vc::Vector<hpx::util::tuple<T...>, Abi>, NewT>
    {
        typedef Vc::Vector<NewT> type;
    };

    template <typename ... T, typename NewT>
    struct rebind_pack<Vc::Scalar::Vector<hpx::util::tuple<T...> >, NewT>
    {
        typedef Vc::Scalar::Vector<NewT> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename V, typename Enable>
    struct vector_pack_load
    {
        typedef typename rebind_pack<
                V, typename std::iterator_traits<Iter>::value_type
            >::type vector_pack_type;

        template <typename Iter_>
        static vector_pack_type aligned(Iter_ const& iter)
        {
            return vector_pack_type(std::addressof(*iter), Vc::Aligned);
        }

        template <typename Iter_>
        static vector_pack_type unaligned(Iter_ const& iter)
        {
            return vector_pack_type(std::addressof(*iter), Vc::Unaligned);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Enable>
    struct vector_pack_store
    {
        template <typename V_, typename Iter_>
        static void aligned(V_ const& value, Iter_ const& iter)
        {
            value.store(std::addressof(*iter), Vc::Aligned);
        }

        template <typename V_, typename Iter_>
        static void unaligned(V_ const& value, Iter_ const& iter)
        {
            value.store(std::addressof(*iter), Vc::Unaligned);
        }
    };
}}}

#endif
#endif
