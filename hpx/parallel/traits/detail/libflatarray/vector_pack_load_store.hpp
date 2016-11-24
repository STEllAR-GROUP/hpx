//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_STORE_LIBFLATARRAY)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_STORE_LIBFLATARRAY

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_LIBFLATARRAY)
#include <hpx/parallel/traits/detail/libflatarray/fake_accessor.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <iterator>
#include <memory>

#include <libflatarray/flat_array.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N, typename NewT>
    struct rebind_pack<LibFlatArray::short_vec<T, N>, NewT>
    {
        typedef LibFlatArray::short_vec<NewT, N> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_load
    {
        typedef typename rebind_pack<V, ValueType>::type value_type;

        template <typename Iter>
        static value_type unaligned(Iter const& iter)
        {
            return value_type(std::addressof(*iter));
        }

        template <typename Iter>
        static value_type aligned(Iter const& iter)
        {
            value_type v;
            v.load_aligned(std::addressof(*iter));
            return v;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename ValueType, typename Enable>
    struct vector_pack_store
    {
        template <typename Iter_>
        static void unaligned(V const& value, Iter_ const& iter)
        {
            value.store(std::addressof(*iter));
        }

        template <typename Iter_>
        static void aligned(V const& value, Iter_ const& iter)
        {
            value.store_aligned(std::addressof(*iter));
        }
    };
}}}

#endif
#endif
