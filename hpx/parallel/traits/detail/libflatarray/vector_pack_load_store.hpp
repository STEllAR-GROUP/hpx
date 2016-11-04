//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_STORE_LIBFLATARRAY)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_LOAD_STORE_LIBFLATARRAY

#include <hpx/parallel/traits/detail/libflatarray/fake_accessor.hpp>

#if defined(HPX_HAVE_DATAPAR_LIBFLATARRAY)
#include <hpx/util/tuple.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <iterator>
#include <memory>

#include <libflatarray/flat_array.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename NewT>
    struct rebind_pack
    {
        typedef typename LibFlatArray::estimate_optimum_short_vec_type<T, fake_accessor>::VALUE type;
    };

    template <typename T, std::size_t N, typename NewT>
    struct rebind_pack<LibFlatArray::short_vec<T, N>, NewT>
    {
        typedef typename LibFlatArray::estimate_optimum_short_vec_type<NewT, fake_accessor>::VALUE type;
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

            return vector_pack_type(
                std::addressof(*iter));
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

            vector_pack_type v;
            v.load_aligned(std::addressof(*iter));
            return v;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename V, typename Enable>
    struct vector_pack_store
    {
        template <typename Iter_>
        static void aligned(V const& value, Iter_ const& iter)
        {
            value.store(std::addressof(*iter));
        }

        template <typename Iter_>
        static void unaligned(V const& value, Iter_ const& iter)
        {
            value.store_aligned(std::addressof(*iter));
        }
    };
}}}

#endif
#endif
