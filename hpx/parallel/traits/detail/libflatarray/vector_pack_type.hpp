//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_LIBFLATARRAY)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_LIBFLATARRAY

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_LIBFLATARRAY)
#include <hpx/util/tuple.hpp>
#include <hpx/parallel/traits/detail/libflatarray/fake_accessor.hpp>

#include <cstddef>

#include <libflatarray/flat_array.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    template <typename T,
        std::size_t N = LibFlatArray::estimate_optimum_short_vec_type<
            T, fake_accessor>::VALUE::ARITY,
        typename Abi = void>
    struct vector_pack_type
    {
        typedef LibFlatArray::short_vec<T, N> type;
    };

    template <typename ... T, std::size_t N>
    struct vector_pack_type<hpx::util::tuple<T...>, N, void>
    {
        typedef hpx::util::tuple<LibFlatArray::short_vec<T, N>...> type;
    };
}}}

#endif
#endif

