//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_LIBFLATARRAY)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_TYPE_LIBFLATARRAY

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_LIBFLATARRAY)
#include <hpx/parallel/traits/detail/libflatarray/fake_accessor.hpp>

#include <cstddef>

#include <libflatarray/flat_array.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T, std::size_t N>
        struct vector_pack_type
        {
            typedef LibFlatArray::short_vec<T, N> type;
        };

        template <typename T>
        struct vector_pack_type<T, 0>
        {
            static std::size_t const N =
                LibFlatArray::estimate_optimum_short_vec_type<
                    T, fake_accessor
                >::VALUE::ARITY;

            typedef LibFlatArray::short_vec<T, N> type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // avoid premature instantiation of
    // LibFlatArray::estimate_optimum_short_vec_type
    template <typename T, std::size_t N, typename Abi>
    struct vector_pack_type
      : detail::vector_pack_type<T, N>
    {};
}}}

#endif
#endif

