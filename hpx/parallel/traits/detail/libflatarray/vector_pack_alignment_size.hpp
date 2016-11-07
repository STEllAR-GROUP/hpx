//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SIZE_LIBFLATARRAY_SEP_29_2016)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SIZE_LIBFLATARRAY_SEP_29_2016

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_LIBFLATARRAY)
#include <hpx/parallel/traits/detail/libflatarray/fake_accessor.hpp>

#include <cstddef>
#include <type_traits>

#include <libflatarray/flat_array.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N>
    struct is_vector_pack<LibFlatArray::short_vec<T, N> >
      : std::true_type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N>
    struct is_scalar_vector_pack<LibFlatArray::short_vec<T, N> >
      : std::false_type
    {};

    template <typename T>
    struct is_scalar_vector_pack<LibFlatArray::short_vec<T, 1> >
      : std::true_type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N>
    struct is_non_scalar_vector_pack<LibFlatArray::short_vec<T, N> >
      : std::true_type
    {};

    template <typename T>
    struct is_non_scalar_vector_pack<LibFlatArray::short_vec<T, 1> >
      : std::false_type
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable>
    struct vector_pack_alignment
    {
        typedef typename LibFlatArray::estimate_optimum_short_vec_type<
                T, fake_accessor
            >::VALUE shortvec;

        static std::size_t const value =
            shortvec::strategy::alignment<T>::ALIGNMENT;
    };

    template <typename T, std::size_t N>
    struct vector_pack_alignment<LibFlatArray::short_vec<T, N> >
    {
        typedef typename LibFlatArray::short_vec<T, N> shortvec;

        static std::size_t const value =
            shortvec::strategy::alignment<T>::ALIGNMENT;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable>
    struct vector_pack_size
    {
        typedef typename LibFlatArray::estimate_optimum_short_vec_type<
                T, fake_accessor
            >::VALUE shortvec;

        static std::size_t const value = shortvec::ARITY;
    };

    template <typename T, std::size_t N>
    struct vector_pack_size<LibFlatArray::short_vec<T, N> >
    {
        static std::size_t const value = LibFlatArray::short_vec<T, N>::ARITY;
    };
}}}

#endif
#endif

