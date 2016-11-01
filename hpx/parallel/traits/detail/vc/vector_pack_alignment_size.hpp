//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SIZE_VC_SEP_29_2016_0905PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SIZE_VC_SEP_29_2016_0905PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)

#include <cstddef>
#include <type_traits>

#include <Vc/Vc>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename T, typename Enable>
    struct vector_pack_alignment
    {
        static std::size_t const value = Vc::Vector<T>::MemoryAlignment;
    };

    template <typename Iter, typename T, typename Abi>
    struct vector_pack_alignment<Iter, Vc::Vector<T, Abi> >
    {
        static std::size_t const value = Vc::Vector<T, Abi>::MemoryAlignment;
    };

    template <typename Iter, typename T, typename Abi>
    struct vector_pack_alignment<Iter, Vc::Scalar::Vector<T, Abi> >
    {
        static std::size_t const value =
            Vc::Scalar::Vector<T, Abi>::MemoryAlignment;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename T, typename Enable>
    struct vector_pack_size
    {
        static std::size_t const value = Vc::Vector<T>::Size;
    };

    template <typename Iter, typename T, typename Abi>
    struct vector_pack_size<Iter, Vc::Vector<T, Abi> >
    {
        static std::size_t const value = Vc::Vector<T, Abi>::Size;
    };

    template <typename Iter, typename T, typename Abi>
    struct vector_pack_size<Iter, Vc::Scalar::Vector<T, Abi> >
    {
        static std::size_t const value = Vc::Scalar::Vector<T, Abi>::Size;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    struct vector_pack_is_scalar<Vc::Vector<T, Abi> >
      : std::false_type
    {};

    template <typename T, typename Abi>
    struct vector_pack_is_scalar<Vc::Scalar::Vector<T, Abi> >
      : std::true_type
    {};
}}}

#endif
#endif

