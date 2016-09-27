//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SIZE_VC_SEP_29_2016_0905PM)
#define HPX_PARALLEL_TRAITS_VECTOR_PACK_ALIGNMENT_SIZE_VC_SEP_29_2016_0905PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)

#include <cstddef>

#include <Vc/Vc>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    template <typename Iter, typename T, typename Enable>
    struct vector_pack_alignment
    {
        static std::size_t const value = Vc::Vector<T>::MemoryAlignment;
    };

    template <typename Iter, typename T, typename Enable>
    struct vector_pack_size
    {
        static std::size_t const value = Vc::Vector<T>::Size;
    };
}}}

#endif
#endif

