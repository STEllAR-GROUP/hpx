//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2016 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_LIBFLATARRAY_COUNT_BITS_SEP_22_2016_0220PM)
#define HPX_PARALLEL_DATAPAR_LIBFLATARRAY_COUNT_BITS_SEP_22_2016_0220PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_LIBFLATARRAY)
#include <cstddef>

#include <libflatarray/flat_array.hpp>

namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t N>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::size_t
    count_bits(LibFlatArray::short_vec<T, N>::mask_type const& mask)
    {
        return LibFlatArray::count_mask(mask);
    }
}}}

#endif
#endif

