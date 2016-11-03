//  Copyright (c) 2016 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TRAITS_LIBFLATARRAY_FAKE_ACCESSOR)
#define HPX_PARALLEL_TRAITS_LIBFLATARRAY_FAKE_ACCESSOR

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_LIBFLATARRAY)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    struct fake_accessor
    {
    public:
        static const int DIM_PROD = 1;
        typedef char element_type;
    };
}}}

#endif
#endif

