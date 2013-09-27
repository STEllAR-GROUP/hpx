/*=============================================================================
    Copyright (c) 2007-2013 Hartmut Kaiser

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/
#if !defined(HPX_UNUSED_FEB_01_2009_1217PM)
#define HPX_UNUSED_FEB_01_2009_1217PM

#include <boost/fusion/include/unused.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // since boost::fusion supports exactly what we need, unused is simply
    // imported from the fusion namespace
    typedef boost::fusion::unused_type unused_type;
    using boost::fusion::unused;
}}

//////////////////////////////////////////////////////////////////////////////
// use this to silence compiler warnings related to unused function arguments.
#define HPX_UNUSED(x)  ::hpx::util::unused = (x)

#endif
