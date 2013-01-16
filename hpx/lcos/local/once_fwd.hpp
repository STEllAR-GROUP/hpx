//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_ONCE_FWD_JAN_05_2013_0427PM)
#define HPX_LCOS_LOCAL_ONCE_FWD_JAN_05_2013_0427PM

namespace hpx { namespace lcos { namespace local
{
    // call_once support
    struct once_flag;
}}}

#define HPX_ONCE_INIT hpx::lcos::local::once_flag()

#endif

