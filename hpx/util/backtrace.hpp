//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_BACKTRACE_DEC_26_0120PM)
#define HPX_UTIL_BACKTRACE_DEC_26_0120PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_STACKTRACES)
#  include <hpx/util/backtrace/backtrace.hpp>
#else

namespace hpx { namespace util
{
    inline std::string trace()
    {
        return "";
    }
    struct backtrace {};
}}

#endif

#endif

