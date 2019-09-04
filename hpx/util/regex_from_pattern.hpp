//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_REGEX_FROM_PATTERN_DEC_13_2017_1016AM)
#define HPX_UTIL_REGEX_FROM_PATTERN_DEC_13_2017_1016AM

#include <hpx/config.hpp>
#include <hpx/errors.hpp>

#include <string>

namespace hpx { namespace util
{
    HPX_EXPORT std::string regex_from_pattern(std::string const& pattern,
        error_code& ec = throws);
}}

#endif
