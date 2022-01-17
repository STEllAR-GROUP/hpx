////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config/export_definitions.hpp>
#include <hpx/config/version.hpp>
#include <hpx/preprocessor/stringize.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    HPX_CORE_EXPORT char const HPX_CHECK_VERSION[] =
        HPX_PP_STRINGIZE(HPX_CHECK_VERSION);
    HPX_CORE_EXPORT char const HPX_CHECK_BOOST_VERSION[] =
        HPX_PP_STRINGIZE(HPX_CHECK_BOOST_VERSION);
}    // namespace hpx
