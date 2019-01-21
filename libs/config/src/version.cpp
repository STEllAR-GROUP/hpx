//  Copyright (c) 2019 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/version.hpp>
#include <hpx/pp/stringize.hpp>

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    char const HPX_CHECK_VERSION[] = HPX_PP_STRINGIZE(HPX_CHECK_VERSION);
    char const HPX_CHECK_BOOST_VERSION[] = HPX_PP_STRINGIZE(HPX_CHECK_BOOST_VERSION);
}
