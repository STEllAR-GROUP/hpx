//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define HPX_NO_VERSION_CHECK

#include <hpx/util/lightweight_test.hpp>
#include <iostream>

namespace hpx { namespace util { namespace detail
{
    fixture global_fixture{std::cerr};
}}}

