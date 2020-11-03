//  Copyright (c) 2017 Christopher Hinz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/include/actions.hpp>

int foo()
{
    return 42;
}

HPX_PLAIN_ACTION(foo);

int main()
{
    return 0;
}
#endif
