////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017-2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <ciso646>

#if defined(_LIBCPP_VERSION) && (_LIBCPP_VERSION < 6000)
#  error "libc++ inplace_merge implementation is not stable"
#endif

int main()
{
    return 0;
}
