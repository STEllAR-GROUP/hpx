////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

struct dummy_type
{
    ~dummy_type(){};
};

thread_local dummy_type dummy;

int main()
{
    return 0;
}
