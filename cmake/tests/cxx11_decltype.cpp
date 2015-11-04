////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2008 Beman Dawes
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

struct test_class { test_class() {} };

test_class get_test_class() { return test_class(); }

int main()
{
    int i;
    decltype(i) j(0);
    decltype(get_test_class()) k;
}
