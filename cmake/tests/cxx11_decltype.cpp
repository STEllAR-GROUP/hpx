////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2008 Beman Dawes
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

struct test_class { test_class() {} };

test_class get_test_class() { return test_class(); }

// N3276
template <typename T> struct is_int { static const bool value = false; };
template <> struct is_int<int> { static const bool value = true; };

template <typename T>
struct instantiation_guard
{
    int check_not_instantiated[!is_int<T>::value ? 1 : -1];
};

template <typename T>
instantiation_guard<T> f(T) { return instantiation_guard<T>{}; }

int main()
{
    int i;
    decltype(i) j(0);
    decltype(get_test_class()) k;

    // N2376
    typedef decltype(f(0)) result_type;
}
