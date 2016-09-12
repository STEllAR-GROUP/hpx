//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
// This is not demonstrating the recommended way of printing things to the
// console - this can be done easily using hpx::cout instead. The purpose of
// this example is to demonstrate how to use templated plain actions.

#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>

#include <iostream>
#include <string>

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void print(T const& t)
{
    std::cout << t << "\n";
}

// define a plain action encapsulating the print function above
template <typename T>
struct print_action
  : hpx::actions::make_action<void (*)(T const&), &print<T>, print_action<T> >
{};

// define a direct action which is semantically equivalent to the plain action
// above
template <typename T>
struct print_direct_action
  : hpx::actions::make_direct_action<
        void (*)(T const&), &print<T>, print_direct_action<T> >
{};

///////////////////////////////////////////////////////////////////////////////
int main()
{
    hpx::naming::id_type console = hpx::find_root_locality();

    // print something to the console
    {
        print_action<std::string> print_string;
        print_action<int> print_int;
        print_action<double> print_double;

        print_string(console, "Hello World!");
        print_int(console, 42);
        print_double(console, 3.1415);
    }

    // print something to the console using a direct action
    {
        print_direct_action<std::string> print_string;
        print_direct_action<int> print_int;
        print_direct_action<double> print_double;

        print_string(console, "Hello World!");
        print_int(console, 42);
        print_double(console, 3.1415);
    }

    return 0;
}

