//  Copyright (c) 2014 Erik Schnetter
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/lightweight_test.hpp>

struct s {
    int f() const { return 42; }
};

struct p {
    s x;
    s const& operator*() const { return x; }
};

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using hpx::traits::is_callable;
    using hpx::util::invoke;

    typedef int (s::*mem_fun_ptr)();
    HPX_TEST_MSG((is_callable<mem_fun_ptr(p)>::value == false), "mem-fun-ptr");

    typedef int (s::*const_mem_fun_ptr)() const;
    HPX_TEST_MSG((is_callable<const_mem_fun_ptr(p)>::value == true),
        "const-mem-fun-ptr");

    HPX_TEST(invoke(&s::f, p()) == 42);

    return hpx::util::report_errors();
}
