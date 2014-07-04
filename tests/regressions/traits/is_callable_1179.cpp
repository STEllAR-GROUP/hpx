//  Copyright (c) 2014 Erik Schnetter
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/lightweight_test.hpp>

struct s {
    int f() const { return 42; }
};

struct p {
    s *x;
    s &operator*() const { return *x; }
};

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    typedef int (s::*mem_fun_ptr)() const;
    HPX_TEST_MSG(hpx::traits::is_callable<mem_fun_ptr(p)>::value, "mem-fun-ptr");

    HPX_TEST(hpx::util::invoke(&s::f, p()) == 42);

    return hpx::util::report_errors();
}
