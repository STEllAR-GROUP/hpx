//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <typeinfo>

struct foo
{
    int operator()() { return 0; }
    void operator()() const {}
};

int main()
{
    {
        hpx::util::function<int()> fun = foo();

        HPX_TEST(fun.target_type() == typeid(foo));
        HPX_TEST(fun.target<foo>() != nullptr);
    }

    {
        hpx::util::function<int()> fun = foo();

        HPX_TEST(fun.target_type() == typeid(foo const));
        HPX_TEST(fun.target<foo const>() != nullptr);
    }

    return hpx::util::report_errors();
}
