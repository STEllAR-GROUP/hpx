//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/util/transform_iterator.hpp>
#include <hpx/util/stencil3_iterator.hpp>

#include <strstream>

void test_stencil3_iterator_full()
{
    std::vector<int> values(10);
    std::iota(std::begin(values), std::end(values), 0);

    auto r = hpx::util::make_stencil3_range(
        values.begin(), values.end(), &values.back(), &values.front());

    typedef std::iterator_traits<decltype(r.first)>::reference reference;

    std::ostringstream str;

    std::for_each(r.first, r.second,
        [&str](reference val)
        {
            using hpx::util::get;
            str << get<0>(val) << get<1>(val) << get<2>(val) << " ";
        });

    HPX_TEST_EQ(str.str(), std::string("901 012 123 234 345 456 567 678 789 890 "));
}

void test_stencil3_iterator()
{
    std::vector<int> values(10);
    std::iota(std::begin(values), std::end(values), 0);

    auto r = hpx::util::make_stencil3_range(values.begin()+1, values.end()-1);

    typedef std::iterator_traits<decltype(r.first)>::reference reference;

    std::ostringstream str;

    std::for_each(r.first, r.second,
        [&str](reference val)
        {
            using hpx::util::get;
            str << get<0>(val) << get<1>(val) << get<2>(val) << " ";
        });

    HPX_TEST_EQ(str.str(), std::string("012 123 234 345 456 567 678 789 "));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_stencil3_iterator_full();
//     test_stencil3_iterator();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
