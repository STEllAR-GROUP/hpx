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

///////////////////////////////////////////////////////////////////////////////
// dereference element to the left of current
void test_left_element_full()
{
    // demonstrate use of 'previous' and 'next' transformers
    std::vector<int> values(10);
    std::iota(std::begin(values), std::end(values), 0);

    auto transformer = hpx::util::detail::make_previous_transformer(
        std::begin(values), &values.back());

    std::ostringstream str;

    std::for_each(
        hpx::util::make_transform_iterator(std::begin(values), transformer),
        hpx::util::make_transform_iterator(std::end(values), transformer),
        [&str](int d)
        {
            str << d << " ";
        });

    HPX_TEST_EQ(str.str(), std::string("9 0 1 2 3 4 5 6 7 8 "));
}

void test_left_element()
{
    // demonstrate use of 'previous' and 'next' transformers
    std::vector<int> values(10);
    std::iota(std::begin(values), std::end(values), 0);

    auto transformer = hpx::util::detail::make_previous_transformer();

    std::ostringstream str;

    std::for_each(
        hpx::util::make_transform_iterator(std::begin(values)+1, transformer),
        hpx::util::make_transform_iterator(std::end(values), transformer),
        [&str](int d)
        {
            str << d << " ";
        });

    HPX_TEST_EQ(str.str(), std::string("0 1 2 3 4 5 6 7 8 "));
}

// dereference element to the right of current
void test_right_element_full()
{
    // demonstrate use of 'previous' and 'next' transformers
    std::vector<int> values(10);
    std::iota(std::begin(values), std::end(values), 0);

    auto transformer = hpx::util::detail::make_next_transformer(
        std::end(values)-1, &values.front());

    std::ostringstream str;

    std::for_each(
        hpx::util::make_transform_iterator(std::begin(values), transformer),
        hpx::util::make_transform_iterator(std::end(values), transformer),
        [&str](int d)
        {
            str << d << " ";
        });

    HPX_TEST_EQ(str.str(), std::string("1 2 3 4 5 6 7 8 9 0 "));
}

void test_right_element()
{
    // demonstrate use of 'previous' and 'next' transformers
    std::vector<int> values(10);
    std::iota(std::begin(values), std::end(values), 0);

    auto transformer = hpx::util::detail::make_next_transformer();

    std::ostringstream str;

    std::for_each(
        hpx::util::make_transform_iterator(std::begin(values), transformer),
        hpx::util::make_transform_iterator(std::end(values)-1, transformer),
        [&str](int d)
        {
            str << d << " ";
        });

    HPX_TEST_EQ(str.str(), std::string("1 2 3 4 5 6 7 8 9 "));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_left_element_full();
    test_right_element_full();

    test_left_element();
    test_right_element();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
