//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <hpx/include/iostreams.hpp>

#include <hpx/util/transform_iterator.hpp>
#include <hpx/util/stencil3_iterator.hpp>

///////////////////////////////////////////////////////////////////////////////
// print element to the left of current
template <typename Container>
void print_left_element(Container const& values)
{
    typedef typename Container::value_type value_type;

    auto transformer = hpx::util::detail::make_previous_transformer(
        std::begin(values), &values.back());

    std::for_each(
        hpx::util::make_transform_iterator(std::begin(values), transformer),
        hpx::util::make_transform_iterator(std::end(values), transformer),
        [](value_type d)
        {
            hpx::cout << d << " ";
        });
    hpx::cout << std::endl;
}

// print element to the right of current
template <typename Container>
void print_right_element(Container const& values)
{
    typedef typename Container::value_type value_type;

    auto transformer = hpx::util::detail::make_next_transformer(
        std::end(values)-1, &values.front());

    std::for_each(
        hpx::util::make_transform_iterator(std::begin(values), transformer),
        hpx::util::make_transform_iterator(std::end(values), transformer),
        [](value_type d)
        {
            hpx::cout << d << " ";
        });
    hpx::cout << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
template <typename Container>
void print_stencil_elements(Container const& values)
{
    auto r = hpx::util::make_stencil3_range(
        values.begin(), values.end(), &values.back(), &values.front());

    typedef typename std::iterator_traits<decltype(r.first)>::reference
        reference;

    std::for_each(r.first, r.second, [](reference val)
    {
        using hpx::util::get;
        hpx::cout
            << get<0>(val) << " "
            << get<1>(val) << " "
            << get<2>(val) << std::endl;
    });
    hpx::cout << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // demonstrate use of 'previous' and 'next' transformers
    std::vector<double> values(10);
    std::iota(std::begin(values), std::end(values), 1.0);

    print_left_element(values);
    print_right_element(values);
    hpx::cout << std::endl;

    // demonstrate stencil iterator
    print_stencil_elements(values);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    return hpx::init(argc, argv);
}
