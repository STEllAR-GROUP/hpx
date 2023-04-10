//  Copyright (c) 2013 Vinay C Amatya
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/functional.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/iterator_support/iterator_range.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

template <typename T>
struct print_obj
{
public:
    void operator()(T const& input) const
    {
        std::cout << input << std::endl;
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    typedef std::vector<std::size_t> vector_type;
    typedef std::vector<std::size_t>::iterator iterator_type;

    vector_type v1, v2;

    std::size_t count = 0;
    while (count < 2000)
    {
        v1.push_back(count);
        ++count;
    }

    iterator_type itr_b = v1.begin();
    iterator_type itr_e = v1.end();

    typedef hpx::util::iterator_range<iterator_type> range_type;

    range_type my_range(itr_b, itr_e);

    v2.reserve(v1.size());
    iterator_type itr_o = v2.begin();
    (void) itr_o;

    std::size_t i = 0;
    for (std::size_t const& v : my_range)
    {
        hpx::async(hpx::util::protect(print_obj<std::size_t>()), v);
        ++i;
    }

    return hpx::local::finalize();    // Handles HPX shutdown
}

int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);
}
