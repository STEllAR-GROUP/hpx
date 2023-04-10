//  Copyright (c) 2016 David Pfander
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/datapar.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <vector>

int hpx_main()
{
    std::vector<double> large(64);

    auto zip_it_begin = hpx::util::zip_iterator(large.begin());
    auto zip_it_end = hpx::util::zip_iterator(large.end());

    hpx::for_each(hpx::execution::par_simd, zip_it_begin, zip_it_end,
        [](auto& t) -> void { hpx::get<0>(t) = 10.0; });

    HPX_TEST_EQ(std::count(large.begin(), large.end(), 10.0),
        std::ptrdiff_t(large.size()));

    return hpx::local::finalize();    // Handles HPX shutdown
}

int main(int argc, char** argv)
{
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv), 0);
    return hpx::util::report_errors();
}
