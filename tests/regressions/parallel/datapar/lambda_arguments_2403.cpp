//  Copyright (c) 2016 David Pfander
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/datapar.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <vector>

int hpx_main()
{
    std::vector<double> large(64);

    auto zip_it_begin = hpx::util::make_zip_iterator(large.begin());
    auto zip_it_end = hpx::util::make_zip_iterator(large.end());

    hpx::parallel::for_each(
        hpx::parallel::execution::datapar, zip_it_begin, zip_it_end,
        [](auto& t) -> void
        {
            hpx::util::get<0>(t) = 10.0;
        });

    HPX_TEST_EQ(
        std::count(large.begin(), large.end(), 10.0),
        std::ptrdiff_t(large.size()));

    return hpx::finalize();    // Handles HPX shutdown
}

int main(int argc, char **argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
