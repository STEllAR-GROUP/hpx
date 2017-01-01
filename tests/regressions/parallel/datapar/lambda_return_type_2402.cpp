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
        [](auto t)
        {
            using comp_type = typename hpx::util::tuple_element<0, decltype(t)>::type;
            using var_type = typename hpx::util::decay<comp_type>::type;

            var_type mass_density = 0.0;
            mass_density(mass_density > 0.0) = 7.0;

            HPX_TEST(all_of(mass_density == 0.0));
        });

    return hpx::finalize();    // Handles HPX shutdown
}

int main(int argc, char **argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
