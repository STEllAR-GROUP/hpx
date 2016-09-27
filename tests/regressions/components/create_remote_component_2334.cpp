//  Copyright (c) 2016 David Pfander
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// verify #2334 is fixed (Cannot construct component with large vector on a
// remote locality)

#include <hpx/hpx_init.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/util.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <utility>
#include <vector>

struct matrix_multiply_multiplier
  : hpx::components::component_base<matrix_multiply_multiplier>
{
    std::vector<double> a_;

    // shouldn't ever get called?
    matrix_multiply_multiplier()
    {
        HPX_TEST(false);
    }

    matrix_multiply_multiplier(std::vector<double> && a)
      : a_(std::move(a))
    {}
};

HPX_REGISTER_COMPONENT(hpx::components::component<matrix_multiply_multiplier>,
    matrix_multiply_multiplier);

int hpx_main()
{
    // works on my computer for N = 4096
    std::size_t const matrix_size = 8192;
    std::vector<double> m(matrix_size * matrix_size);

    std::vector<hpx::id_type> remote_ids = hpx::find_remote_localities();
    HPX_TEST(!remote_ids.empty());

    if (!remote_ids.empty())
    {
        hpx::components::client<matrix_multiply_multiplier> comp =
            hpx::new_<matrix_multiply_multiplier>(remote_ids[0], std::move(m));
    }

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
