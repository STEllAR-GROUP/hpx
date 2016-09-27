//  Copyright (c) 2016 David Pfander
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// verify #2323 is fixed (Constructing a vector of components only correctly
// initializes the first component)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/util.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::size_t const test_array_size = 1000ull;
std::size_t const test_num_components = 100ull;

boost::atomic<std::size_t> count(0);

struct component_server : hpx::components::component_base<component_server>
{
    component_server()
    {
        HPX_TEST(false);    // shouldn't be called
    }

    component_server(std::vector<double> const& a)
    {
        HPX_TEST_EQ(a.size(), test_array_size);
        ++count;
    }
};

HPX_REGISTER_COMPONENT(
    hpx::components::component<component_server>, component_server_component);

int hpx_main()
{
    std::vector<double> a(test_array_size);

    typedef hpx::components::client<component_server> client_type;

    hpx::future<std::vector<client_type> > mass_construct =
        hpx::new_<client_type[]>(hpx::find_here(), test_num_components, a);

    for (auto const& c: mass_construct.get())
    {
        c.get();
    }

    HPX_TEST_EQ(count, test_num_components);

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
