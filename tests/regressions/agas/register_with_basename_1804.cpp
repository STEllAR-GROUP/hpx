//  Copyright (c) 2015 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This illustrates the issue as reported by #1804: register_with_basename
// causes hangs

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

static std::string itoa(int i)
{
    std::stringstream buf;
    buf << i;
    return buf.str();
}

struct test_server
  : hpx::components::simple_component_base<test_server>
{
    test_server()
    {}

    hpx::id_type call() const
    {
        return hpx::find_here();
    }
    HPX_DEFINE_COMPONENT_ACTION(test_server, call, call_action);
};

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION(call_action);

std::string gen_name(int source, int target)
{
    std::string basename = "/0/HPXSimulatorUpdateGroupSdfafafasdasd";

    return basename + "/PatchLink/" +
        itoa(source) + "-" +
        itoa(target);
}

void test()
{
    int rank = hpx::get_locality_id();

    std::vector<hpx::id_type> boundingBoxReceivers;
    for (int i = 0; i < 2; ++i) {
        if (i == rank)
            continue;

        for (int j = 0; j < 2; ++j) {
            std::string name = gen_name(j, rank);
            std::cout << "registration: " << name << "\n";

            hpx::id_type id = hpx::new_<test_server>(hpx::find_here()).get();
            hpx::register_with_basename(name, id, 0).get();
            boundingBoxReceivers.push_back(id);
        }
    }

    std::vector<hpx::id_type> boundingBoxAccepters;
    for (int i = 0; i < 2; ++i) {
        if (i != rank)
            continue;

        for (int j = 0; j < 2; ++j) {
            std::string name = gen_name(j, rank);
            std::cout << "lookup: " << name << "\n";
            std::vector<hpx::future<hpx::id_type> > ids =
                hpx::find_all_from_basename(name, 1);
            boundingBoxAccepters.push_back(ids[0].get());
        }
    }

    std::cout << "all done " << rank << "\n";
}

int hpx_main(int argc, char **argv)
{
    // this test must run using 2 localities
    HPX_TEST_EQ(hpx::get_num_localities().get(), 2u);

    test();

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    // We want HPX to run hpx_main() on all localities to avoid the
    // initial overhead caused by broadcasting the work from one to
    // all other localities:
    std::vector<std::string> config(1, "hpx.run_hpx_main!=1");

    HPX_TEST_EQ(hpx::init(argc, argv, config), 0);
    return hpx::util::report_errors();
}
