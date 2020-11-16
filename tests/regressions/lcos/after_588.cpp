//  Copyright (c) 2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <vector>

struct test
    : hpx::components::managed_component_base<test>
{
    test()
      : finished(false)
    {}

    ~test()
    {
        HPX_TEST(finished);
    }

    void pong()
    {}

    HPX_DEFINE_COMPONENT_ACTION(test, pong);

    void ping(hpx::id_type id, std::size_t iterations)
    {
        for(std::size_t i = 0; i != iterations; ++i)
        {
            const std::size_t num_pongs = 50;
            std::vector<hpx::future<void> > futures;
            futures.reserve(num_pongs);
            for(std::size_t j = 0; j != num_pongs; ++j)
            {
                pong_action act;
                futures.push_back(hpx::async(act, id));
            }
            hpx::wait_all(futures);
        }
        finished = true;
    }

    HPX_DEFINE_COMPONENT_ACTION(test, ping);

    std::atomic<bool> finished;
};

typedef hpx::components::managed_component<test> test_component;
HPX_REGISTER_COMPONENT(test_component);

HPX_REGISTER_ACTION(test::pong_action, test_pong_action);
HPX_REGISTER_ACTION(test::ping_action, test_ping_action);

int hpx_main(hpx::program_options::variables_map & vm)
{
    {
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        hpx::id_type there = localities.size() == 1 ? localities[0] : localities[1];

        hpx::id_type id0 = hpx::components::new_<test>(localities[0]).get();
        hpx::id_type id1 = hpx::components::new_<test>(there).get();

        hpx::future<void> f0 = hpx::async(test::ping_action(), id0, id1,
            vm["iterations"].as<std::size_t>());
        hpx::future<void> f1 = hpx::async(test::ping_action(), id1, id0,
            vm["iterations"].as<std::size_t>());
        hpx::wait_all(f0, f1);
    }
    hpx::finalize();

    return hpx::util::report_errors();
}

int main(int argc, char **argv)
{
    hpx::program_options::options_description desc(
        "usage: " HPX_APPLICATION_STRING " [options]");

    desc.add_options()
        ( "iterations",
          hpx::program_options::value<std::size_t>()->default_value(100),
          "number of times to repeat the test")
        ;

    hpx::init_params init_args;
    init_args.desc_cmdline = desc;

    return hpx::init(argc, argv, init_args);
}
#endif
