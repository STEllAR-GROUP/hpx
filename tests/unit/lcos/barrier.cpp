//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void barrier_test(hpx::id_type const& id, boost::atomic<std::size_t>& c)
{
    ++c;

    // wait for all threads to enter the barrier
    hpx::lcos::stubs::barrier::wait(id);
}

///////////////////////////////////////////////////////////////////////////////
void local_tests(boost::program_options::variables_map& vm)
{
    std::size_t pxthreads = 0;
    if (vm.count("pxthreads"))
        pxthreads = vm["pxthreads"].as<std::size_t>();

    std::size_t iterations = 0;
    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    hpx::id_type here = hpx::find_here();
    for (std::size_t i = 0; i < iterations; ++i)
    {
        using hpx::lcos::barrier;

        // create a barrier waiting on 'count' threads
        barrier b = barrier::create(here, pxthreads + 1);

        boost::atomic<std::size_t> c(0);
        for (std::size_t j = 0; j < pxthreads; ++j)
        {
            hpx::async(hpx::util::bind(&barrier_test, b.get_id(), std::ref(c)));
        }

        b.wait();       // wait for all threads to enter the barrier
        HPX_TEST_EQ(pxthreads, c.load());
    }
}

///////////////////////////////////////////////////////////////////////////////
void remote_test_multiple(boost::program_options::variables_map& vm)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    if (localities.size() == 1)
        return;     // nothing to be done here

    std::size_t iterations = 0;
    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    using hpx::lcos::barrier;

    char const* const barrier_test_name = "/test/barrier/multiple";

    hpx::id_type here = hpx::find_here();
    if (hpx::find_root_locality() == here)
    {
        // create the barrier, register it with AGAS
        barrier b = barrier::create(here, localities.size());
        HPX_TEST(hpx::agas::register_name(hpx::launch::sync, barrier_test_name,
            b.get_id()));

        for (std::size_t i = 0; i != iterations; ++i)
            b.wait();

        HPX_TEST(hpx::agas::unregister_name(hpx::launch::sync, barrier_test_name));
    }
    else
    {
        hpx::id_type id = hpx::agas::on_symbol_namespace_event(
                barrier_test_name, hpx::agas::symbol_ns_bind, true).get();
        hpx::lcos::barrier b(id);

        for (std::size_t i = 0; i != iterations; ++i)
            b.wait();
    }
}

void remote_test_single(boost::program_options::variables_map& vm)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    if (localities.size() == 1)
        return;     // nothing to be done here

    std::size_t iterations = 0;
    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    using hpx::lcos::barrier;

    char const* const barrier_test_name_outer = "/test/barrier/single_outer";

    barrier outer;
    hpx::id_type here = hpx::find_here();
    if (hpx::find_root_locality() == here)
    {
        outer = barrier::create(here, localities.size());
        HPX_TEST(hpx::agas::register_name(hpx::launch::sync,
            barrier_test_name_outer, outer.get_id()));
    }
    else
    {
        hpx::id_type id = hpx::agas::on_symbol_namespace_event(
                barrier_test_name_outer, hpx::agas::symbol_ns_bind, true).get();
        outer = barrier(id);
    }

    char const* const barrier_test_name = "/test/barrier/single";
    for (std::size_t i = 0; i != iterations; ++i)
    {
        if (hpx::find_root_locality() == here)
        {
            // create the barrier, register it with AGAS
            barrier b = barrier::create(here, localities.size());
            HPX_TEST(hpx::agas::register_name(hpx::launch::sync,
                barrier_test_name, b.get_id()));

            b.wait();

            HPX_TEST(hpx::agas::unregister_name(hpx::launch::sync,
                barrier_test_name));
        }
        else
        {
            hpx::id_type id = hpx::agas::on_symbol_namespace_event(
                    barrier_test_name, hpx::agas::symbol_ns_bind, true).get();
            hpx::lcos::barrier b(id);
            b.wait();
        }

        outer.wait();
    }

    HPX_TEST(hpx::agas::unregister_name(hpx::launch::sync,
        barrier_test_name_outer));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    local_tests(vm);

    remote_test_multiple(vm);
    remote_test_multiple(vm);

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using namespace boost::program_options;

    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("pxthreads,T", value<std::size_t>()->default_value(64),
            "the number of PX threads to invoke")
        ("iterations", value<std::size_t>()->default_value(64),
            "the number of times to repeat the test")
        ;

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all",
        "hpx.run_hpx_main!=1"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
      "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

