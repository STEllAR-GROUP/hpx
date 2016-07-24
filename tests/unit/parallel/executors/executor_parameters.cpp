//  Copyright (c) 2015-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include <boost/range/iterator_range.hpp>
#include <boost/range/functions.hpp>
#include <boost/ref.hpp>

#include "../algorithms/foreach_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ... Parameters>
void parameters_test_impl(Parameters &&... params)
{
    using namespace hpx::parallel;

    typedef std::random_access_iterator_tag iterator_tag;
    test_for_each(seq.with(params...), iterator_tag());
    test_for_each(par.with(params...), iterator_tag());
    test_for_each_async(seq(task).with(params...), iterator_tag());
    test_for_each_async(par(task).with(params...), iterator_tag());

    sequential_executor seq_exec;
    test_for_each(seq.on(seq_exec).with(params...), iterator_tag());
    test_for_each_async(seq(task).on(seq_exec).with(params...), iterator_tag());

    parallel_executor par_exec;
    test_for_each(par.on(par_exec).with(params...), iterator_tag());
    test_for_each_async(par(task).on(par_exec).with(params...), iterator_tag());
}

template <typename ... Parameters>
void parameters_test(Parameters &&... params)
{
    parameters_test_impl(boost::ref(params)...);
    parameters_test_impl(std::ref(params)...);
    parameters_test_impl(std::forward<Parameters>(params)...);
}

void test_dynamic_chunk_size()
{
    {
        hpx::parallel::dynamic_chunk_size dcs;
        parameters_test(dcs);
    }

    {
        hpx::parallel::dynamic_chunk_size dcs(100);
        parameters_test(dcs);
    }
}

void test_static_chunk_size()
{
    {
        hpx::parallel::static_chunk_size scs;
        parameters_test(scs);
    }

    {
        hpx::parallel::static_chunk_size scs(100);
        parameters_test(scs);
    }
}

void test_guided_chunk_size()
{
    {
        hpx::parallel::guided_chunk_size gcs;
        parameters_test(gcs);
    }

    {
        hpx::parallel::guided_chunk_size gcs(100);
        parameters_test(gcs);
    }
}

void test_auto_chunk_size()
{
    {
        hpx::parallel::auto_chunk_size acs;
        parameters_test(acs);
    }

    {
        hpx::parallel::auto_chunk_size acs(boost::chrono::milliseconds(1));
        parameters_test(acs);
    }
}

void test_persistent_auto_chunk_size()
{
    {
        hpx::parallel::persistent_auto_chunk_size pacs;
        parameters_test(pacs);
    }

    {
        hpx::parallel::persistent_auto_chunk_size pacs(
            boost::chrono::milliseconds(0),
            boost::chrono::milliseconds(1));
        parameters_test(pacs);
    }

    {
        hpx::parallel::persistent_auto_chunk_size pacs(
            boost::chrono::milliseconds(0));
        parameters_test(pacs);
    }
}

///////////////////////////////////////////////////////////////////////////////
struct timer_hooks_parameters : hpx::parallel::executor_parameters_tag
{
    timer_hooks_parameters(char const* name)
      : name_(name)
    {}

    void mark_begin_execution()
    {
    }

    void mark_end_execution()
    {
    }

    std::string name_;
};

void test_combined_hooks()
{
    timer_hooks_parameters pacs("time_hooks");
    hpx::parallel::auto_chunk_size acs;

    parameters_test(acs, pacs);
    parameters_test(pacs, acs);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_dynamic_chunk_size();
    test_static_chunk_size();
    test_guided_chunk_size();
    test_auto_chunk_size();
    test_persistent_auto_chunk_size();

    test_combined_hooks();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=" +
            std::to_string(hpx::threads::hardware_concurrency())
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
