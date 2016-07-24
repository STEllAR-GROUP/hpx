//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>
#include <vector>

#include <boost/ref.hpp>
#include <boost/atomic.hpp>

#include "../algorithms/foreach_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename Parameters>
void chunk_size_test_seq(Parameters && params)
{
    using namespace hpx::parallel;

    typedef std::random_access_iterator_tag iterator_tag;
    test_for_each(seq.with(boost::ref(params)), iterator_tag());
    test_for_each_async(seq(task).with(boost::ref(params)), iterator_tag());

    sequential_executor seq_exec;
    test_for_each(seq.on(seq_exec).with(boost::ref(params)), iterator_tag());
    test_for_each_async(seq(task).on(seq_exec).with(boost::ref(params)), iterator_tag());
}

template <typename Parameters>
void chunk_size_test_par(Parameters && params)
{
    using namespace hpx::parallel;

    typedef std::random_access_iterator_tag iterator_tag;
    test_for_each(par.with(boost::ref(params)), iterator_tag());
    test_for_each_async(par(task).with(boost::ref(params)), iterator_tag());

    parallel_executor par_exec;
    test_for_each(par.on(par_exec).with(boost::ref(params)), iterator_tag());
    test_for_each_async(par(task).on(par_exec).with(boost::ref(params)), iterator_tag());
}

struct timer_hooks_parameters : hpx::parallel::executor_parameters_tag
{
    timer_hooks_parameters(char const* name)
      : name_(name), time_(0), count_(0)
    {}

    void mark_begin_execution()
    {
        ++count_;
        time_ = hpx::util::high_resolution_clock::now();
    }

    void mark_end_execution()
    {
        time_ = hpx::util::high_resolution_clock::now() - time_;
        ++count_;
    }

    std::string name_;
    boost::uint64_t time_;
    boost::atomic<std::size_t> count_;
};

void test_timer_hooks()
{
    timer_hooks_parameters pacs("time_hooks");

    chunk_size_test_seq(pacs);
    HPX_TEST_EQ(pacs.count_, std::size_t(8));

    chunk_size_test_par(pacs);
    HPX_TEST_EQ(pacs.count_, std::size_t(16));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_timer_hooks();

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
