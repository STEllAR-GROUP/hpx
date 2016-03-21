//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <iostream>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iterator>

#include <boost/ref.hpp>

#include "../algorithms/foreach_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
struct persistent_parameters : hpx::parallel::executor_parameters_tag
{
public:
    persistent_parameters()
      : chunk_size_(0)
    {}

    persistent_parameters(persistent_parameters const&)
      : chunk_size_(0)
    {
    }

    persistent_parameters& operator=(persistent_parameters const&)
    {
        chunk_size_ = 0;
        return *this;
    }

    template <typename Executor, typename F>
    std::size_t get_chunk_size(Executor& exec, F &&, std::size_t count)
    {
        if (0 == chunk_size_)
        {
            std::size_t const cores =
                hpx::parallel::executor_information_traits<Executor>::
                    processing_units_count(exec, *this);

            chunk_size_ = (count + cores - 1) / cores;
        }

        return chunk_size_;
    }

    std::size_t chunk_size_;
};

void test_persistent_executitor_parameters()
{
    using namespace hpx::parallel;

    typedef std::random_access_iterator_tag iterator_tag;
    {
        persistent_parameters p;
        auto policy = par.with(p);
        test_for_each(policy, iterator_tag());
        HPX_TEST_NEQ(policy.parameters().chunk_size_, std::size_t(0));
    }

    {
        persistent_parameters p;
        auto policy = par(task).with(p);
        test_for_each_async(policy, iterator_tag());
        HPX_TEST_NEQ(policy.parameters().chunk_size_, std::size_t(0));
    }

    parallel_executor par_exec;
    {
        persistent_parameters p;
        auto policy = par.on(par_exec).with(p);
        test_for_each(policy, iterator_tag());
        HPX_TEST_NEQ(policy.parameters().chunk_size_, std::size_t(0));
    }

    {
        persistent_parameters p;
        auto policy = par(task).on(par_exec).with(p);
        test_for_each_async(policy, iterator_tag());
        HPX_TEST_NEQ(policy.parameters().chunk_size_, std::size_t(0));
    }
}

void test_persistent_executitor_parameters_ref()
{
    using namespace hpx::parallel;

    typedef std::random_access_iterator_tag iterator_tag;
    {
        persistent_parameters p;
        test_for_each(par.with(boost::ref(p)), iterator_tag());
        HPX_TEST_NEQ(p.chunk_size_, std::size_t(0));
    }

    {
        persistent_parameters p;
        test_for_each_async(par(task).with(boost::ref(p)), iterator_tag());
        HPX_TEST_NEQ(p.chunk_size_, std::size_t(0));
    }

    parallel_executor par_exec;
    {
        persistent_parameters p;
        test_for_each(par.on(par_exec).with(boost::ref(p)), iterator_tag());
        HPX_TEST_NEQ(p.chunk_size_, std::size_t(0));
    }

    {
        persistent_parameters p;
        test_for_each_async(par(task).on(par_exec).with(boost::ref(p)),
            iterator_tag());
        HPX_TEST_NEQ(p.chunk_size_, std::size_t(0));
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = static_cast<unsigned int>(std::time(0));
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_persistent_executitor_parameters();
    test_persistent_executitor_parameters_ref();

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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
