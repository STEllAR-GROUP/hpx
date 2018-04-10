//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_adjacent_find.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(2,101);
std::uniform_int_distribution<> dist(2,10005);

template <typename ExPolicy, typename IteratorTag>
void test_adjacent_find(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values about 1
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), dis(gen));

    std::size_t random_pos = dist(gen); //-V101

    c[random_pos] = 1;
    c[random_pos + 1] = 1;

    iterator index = hpx::parallel::adjacent_find(policy,
        iterator(std::begin(c)), iterator(std::end(c)));

    base_iterator test_index = std::begin(c) + random_pos;

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_adjacent_find_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 1
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), dis(gen));

    std::size_t random_pos = dist(gen); //-V101

    c[random_pos] = 1;
    c[random_pos + 1] = 1;

    hpx::future<iterator> f =
        hpx::parallel::adjacent_find(p,
            iterator(std::begin(c)), iterator(std::end(c)));
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = std::begin(c) + random_pos;

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_adjacent_find()
{
    using namespace hpx::parallel;
    test_adjacent_find(execution::seq, IteratorTag());
    test_adjacent_find(execution::par, IteratorTag());
    test_adjacent_find(execution::par_unseq, IteratorTag());

    test_adjacent_find_async(execution::seq(execution::task), IteratorTag());
    test_adjacent_find_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_adjacent_find(execution_policy(execution::seq), IteratorTag());
    test_adjacent_find(execution_policy(execution::par), IteratorTag());
    test_adjacent_find(execution_policy(execution::par_unseq), IteratorTag());

    test_adjacent_find(execution_policy(execution::seq(execution::task)), IteratorTag());
    test_adjacent_find(execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void adjacent_find_test()
{
    test_adjacent_find<std::random_access_iterator_tag>();
    test_adjacent_find<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    adjacent_find_test();
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
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
