//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_adjacent_find.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_adjacent_find(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill vector with random values about 1
    std::iota(boost::begin(c), boost::end(c), (std::rand() % 100) + 2);

    std::size_t random_pos = (std::rand() % 10004) + 2; //-V101

    c[random_pos] = 100000;
    c[random_pos+1] = 1;

    iterator index = hpx::parallel::adjacent_find(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        std::greater<std::size_t>());

    base_iterator test_index = boost::begin(c) + random_pos;

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_adjacent_find_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 1
    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), (std::rand() % 100) + 2);

    std::size_t random_pos = (std::rand() % 10004) + 2; //-V101

    c[random_pos] = 100000;
    c[random_pos+1] = 1;

    hpx::future<iterator> f =
        hpx::parallel::adjacent_find(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            std::greater<std::size_t>());
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = boost::begin(c) + random_pos;
    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_adjacent_find()
{
    using namespace hpx::parallel;
    test_adjacent_find(seq, IteratorTag());
    test_adjacent_find(par, IteratorTag());
    test_adjacent_find(par_vec, IteratorTag());

    test_adjacent_find_async(seq(task), IteratorTag());
    test_adjacent_find_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_adjacent_find(execution_policy(seq), IteratorTag());
    test_adjacent_find(execution_policy(par), IteratorTag());
    test_adjacent_find(execution_policy(par_vec), IteratorTag());

    test_adjacent_find(execution_policy(seq(task)), IteratorTag());
    test_adjacent_find(execution_policy(par(task)), IteratorTag());
#endif
}

void adjacent_find_test()
{
    test_adjacent_find<std::random_access_iterator_tag>();
    test_adjacent_find<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

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
