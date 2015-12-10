//  Copyright (c) 2015 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_make_heap.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_make_heap(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);
    std::make_heap(boost::begin(c), boost::end(c));

    hpx::parallel::make_heap(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)));
}

template <typename ExPolicy, typename IteratorTag>
void test_make_heap_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);
    std::make_heap(boost::begin(c), boost::end(c));

    hpx::future<void> test =
        hpx::parallel::make_heap(p,
                iterator(boost::begin(c)), iterator(boost::end(c)));

    test.wait();
}

template <typename IteratorTag>
void test_make_heap()
{
    using namespace hpx::parallel;

    test_make_heap(seq, IteratorTag());
    test_make_heap(par, IteratorTag());
    //test_make_heap(par_vec, IteratorTag());

    /*
    test_make_heap_async(seq(task), IteratorTag());
    test_make_heap_async(par(task), IteratorTag());

    test_make_heap(execution_policy(seq), IteratorTag());
    test_make_heap(execution_policy(par), IteratorTag());
    test_make_heap(execution_policy(par_vec), IteratorTag());
    test_make_heap(execution_policy(seq(task)), IteratorTag());
    test_make_heap(execution_policy(par(task)), IteratorTag());
    */
}

void make_heap_test()
{
    test_make_heap<std::random_access_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    make_heap_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;
    options_description desc_commandline(
            "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
            boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
            "HPX main exited with a non-zero status");

    return hpx::util::report_errors();
}
