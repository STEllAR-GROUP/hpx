//  Copyright (c) 2015 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_is_heap.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_is_heap0(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);
    std::make_heap(boost::begin(c), boost::end(c));

    auto test =
        hpx::parallel::is_heap_until(policy,
                iterator(boost::begin(c)), iterator(boost::end(c)));

    HPX_TEST(test == iterator(boost::end(c)));
}

template <typename ExPolicy, typename IteratorTag>
void test_is_heap_async0(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);
    std::make_heap(boost::begin(c), boost::end(c));

    auto test =
        hpx::parallel::is_heap_until(p,
                iterator(boost::begin(c)), iterator(boost::end(c)));

    HPX_TEST(test.get() == iterator(boost::end(c)));
}

template <typename IteratorTag>
void test_is_heap0()
{
    using namespace hpx::parallel;

    test_is_heap0(seq, IteratorTag());
    test_is_heap0(par, IteratorTag());
    test_is_heap0(par_vec, IteratorTag());

    test_is_heap_async0(seq(task), IteratorTag());
    test_is_heap_async0(par(task), IteratorTag());

    test_is_heap0(execution_policy(seq), IteratorTag());
    test_is_heap0(execution_policy(par), IteratorTag());
    test_is_heap0(execution_policy(par_vec), IteratorTag());
    test_is_heap0(execution_policy(seq(task)), IteratorTag());
    test_is_heap0(execution_policy(par(task)), IteratorTag()); 
}

void is_heap_test0()
{
    test_is_heap0<std::random_access_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_is_heap1(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);
    
    auto middle = boost::begin(c) + 
        std::distance(boost::begin(c), boost::end(c)) / 2;
    
    std::make_heap(boost::begin(c), middle);

    auto test =
        hpx::parallel::is_heap_until(policy,
                iterator(boost::begin(c)), iterator(boost::end(c)));

    HPX_TEST(test == iterator(middle));
}

template <typename ExPolicy, typename IteratorTag>
void test_is_heap_async1(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);

    auto middle = boost::begin(c) +
        std::distance(boost::begin(c), boost::end(c)) / 2;

    std::make_heap(boost::begin(c), middle);

    auto test =
        hpx::parallel::is_heap_until(p,
                iterator(boost::begin(c)), iterator(boost::end(c)));

    HPX_TEST(test.get() == iterator(middle));
}

template <typename IteratorTag>
void test_is_heap1()
{
    using namespace hpx::parallel;

    test_is_heap1(seq, IteratorTag());
    test_is_heap1(par, IteratorTag());
    test_is_heap1(par_vec, IteratorTag());

    test_is_heap_async1(seq(task), IteratorTag());
    test_is_heap_async1(par(task), IteratorTag());

    test_is_heap1(execution_policy(seq), IteratorTag());
    test_is_heap1(execution_policy(par), IteratorTag());
    test_is_heap1(execution_policy(par_vec), IteratorTag());
    test_is_heap1(execution_policy(seq(task)), IteratorTag());
    test_is_heap1(execution_policy(par(task)), IteratorTag());
}

void is_heap_test1()
{
    test_is_heap1<std::random_access_iterator_tag>();
}
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    is_heap_test0();
    is_heap_test1();
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
