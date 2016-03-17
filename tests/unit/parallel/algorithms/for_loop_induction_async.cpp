//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/include/parallel_for_loop.hpp>

#include <algorithm>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_loop_induction(ExPolicy && policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    auto f = hpx::parallel::for_loop(
        std::forward<ExPolicy>(policy),
        iterator(boost::begin(c)), iterator(boost::end(c)),
        hpx::parallel::induction(0),
        [&d](iterator it, std::size_t i)
        {
            *it = 42;
            d[i] = 42;
        });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    std::for_each(boost::begin(d), boost::end(d),
        [](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_induction_stride(ExPolicy && policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    auto f = hpx::parallel::for_loop(
        std::forward<ExPolicy>(policy),
        iterator(boost::begin(c)), iterator(boost::end(c)),
        hpx::parallel::induction(0),
        hpx::parallel::induction(0, 2),
        [&d](iterator it, std::size_t i, std::size_t j)
        {
            *it = 42;
            d[i] = 42;
            HPX_TEST_EQ(2*i, j);
        });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    std::for_each(boost::begin(d), boost::end(d),
        [](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_induction_life_out(ExPolicy && policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::size_t curr = 0;

    auto f = hpx::parallel::for_loop(
        std::forward<ExPolicy>(policy),
        iterator(boost::begin(c)), iterator(boost::end(c)),
        hpx::parallel::induction(curr),
        [&d](iterator it, std::size_t i)
        {
            *it = 42;
            d[i] = 42;
        });
    f.wait();
    HPX_TEST_EQ(curr, c.size());

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    std::for_each(boost::begin(d), boost::end(d),
        [](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_induction_stride_life_out(ExPolicy && policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::size_t curr1 = 0;
    std::size_t curr2 = 0;

    auto f = hpx::parallel::for_loop(
        std::forward<ExPolicy>(policy),
        iterator(boost::begin(c)), iterator(boost::end(c)),
        hpx::parallel::induction(curr1),
        hpx::parallel::induction(curr2, 2),
        [&d](iterator it, std::size_t i, std::size_t j)
        {
            *it = 42;
            d[i] = 42;
            HPX_TEST_EQ(2*i, j);
        });
    f.wait();
    HPX_TEST_EQ(curr1, c.size());
    HPX_TEST_EQ(curr2, 2*c.size());

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    std::for_each(boost::begin(d), boost::end(d),
        [](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
        });
    HPX_TEST_EQ(count, c.size());
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_loop_induction()
{
    using namespace hpx::parallel;

    test_for_loop_induction(seq(task), IteratorTag());
    test_for_loop_induction(par(task), IteratorTag());

    test_for_loop_induction_stride(seq(task), IteratorTag());
    test_for_loop_induction_stride(par(task), IteratorTag());

    test_for_loop_induction_life_out(seq(task), IteratorTag());
    test_for_loop_induction_life_out(par(task), IteratorTag());

    test_for_loop_induction_stride_life_out(seq(task), IteratorTag());
    test_for_loop_induction_stride_life_out(par(task), IteratorTag());
}

void for_loop_induction_test()
{
    test_for_loop_induction<std::random_access_iterator_tag>();
    test_for_loop_induction<std::forward_iterator_tag>();
    test_for_loop_induction<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_for_loop_induction_idx(ExPolicy && policy)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    auto f = hpx::parallel::for_loop(
        std::forward<ExPolicy>(policy),
        0, c.size(),
        hpx::parallel::induction(0),
        [&c](std::size_t i, std::size_t j)
        {
            c[i] = 42;
            HPX_TEST_EQ(i, j);
        });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy>
void test_for_loop_induction_stride_idx(ExPolicy && policy)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    auto f = hpx::parallel::for_loop(
        std::forward<ExPolicy>(policy),
        0, c.size(),
        hpx::parallel::induction(0),
        hpx::parallel::induction(0, 2),
        [&c](std::size_t i, std::size_t j, std::size_t k)
        {
            c[i] = 42;
            HPX_TEST_EQ(i, j);
            HPX_TEST_EQ(2*i, k);
        });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

void for_loop_induction_test_idx()
{
    using namespace hpx::parallel;

    test_for_loop_induction_idx(seq(task));
    test_for_loop_induction_idx(par(task));

    test_for_loop_induction_stride_idx(seq(task));
    test_for_loop_induction_stride_idx(par(task));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for_loop_induction_test();
    for_loop_induction_test_idx();

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
