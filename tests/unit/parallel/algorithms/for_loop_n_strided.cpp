//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/include/parallel_for_loop.hpp>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_loop_n_strided(ExPolicy && policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::for_each(boost::begin(c), boost::end(c),
        [](std::size_t& v) -> void
        {
            if (v == 42)
                v = 43;
        });

    int stride = (std::rand() % (c.size() - 1)) + 1; //-V103

    hpx::parallel::for_loop_n_strided(
        std::forward<ExPolicy>(policy),
        iterator(boost::begin(c)), c.size(), stride,
        [](iterator it)
        {
            *it = 42;
        });

    // verify values
    std::size_t count = 0;
    for (std::size_t i = 0; i != c.size(); ++i)
    {
        if (i % stride == 0) //-V104
        {
            HPX_TEST_EQ(c[i], std::size_t(42));
        }
        else
        {
            HPX_TEST_NEQ(c[i], std::size_t(42));
        }
        ++count;
    }
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_n_strided_async(ExPolicy && p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::for_each(boost::begin(c), boost::end(c),
        [](std::size_t& v) -> void
        {
            if (v == 42)
                v = 43;
        });

    int stride = (std::rand() % (c.size() - 1)) + 1; //-V103

    auto f =
        hpx::parallel::for_loop_n_strided(
            std::forward<ExPolicy>(p),
            iterator(boost::begin(c)), c.size(), stride,
            [](iterator it)
            {
                *it = 42;
            });
    f.wait();

    // verify values
    std::size_t count = 0;
    for (std::size_t i = 0; i != c.size(); ++i)
    {
        if (i % stride == 0) //-V104
        {
            HPX_TEST_EQ(c[i], std::size_t(42));
        }
        else
        {
            HPX_TEST_NEQ(c[i], std::size_t(42));
        }
        ++count;
    }
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_for_loop_n_strided()
{
    using namespace hpx::parallel;

    test_for_loop_n_strided(seq, IteratorTag());
    test_for_loop_n_strided(par, IteratorTag());
    test_for_loop_n_strided(par_vec, IteratorTag());

    test_for_loop_n_strided_async(seq(task), IteratorTag());
    test_for_loop_n_strided_async(par(task), IteratorTag());
}

void for_loop_n_strided_test()
{
    test_for_loop_n_strided<std::random_access_iterator_tag>();
    test_for_loop_n_strided<std::forward_iterator_tag>();
    test_for_loop_n_strided<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_for_loop_n_strided_idx(ExPolicy && policy)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::for_each(boost::begin(c), boost::end(c),
        [](std::size_t& v) -> void
        {
            if (v == 42)
                v = 43;
        });

    int stride = (std::rand() % (c.size() - 1)) + 1; //-V103

    hpx::parallel::for_loop_n_strided(
        std::forward<ExPolicy>(policy),
        0, c.size(), stride,
        [&c](std::size_t i)
        {
            c[i] = 42;
        });

    // verify values
    std::size_t count = 0;
    for (std::size_t i = 0; i != c.size(); ++i)
    {
        if (i % stride == 0) //-V104
        {
            HPX_TEST_EQ(c[i], std::size_t(42));
        }
        else
        {
            HPX_TEST_NEQ(c[i], std::size_t(42));
        }
        ++count;
    }
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy>
void test_for_loop_n_strided_idx_async(ExPolicy && p)
{
    typedef std::vector<std::size_t>::iterator base_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::for_each(boost::begin(c), boost::end(c),
        [](std::size_t& v) -> void
        {
            if (v == 42)
                v = 43;
        });

    int stride = (std::rand() % (c.size() - 1)) + 1; //-V103

    auto f =
        hpx::parallel::for_loop_n_strided(
            std::forward<ExPolicy>(p),
            0, c.size(), stride,
            [&c](std::size_t i)
            {
                c[i] = 42;
            });
    f.wait();

    // verify values
    std::size_t count = 0;
    for (std::size_t i = 0; i != c.size(); ++i)
    {
        if (i % stride == 0) //-V104
        {
            HPX_TEST_EQ(c[i], std::size_t(42));
        }
        else
        {
            HPX_TEST_NEQ(c[i], std::size_t(42));
        }
        ++count;
    }
    HPX_TEST_EQ(count, c.size());
}

void for_loop_n_strided_test_idx()
{
    using namespace hpx::parallel;

    test_for_loop_n_strided_idx(seq);
    test_for_loop_n_strided_idx(par);
    test_for_loop_n_strided_idx(par_vec);

    test_for_loop_n_strided_idx_async(seq(task));
    test_for_loop_n_strided_idx_async(par(task));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for_loop_n_strided_test();
    for_loop_n_strided_test_idx();

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
