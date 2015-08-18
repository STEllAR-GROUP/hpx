//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>
#include <boost/atomic.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each_n(ExPolicy policy, IteratorTag, Proj && proj)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::fill(boost::begin(c), boost::end(c), std::size_t(42));

    boost::atomic<std::size_t> count(0);

    iterator result =
        hpx::parallel::for_each_n(policy,
            iterator(boost::begin(c)), c.size(),
            [&count, &proj](std::size_t v) {
                HPX_TEST_EQ(v, proj(std::size_t(42)));
                ++count;
            },
            proj);

    HPX_TEST(result == iterator(boost::end(c)));
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each_n_async(ExPolicy p, IteratorTag, Proj && proj)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::fill(boost::begin(c), boost::end(c), std::size_t(42));

    boost::atomic<std::size_t> count(0);

    hpx::future<iterator> f =
        hpx::parallel::for_each_n(p,
            iterator(boost::begin(c)), c.size(),
            [&count, &proj](std::size_t v) {
                HPX_TEST_EQ(v, proj(std::size_t(42)));
                ++count;
            },
            proj);

    HPX_TEST(f.get() == iterator(boost::end(c)));
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag, typename Proj>
void test_for_each_n()
{
    using namespace hpx::parallel;

    test_for_each_n(seq, IteratorTag(), Proj());
    test_for_each_n(par, IteratorTag(), Proj());
    test_for_each_n(par_vec, IteratorTag(), Proj());

    test_for_each_n_async(seq(task), IteratorTag(), Proj());
    test_for_each_n_async(par(task), IteratorTag(), Proj());

    test_for_each_n(execution_policy(seq), IteratorTag(), Proj());
    test_for_each_n(execution_policy(par), IteratorTag(), Proj());
    test_for_each_n(execution_policy(par_vec), IteratorTag(), Proj());

    test_for_each_n(execution_policy(seq(task)), IteratorTag(), Proj());
    test_for_each_n(execution_policy(par(task)), IteratorTag(), Proj());
}

template <typename Proj>
void for_each_n_test()
{
    test_for_each_n<std::random_access_iterator_tag, Proj>();
    test_for_each_n<std::forward_iterator_tag, Proj>();
    test_for_each_n<std::input_iterator_tag, Proj>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each_n_exception(ExPolicy policy, IteratorTag, Proj && proj)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::for_each_n(policy,
            iterator(boost::begin(c)), c.size(),
            [](std::size_t v) { throw std::runtime_error("test"); },
            proj);

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each_n_exception_async(ExPolicy p, IteratorTag, Proj && proj)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<iterator> f =
            hpx::parallel::for_each_n(p,
                iterator(boost::begin(c)), c.size(),
                [](std::size_t v) { throw std::runtime_error("test"); },
                proj);
        returned_from_algorithm = true;
        f.get();    // rethrow exception

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag, typename Proj>
void test_for_each_n_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_for_each_n_exception(seq, IteratorTag(), Proj());
    test_for_each_n_exception(par, IteratorTag(), Proj());

    test_for_each_n_exception_async(seq(task), IteratorTag(), Proj());
    test_for_each_n_exception_async(par(task), IteratorTag(), Proj());

    test_for_each_n_exception(execution_policy(seq), IteratorTag(), Proj());
    test_for_each_n_exception(execution_policy(par), IteratorTag(), Proj());

    test_for_each_n_exception(execution_policy(seq(task)), IteratorTag(), Proj());
    test_for_each_n_exception(execution_policy(par(task)), IteratorTag(), Proj());
}

template <typename Proj>
void for_each_n_exception_test()
{
    test_for_each_n_exception<std::random_access_iterator_tag, Proj>();
    test_for_each_n_exception<std::forward_iterator_tag, Proj>();
    test_for_each_n_exception<std::input_iterator_tag, Proj>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each_n_bad_alloc(ExPolicy policy, IteratorTag, Proj && proj)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::for_each_n(policy,
            iterator(boost::begin(c)), c.size(),
            [](std::size_t v) { throw std::bad_alloc(); },
            proj);

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag, typename Proj>
void test_for_each_n_bad_alloc_async(ExPolicy p, IteratorTag, Proj && proj)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<iterator> f =
            hpx::parallel::for_each_n(p,
                iterator(boost::begin(c)), c.size(),
                [](std::size_t v) { throw std::bad_alloc(); },
                proj);
        returned_from_algorithm = true;
        f.get();    // rethrow bad_alloc

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag, typename Proj>
void test_for_each_n_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_for_each_n_bad_alloc(seq, IteratorTag(), Proj());
    test_for_each_n_bad_alloc(par, IteratorTag(), Proj());

    test_for_each_n_bad_alloc_async(seq(task), IteratorTag(), Proj());
    test_for_each_n_bad_alloc_async(par(task), IteratorTag(), Proj());

    test_for_each_n_bad_alloc(execution_policy(seq), IteratorTag(), Proj());
    test_for_each_n_bad_alloc(execution_policy(par), IteratorTag(), Proj());

    test_for_each_n_bad_alloc(execution_policy(seq(task)), IteratorTag(), Proj());
    test_for_each_n_bad_alloc(execution_policy(par(task)), IteratorTag(), Proj());
}

template <typename Proj>
void for_each_n_bad_alloc_test()
{
    test_for_each_n_bad_alloc<std::random_access_iterator_tag, Proj>();
    test_for_each_n_bad_alloc<std::forward_iterator_tag, Proj>();
    test_for_each_n_bad_alloc<std::input_iterator_tag, Proj>();
}

///////////////////////////////////////////////////////////////////////////////
struct projection_square
{
    template <typename T>
    T operator()(T const& val) const
    {
        return val * val;
    }
};

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for_each_n_test<hpx::parallel::util::projection_identity>();
    for_each_n_exception_test<hpx::parallel::util::projection_identity>();
    for_each_n_bad_alloc_test<hpx::parallel::util::projection_identity>();

    for_each_n_test<projection_square>();
    for_each_n_exception_test<projection_square>();
    for_each_n_bad_alloc_test<projection_square>();

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


