//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_transform_scan.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_exclusive_scan(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op = [val](std::size_t v1, std::size_t v2) { return v1 + v2; };
    auto conv = [](std::size_t val) { return 2*val; };

    hpx::parallel::transform_exclusive_scan(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
        conv, val, op);

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::detail::sequential_transform_exclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), conv, val, op);

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_exclusive_scan_async(ExPolicy const& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op = [val](std::size_t v1, std::size_t v2) { return v1 + v2; };
    auto conv = [](std::size_t val) { return 2*val; };

    hpx::future<void> f =
        hpx::parallel::transform_exclusive_scan(p,
            iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
            conv, val, op);
    f.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::detail::sequential_transform_exclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), conv, val, op);

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename IteratorTag>
void test_transform_exclusive_scan()
{
    using namespace hpx::parallel;

    test_transform_exclusive_scan(seq, IteratorTag());
    test_transform_exclusive_scan(par, IteratorTag());
    test_transform_exclusive_scan(par_vec, IteratorTag());

    test_transform_exclusive_scan_async(seq(task), IteratorTag());
    test_transform_exclusive_scan_async(par(task), IteratorTag());

    test_transform_exclusive_scan(execution_policy(seq), IteratorTag());
    test_transform_exclusive_scan(execution_policy(par), IteratorTag());
    test_transform_exclusive_scan(execution_policy(par_vec), IteratorTag());

    test_transform_exclusive_scan(execution_policy(seq(task)), IteratorTag());
    test_transform_exclusive_scan(execution_policy(par(task)), IteratorTag());
}

void transform_exclusive_scan_test()
{
    test_transform_exclusive_scan<std::random_access_iterator_tag>();
    test_transform_exclusive_scan<std::forward_iterator_tag>();
    test_transform_exclusive_scan<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_exclusive_scan_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    try {
        hpx::parallel::transform_exclusive_scan(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d),
            [](std::size_t val) { return val; },
            std::size_t(0),
            [](std::size_t v1, std::size_t v2)
            {
                return throw std::runtime_error("test"), v1 + v2;
            });

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

template <typename ExPolicy, typename IteratorTag>
void test_transform_exclusive_scan_exception_async(ExPolicy const& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::transform_exclusive_scan(p,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                boost::begin(d),
                [](std::size_t val) { return val; },
                std::size_t(0),
                [](std::size_t v1, std::size_t v2)
                {
                    return throw std::runtime_error("test"), v1 + v2;
                });

        returned_from_algorithm = true;
        f.get();

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

template <typename IteratorTag>
void test_transform_exclusive_scan_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_exclusive_scan_exception(seq, IteratorTag());
    test_transform_exclusive_scan_exception(par, IteratorTag());

    test_transform_exclusive_scan_exception_async(seq(task), IteratorTag());
    test_transform_exclusive_scan_exception_async(par(task), IteratorTag());

    test_transform_exclusive_scan_exception(execution_policy(seq), IteratorTag());
    test_transform_exclusive_scan_exception(execution_policy(par), IteratorTag());

    test_transform_exclusive_scan_exception(execution_policy(seq(task)), IteratorTag());
    test_transform_exclusive_scan_exception(execution_policy(par(task)), IteratorTag());
}

void transform_exclusive_scan_exception_test()
{
    test_transform_exclusive_scan_exception<std::random_access_iterator_tag>();
    test_transform_exclusive_scan_exception<std::forward_iterator_tag>();
    test_transform_exclusive_scan_exception<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_exclusive_scan_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    try {
        hpx::parallel::transform_exclusive_scan(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d),
            [](std::size_t val) { return val; },
            std::size_t(0),
            [](std::size_t v1, std::size_t v2)
            {
                return throw std::bad_alloc(), v1 + v2;
            });

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_exception = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_exclusive_scan_bad_alloc_async(ExPolicy const& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::transform_exclusive_scan(p,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                boost::begin(d),
                [](std::size_t val) { return val; },
                std::size_t(0),
                [](std::size_t v1, std::size_t v2)
                {
                    return throw std::bad_alloc(), v1 + v2;
                });

        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_exception = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_transform_exclusive_scan_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_exclusive_scan_bad_alloc(seq, IteratorTag());
    test_transform_exclusive_scan_bad_alloc(par, IteratorTag());

    test_transform_exclusive_scan_bad_alloc_async(seq(task), IteratorTag());
    test_transform_exclusive_scan_bad_alloc_async(par(task), IteratorTag());

    test_transform_exclusive_scan_bad_alloc(execution_policy(seq), IteratorTag());
    test_transform_exclusive_scan_bad_alloc(execution_policy(par), IteratorTag());

    test_transform_exclusive_scan_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_transform_exclusive_scan_bad_alloc(execution_policy(par(task)), IteratorTag());
}

void transform_exclusive_scan_bad_alloc_test()
{
    test_transform_exclusive_scan_bad_alloc<std::random_access_iterator_tag>();
    test_transform_exclusive_scan_bad_alloc<std::forward_iterator_tag>();
    test_transform_exclusive_scan_bad_alloc<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    transform_exclusive_scan_test();

    transform_exclusive_scan_exception_test();
    transform_exclusive_scan_bad_alloc_test();

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
