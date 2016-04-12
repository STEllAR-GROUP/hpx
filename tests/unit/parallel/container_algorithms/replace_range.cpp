//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_replace.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <string>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_replace(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef test::test_container<std::vector<std::size_t>, IteratorTag> test_vector;

    test_vector c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c.base()), boost::end(c.base()), std::rand());
    std::copy(boost::begin(c.base()), boost::end(c.base()), boost::begin(d));

    std::size_t idx = std::rand() % c.size(); //-V104

    hpx::parallel::replace(policy, c, c[idx], c[idx]+1);

    std::replace(boost::begin(d), boost::end(d), d[idx], d[idx]+1);

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c.base()), boost::end(c.base()),
        boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_replace_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef test::test_container<std::vector<std::size_t>, IteratorTag> test_vector;

    test_vector c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c.base()), boost::end(c.base()), std::rand());
    std::copy(boost::begin(c.base()), boost::end(c.base()), boost::begin(d));

    std::size_t idx = std::rand() % c.size(); //-V104

    auto f = hpx::parallel::replace(p, c, c[idx], c[idx]+1);
    f.wait();

    std::replace(boost::begin(d), boost::end(d), d[idx], d[idx]+1);

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c.base()), boost::end(c.base()),
        boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename IteratorTag>
void test_replace()
{
    using namespace hpx::parallel;
    test_replace(seq, IteratorTag());
    test_replace(par, IteratorTag());
    test_replace(par_vec, IteratorTag());

    test_replace_async(seq(task), IteratorTag());
    test_replace_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_replace(execution_policy(seq), IteratorTag());
    test_replace(execution_policy(par), IteratorTag());
    test_replace(execution_policy(par_vec), IteratorTag());

    test_replace(execution_policy(seq(task)), IteratorTag());
    test_replace(execution_policy(par(task)), IteratorTag());
#endif
}

void replace_test()
{
    test_replace<std::random_access_iterator_tag>();
    test_replace<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_replace_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::replace(policy,
            boost::make_iterator_range(
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(boost::end(c))),
            std::size_t(42), std::size_t(43));
        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_replace_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::replace(p,
                boost::make_iterator_range(
                    decorated_iterator(
                        boost::begin(c),
                        [](){ throw std::runtime_error("test"); }),
                    decorated_iterator(boost::end(c))),
                std::size_t(42), std::size_t(43));
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_replace_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_replace_exception(seq, IteratorTag());
    test_replace_exception(par, IteratorTag());

    test_replace_exception_async(seq(task), IteratorTag());
    test_replace_exception_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_replace_exception(execution_policy(seq), IteratorTag());
    test_replace_exception(execution_policy(par), IteratorTag());

    test_replace_exception(execution_policy(seq(task)), IteratorTag());
    test_replace_exception(execution_policy(par(task)), IteratorTag());
#endif
}

void replace_exception_test()
{
    test_replace_exception<std::random_access_iterator_tag>();
    test_replace_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_replace_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::replace(policy,
            boost::make_iterator_range(
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(boost::end(c))),
            std::size_t(42), std::size_t(43));
        HPX_TEST(false);
    }
    catch (std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag>
void test_replace_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::replace(p,
                boost::make_iterator_range(
                    decorated_iterator(
                        boost::begin(c),
                        [](){ throw std::bad_alloc(); }),
                    decorated_iterator(boost::end(c))),
                std::size_t(42), std::size_t(43));
        returned_from_algorithm = true;
        f.get();

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

template <typename IteratorTag>
void test_replace_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_replace_bad_alloc(seq, IteratorTag());
    test_replace_bad_alloc(par, IteratorTag());

    test_replace_bad_alloc_async(seq(task), IteratorTag());
    test_replace_bad_alloc_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_replace_bad_alloc(execution_policy(seq), IteratorTag());
    test_replace_bad_alloc(execution_policy(par), IteratorTag());

    test_replace_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_replace_bad_alloc(execution_policy(par(task)), IteratorTag());
#endif
}

void replace_bad_alloc_test()
{
    test_replace_bad_alloc<std::random_access_iterator_tag>();
    test_replace_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    replace_test();
    replace_exception_test();
    replace_bad_alloc_test();
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
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
