//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_set_operations.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_includes1(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);

    std::size_t start = std::rand() % c1.size();
    std::size_t end = start + (std::rand() % (c1.size() - start));

    HPX_ASSERT(start <= end);

    base_iterator start_it = boost::next(boost::begin(c1), start);
    base_iterator end_it = boost::next(boost::begin(c1), end);

    {
        bool result = hpx::parallel::includes(policy,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            start_it, end_it);

        bool expected = std::includes(
            boost::begin(c1), boost::end(c1),
            start_it, end_it);

        // verify values
        HPX_TEST_EQ(result, expected);
    }

    {
        std::vector<std::size_t> c2;
        std::copy(start_it, end_it, std::back_inserter(c2));

        if (!c2.empty())
        {
            ++c2[std::rand() % c2.size()]; //-V104

            bool result = hpx::parallel::includes(policy,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                boost::begin(c2), boost::end(c2));

            bool expected = std::includes(boost::begin(c1), boost::end(c1),
                boost::begin(c2), boost::end(c2));

            // verify values
            HPX_TEST_EQ(result, expected);
        }
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_includes1_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);

    std::size_t start = std::rand() % c1.size();
    std::size_t end = start + (std::rand() % (c1.size() - start));

    HPX_ASSERT(start <= end);

    base_iterator start_it = boost::next(boost::begin(c1), start);
    base_iterator end_it = boost::next(boost::begin(c1), end);

    {
        hpx::future<bool> result =
            hpx::parallel::includes(p,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                start_it, end_it);
        result.wait();

        bool expected = std::includes(
            boost::begin(c1), boost::end(c1),
            start_it, end_it);

        // verify values
        HPX_TEST_EQ(result.get(), expected);
    }

    {
        std::vector<std::size_t> c2;
        std::copy(start_it, end_it, std::back_inserter(c2));

        if (!c2.empty())
        {
            ++c2[std::rand() % c2.size()]; //-V104

            hpx::future<bool> result =
                hpx::parallel::includes(p,
                    iterator(boost::begin(c1)), iterator(boost::end(c1)),
                    boost::begin(c2), boost::end(c2));
            result.wait();

            bool expected = std::includes(boost::begin(c1), boost::end(c1),
                boost::begin(c2), boost::end(c2));

            // verify values
            HPX_TEST_EQ(result.get(), expected);
        }
    }
}

template <typename IteratorTag>
void test_includes1()
{
    using namespace hpx::parallel;

    test_includes1(seq, IteratorTag());
    test_includes1(par, IteratorTag());
    test_includes1(par_vec, IteratorTag());

    test_includes1_async(seq(task), IteratorTag());
    test_includes1_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_includes1(execution_policy(seq), IteratorTag());
    test_includes1(execution_policy(par), IteratorTag());
    test_includes1(execution_policy(par_vec), IteratorTag());

    test_includes1(execution_policy(seq(task)), IteratorTag());
    test_includes1(execution_policy(par(task)), IteratorTag());
#endif
}

void includes_test1()
{
    test_includes1<std::random_access_iterator_tag>();
    test_includes1<std::forward_iterator_tag>();
    test_includes1<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_includes2(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);

    std::size_t start = std::rand() % c1.size();
    std::size_t end = start + (std::rand() % (c1.size() - start));

    HPX_ASSERT(start <= end);

    base_iterator start_it = boost::next(boost::begin(c1), start);
    base_iterator end_it = boost::next(boost::begin(c1), end);

    {
        bool result = hpx::parallel::includes(policy,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            start_it, end_it, std::less<std::size_t>());

        bool expected = std::includes(
            boost::begin(c1), boost::end(c1),
            start_it, end_it, std::less<std::size_t>());

        // verify values
        HPX_TEST_EQ(result, expected);
    }

    {
        std::vector<std::size_t> c2;
        std::copy(start_it, end_it, std::back_inserter(c2));

        if (!c2.empty())
        {
            ++c2[std::rand() % c2.size()]; //-V104

            bool result = hpx::parallel::includes(policy,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                boost::begin(c2), boost::end(c2), std::less<std::size_t>());

            bool expected = std::includes(boost::begin(c1), boost::end(c1),
                boost::begin(c2), boost::end(c2), std::less<std::size_t>());

            // verify values
            HPX_TEST_EQ(result, expected);
        }
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_includes2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);

    std::size_t start = std::rand() % c1.size();
    std::size_t end = start + (std::rand() % (c1.size() - start));

    HPX_ASSERT(start <= end);

    base_iterator start_it = boost::next(boost::begin(c1), start);
    base_iterator end_it = boost::next(boost::begin(c1), end);

    {
        hpx::future<bool> result =
            hpx::parallel::includes(p,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                start_it, end_it, std::less<std::size_t>());
        result.wait();

        bool expected = std::includes(
            boost::begin(c1), boost::end(c1),
            start_it, end_it, std::less<std::size_t>());

        // verify values
        HPX_TEST_EQ(result.get(), expected);
    }

    {
        std::vector<std::size_t> c2;
        std::copy(start_it, end_it, std::back_inserter(c2));

        if (!c2.empty())
        {
            ++c2[std::rand() % c2.size()]; //-V104

            hpx::future<bool> result =
                hpx::parallel::includes(p,
                    iterator(boost::begin(c1)), iterator(boost::end(c1)),
                    boost::begin(c2), boost::end(c2), std::less<std::size_t>());
            result.wait();

            bool expected = std::includes(boost::begin(c1), boost::end(c1),
                boost::begin(c2), boost::end(c2), std::less<std::size_t>());

            // verify values
            HPX_TEST_EQ(result.get(), expected);
        }
    }
}

template <typename IteratorTag>
void test_includes2()
{
    using namespace hpx::parallel;

    test_includes2(seq, IteratorTag());
    test_includes2(par, IteratorTag());
    test_includes2(par_vec, IteratorTag());

    test_includes2_async(seq(task), IteratorTag());
    test_includes2_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_includes2(execution_policy(seq), IteratorTag());
    test_includes2(execution_policy(par), IteratorTag());
    test_includes2(execution_policy(par_vec), IteratorTag());

    test_includes2(execution_policy(seq(task)), IteratorTag());
    test_includes2(execution_policy(par(task)), IteratorTag());
#endif
}

void includes_test2()
{
    test_includes2<std::random_access_iterator_tag>();
    test_includes2<std::forward_iterator_tag>();
    test_includes2<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_includes_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);

    std::size_t start = std::rand() % c1.size();
    std::size_t end = start + (std::rand() % (c1.size() - start));

    HPX_ASSERT(start <= end);

    if (start == end)
        ++end;

    HPX_ASSERT(end <= c1.size());

    base_iterator start_it = boost::next(boost::begin(c1), start);
    base_iterator end_it = boost::next(boost::begin(c1), end);

    bool caught_exception = false;
    try {
        hpx::parallel::includes(policy,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            start_it, end_it,
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), true;
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
void test_includes_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);

    std::size_t start = std::rand() % c1.size();
    std::size_t end = start + (std::rand() % (c1.size() - start));

    HPX_ASSERT(start <= end);

    if (start == end)
        ++end;

    HPX_ASSERT(end <= c1.size());

    base_iterator start_it = boost::next(boost::begin(c1), start);
    base_iterator end_it = boost::next(boost::begin(c1), end);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<bool> f =
            hpx::parallel::includes(p,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                start_it, end_it,
                [](std::size_t v1, std::size_t v2) {
                    return throw std::runtime_error("test"), true;
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
void test_includes_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_includes_exception(seq, IteratorTag());
    test_includes_exception(par, IteratorTag());

    test_includes_exception_async(seq(task), IteratorTag());
    test_includes_exception_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_includes_exception(execution_policy(seq), IteratorTag());
    test_includes_exception(execution_policy(par), IteratorTag());

    test_includes_exception(execution_policy(seq(task)), IteratorTag());
    test_includes_exception(execution_policy(par(task)), IteratorTag());
#endif
}

void includes_exception_test()
{
    test_includes_exception<std::random_access_iterator_tag>();
    test_includes_exception<std::forward_iterator_tag>();
    test_includes_exception<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_includes_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);

    std::size_t start = std::rand() % c1.size();
    std::size_t end = start + (std::rand() % (c1.size() - start));

    HPX_ASSERT(start <= end);

    if (start == end)
        ++end;

    HPX_ASSERT(end <= c1.size());

    base_iterator start_it = boost::next(boost::begin(c1), start);
    base_iterator end_it = boost::next(boost::begin(c1), end);

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::includes(policy,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            start_it, end_it,
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), true;
            });

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

template <typename ExPolicy, typename IteratorTag>
void test_includes_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::size_t first_value = std::rand(); //-V101
    std::iota(boost::begin(c1), boost::end(c1), first_value);

    std::size_t start = std::rand() % c1.size();
    std::size_t end = start + (std::rand() % (c1.size() - start));

    HPX_ASSERT(start <= end);

    if (start == end)
        ++end;

    HPX_ASSERT(end <= c1.size());

    base_iterator start_it = boost::next(boost::begin(c1), start);
    base_iterator end_it = boost::next(boost::begin(c1), end);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<bool> f =
            hpx::parallel::includes(p,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                start_it, end_it,
                [](std::size_t v1, std::size_t v2) {
                    return throw std::bad_alloc(), true;
                });
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
void test_includes_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_includes_bad_alloc(seq, IteratorTag());
    test_includes_bad_alloc(par, IteratorTag());

    test_includes_bad_alloc_async(seq(task), IteratorTag());
    test_includes_bad_alloc_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_includes_bad_alloc(execution_policy(seq), IteratorTag());
    test_includes_bad_alloc(execution_policy(par), IteratorTag());

    test_includes_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_includes_bad_alloc(execution_policy(par(task)), IteratorTag());
#endif
}

void includes_bad_alloc_test()
{
    test_includes_bad_alloc<std::random_access_iterator_tag>();
    test_includes_bad_alloc<std::forward_iterator_tag>();
    test_includes_bad_alloc<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    includes_test1();
    includes_test2();
    includes_exception_test();
    includes_bad_alloc_test();
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


