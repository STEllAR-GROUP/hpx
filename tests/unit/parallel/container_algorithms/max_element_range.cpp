//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_minmax.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/iterator_range.hpp>

#include <cstddef>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_max_element(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_container<std::vector<std::size_t>, IteratorTag>
        test_vector;

    test_vector c = test::random_iota<test_vector>(10007);

    base_iterator ref_end = std::end(c.base());

    auto r = hpx::parallel::max_element(policy, c, std::less<std::size_t>());
    HPX_TEST(r != std::end(c));

    base_iterator ref = std::max_element(
        std::begin(c.base()), std::end(c.base()),
        std::less<std::size_t>());

    HPX_TEST(ref != ref_end);
    HPX_TEST_EQ(*ref, *r);

    r = hpx::parallel::max_element(policy, c);
    HPX_TEST(r != std::end(c));

    ref = std::max_element(std::begin(c.base()), std::end(c.base()));
    HPX_TEST(ref != ref_end);
    HPX_TEST_EQ(*ref, *r);
}

template <typename ExPolicy, typename IteratorTag>
void test_max_element_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef test::test_container<std::vector<std::size_t>, IteratorTag>
        test_vector;

    test_vector c = test::random_iota<test_vector>(10007);

    base_iterator ref_end(std::end(c.base()));

    auto r = hpx::parallel::max_element(p, c, std::less<std::size_t>());
    iterator rit = r.get();
    HPX_TEST(rit != std::end(c));

    base_iterator ref = std::max_element(
        std::begin(c.base()), std::end(c.base()),
        std::less<std::size_t>());
    HPX_TEST(ref != ref_end);
    HPX_TEST_EQ(*ref, *rit);

    r = hpx::parallel::max_element(p, c);
    rit = r.get();
    HPX_TEST(rit != std::end(c));

    ref = std::max_element(std::begin(c.base()), std::end(c.base()));
    HPX_TEST(ref != ref_end);
    HPX_TEST_EQ(*ref, *rit);
}

template <typename IteratorTag>
void test_max_element()
{
    using namespace hpx::parallel;

    test_max_element(execution::seq, IteratorTag());
    test_max_element(execution::par, IteratorTag());
    test_max_element(execution::par_unseq, IteratorTag());

    test_max_element_async(execution::seq(execution::task), IteratorTag());
    test_max_element_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_max_element(execution_policy(execution::seq), IteratorTag());
    test_max_element(execution_policy(execution::par), IteratorTag());
    test_max_element(execution_policy(execution::par_unseq), IteratorTag());

    test_max_element(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_max_element(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void max_element_test()
{
    test_max_element<std::random_access_iterator_tag>();
    test_max_element<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_max_element_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c = test::random_iota(10007);

    {
        bool caught_exception = false;
        try {
            hpx::parallel::max_element(policy,
                hpx::util::make_iterator_range(
                    decorated_iterator(
                        std::begin(c),
                        [](){ throw std::runtime_error("test"); }),
                    decorated_iterator(std::end(c))),
                std::less<std::size_t>());

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

    {
        bool caught_exception = false;
        try {
            hpx::parallel::max_element(policy,
                decorated_iterator(
                    std::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(std::end(c)));

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
}

template <typename ExPolicy, typename IteratorTag>
void test_max_element_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c = test::random_iota(10007);

    {
        bool returned_from_algorithm = false;
        bool caught_exception = false;

        try {
            hpx::future<decorated_iterator> f =
                hpx::parallel::max_element(p,
                    decorated_iterator(
                        std::begin(c),
                        [](){ throw std::runtime_error("test"); }),
                    decorated_iterator(std::end(c)),
                    std::less<std::size_t>());

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

    {
        bool caught_exception = false;
        bool returned_from_algorithm = false;

        try {
            hpx::future<decorated_iterator> f =
                hpx::parallel::max_element(p,
                    hpx::util::make_iterator_range(
                        decorated_iterator(
                            std::begin(c),
                            [](){ throw std::runtime_error("test"); }),
                        decorated_iterator(std::end(c))));

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
}

template <typename IteratorTag>
void test_max_element_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_max_element_exception(execution::seq, IteratorTag());
    test_max_element_exception(execution::par, IteratorTag());

    test_max_element_exception_async(execution::seq(execution::task),
        IteratorTag());
    test_max_element_exception_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_max_element_exception(execution_policy(execution::seq), IteratorTag());
    test_max_element_exception(execution_policy(execution::par), IteratorTag());

    test_max_element_exception(
        execution_policy(execution::seq(execution::task)), IteratorTag());
    test_max_element_exception(
        execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void max_element_exception_test()
{
    test_max_element_exception<std::random_access_iterator_tag>();
    test_max_element_exception<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_max_element_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c = test::random_iota(10007);

    {
        bool caught_exception = false;
        try {
            hpx::parallel::max_element(policy,
                hpx::util::make_iterator_range(
                    decorated_iterator(
                        std::begin(c),
                        [](){ throw std::bad_alloc(); }),
                    decorated_iterator(std::end(c))),
                std::less<std::size_t>());

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

    {
        bool caught_exception = false;
        try {
            hpx::parallel::max_element(policy,
                hpx::util::make_iterator_range(
                    decorated_iterator(
                        std::begin(c),
                        [](){ throw std::bad_alloc(); }),
                    decorated_iterator(std::end(c))));

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
}

template <typename ExPolicy, typename IteratorTag>
void test_max_element_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c = test::random_iota(10007);

    {
        bool returned_from_algorithm = false;
        bool caught_exception = false;

        try {
            hpx::future<decorated_iterator> f =
                hpx::parallel::max_element(p,
                    hpx::util::make_iterator_range(
                        decorated_iterator(
                            std::begin(c),
                            [](){ throw std::bad_alloc(); }),
                        decorated_iterator(std::end(c))),
                    std::less<std::size_t>());

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

    {
        bool caught_exception = false;
        bool returned_from_algorithm = false;

        try {
            hpx::future<decorated_iterator> f =
                hpx::parallel::max_element(p,
                    hpx::util::make_iterator_range(
                        decorated_iterator(
                            std::begin(c),
                            [](){ throw std::bad_alloc(); }),
                        decorated_iterator(std::end(c))));

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
}

template <typename IteratorTag>
void test_max_element_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_max_element_bad_alloc(execution::seq, IteratorTag());
    test_max_element_bad_alloc(execution::par, IteratorTag());

    test_max_element_bad_alloc_async(execution::seq(execution::task),
        IteratorTag());
    test_max_element_bad_alloc_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_max_element_bad_alloc(execution_policy(execution::seq), IteratorTag());
    test_max_element_bad_alloc(execution_policy(execution::par), IteratorTag());

    test_max_element_bad_alloc(
        execution_policy(execution::seq(execution::task)), IteratorTag());
    test_max_element_bad_alloc(
        execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void max_element_bad_alloc_test()
{
    test_max_element_bad_alloc<std::random_access_iterator_tag>();
    test_max_element_bad_alloc<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    max_element_test();
    max_element_exception_test();
    max_element_bad_alloc_test();

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


