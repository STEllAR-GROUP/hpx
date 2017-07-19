//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_remove_copy.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    std::size_t middle_idx = std::rand() % (c.size()/2);
    auto middle = std::begin(c) + middle_idx;
    std::iota(std::begin(c), middle, static_cast<int>(std::rand() % c.size()));
    std::fill(middle, std::end(c), -1);

    hpx::parallel::remove_copy_if(policy,
        iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(d), [](int i) { return i < 0; });

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(c), middle, std::begin(d),
        [&count](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    HPX_TEST(std::equal(middle, std::end(c),
        std::begin(d) + middle_idx,
        [&count](int v1, int v2) -> bool {
            HPX_TEST_NEQ(v1, v2);
            ++count;
            return v1 != v2;
        }));

    HPX_TEST_EQ(count, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    std::size_t middle_idx = std::rand() % (c.size()/2);
    auto middle = std::begin(c) + middle_idx;
    std::iota(std::begin(c), middle, static_cast<int>(std::rand() % c.size()));
    std::fill(middle, std::end(c), -1);

    auto f =
        hpx::parallel::remove_copy_if(p,
            iterator(std::begin(c)), iterator(std::end(c)),
            std::begin(d), [](int i){ return i < 0; });
    f.wait();

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(c), middle, std::begin(d),
        [&count](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    HPX_TEST(std::equal(middle, std::end(c),
        std::begin(d) + middle_idx,
        [&count](int v1, int v2) -> bool {
            HPX_TEST_NEQ(v1, v2);
            ++count;
            return v1!=v2;
        }));

    HPX_TEST_EQ(count, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_outiter(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(0);
    std::size_t middle_idx = std::rand() % (c.size()/2);
    auto middle = std::begin(c) + middle_idx;
    std::iota(std::begin(c), middle, static_cast<int>(std::rand() % c.size()));
    std::fill(middle, std::end(c), -1);

    hpx::parallel::remove_copy_if(policy,
        iterator(std::begin(c)), iterator(std::end(c)),
        std::back_inserter(d), [](int i){ return i < 0; });

    HPX_TEST(std::equal(std::begin(c), middle, std::begin(d),
        [](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            return v1 == v2;
        }));

    HPX_TEST_EQ(middle_idx, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_outiter_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(0);
    std::size_t middle_idx = std::rand() % (c.size()/2);
    auto middle = std::begin(c) + middle_idx;
    std::iota(std::begin(c), middle, static_cast<int>(std::rand() % c.size()));
    std::fill(middle, std::end(c), -1);

    auto f =
        hpx::parallel::remove_copy_if(p,
            iterator(std::begin(c)), iterator(std::end(c)),
            std::back_inserter(d), [](int i){ return i < 0; });
    f.wait();

    HPX_TEST(std::equal(std::begin(c), middle, std::begin(d),
        [](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            return v1 == v2;
        }));

    HPX_TEST_EQ(middle_idx, d.size());
}

template <typename IteratorTag>
void test_remove_copy_if()
{
    using namespace hpx::parallel;

    test_remove_copy_if(execution::seq, IteratorTag());
    test_remove_copy_if(execution::par, IteratorTag());
    test_remove_copy_if(execution::par_unseq, IteratorTag());

    test_remove_copy_if_async(execution::seq(execution::task), IteratorTag());
    test_remove_copy_if_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_remove_copy_if(execution_policy(execution::seq), IteratorTag());
    test_remove_copy_if(execution_policy(execution::par), IteratorTag());
    test_remove_copy_if(execution_policy(execution::par_unseq), IteratorTag());

    test_remove_copy_if(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_remove_copy_if(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif

#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_remove_copy_if_outiter(execution::seq, IteratorTag());
    test_remove_copy_if_outiter(execution::par, IteratorTag());
    test_remove_copy_if_outiter(execution::par_unseq, IteratorTag());

    test_remove_copy_if_outiter_async(execution::seq(execution::task),
        IteratorTag());
    test_remove_copy_if_outiter_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_remove_copy_if_outiter(execution_policy(execution::seq), IteratorTag());
    test_remove_copy_if_outiter(execution_policy(execution::par), IteratorTag());
    test_remove_copy_if_outiter(execution_policy(execution::par_unseq), IteratorTag());

    test_remove_copy_if_outiter(
        execution_policy(execution::seq(execution::task)), IteratorTag());
    test_remove_copy_if_outiter(
        execution_policy(execution::par(execution::task)), IteratorTag());
#endif
#endif
}

void remove_copy_if_test()
{
    test_remove_copy_if<std::random_access_iterator_tag>();
    test_remove_copy_if<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_remove_copy_if<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::remove_copy_if(policy,
            iterator(std::begin(c)), iterator(std::end(c)), std::begin(d),
            [](std::size_t v) {
                return throw std::runtime_error("test"), v == 0;
            });
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
void test_remove_copy_if_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::remove_copy_if(p,
                iterator(std::begin(c)), iterator(std::end(c)),
                std::begin(d),
                [](std::size_t v) {
                    return throw std::runtime_error("test"), v == 0;
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
void test_remove_copy_if_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_remove_copy_if_exception(execution::seq, IteratorTag());
    test_remove_copy_if_exception(execution::par, IteratorTag());

    test_remove_copy_if_exception_async(execution::seq(execution::task),
        IteratorTag());
    test_remove_copy_if_exception_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_remove_copy_if_exception(execution_policy(execution::seq), IteratorTag());
    test_remove_copy_if_exception(execution_policy(execution::par), IteratorTag());

    test_remove_copy_if_exception(
        execution_policy(execution::seq(execution::task)), IteratorTag());
    test_remove_copy_if_exception(
        execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void remove_copy_if_exception_test()
{
    test_remove_copy_if_exception<std::random_access_iterator_tag>();
    test_remove_copy_if_exception<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_remove_copy_if_exception<std::input_iterator_tag>();
#endif
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_remove_copy_if_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::remove_copy_if(policy,
            iterator(std::begin(c)), iterator(std::end(c)), std::begin(d),
            [](std::size_t v) {
                return throw std::bad_alloc(), v;
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
void test_remove_copy_if_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::remove_copy_if(p,
            iterator(std::begin(c)), iterator(std::end(c)),
            std::begin(d),
            [](std::size_t v) {
                return throw std::bad_alloc(), v;
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
void test_remove_copy_if_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_remove_copy_if_bad_alloc(execution::seq, IteratorTag());
    test_remove_copy_if_bad_alloc(execution::par, IteratorTag());

    test_remove_copy_if_bad_alloc_async(execution::seq(execution::task),
        IteratorTag());
    test_remove_copy_if_bad_alloc_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_remove_copy_if_bad_alloc(execution_policy(execution::seq), IteratorTag());
    test_remove_copy_if_bad_alloc(execution_policy(execution::par), IteratorTag());

    test_remove_copy_if_bad_alloc(
        execution_policy(execution::seq(execution::task)), IteratorTag());
    test_remove_copy_if_bad_alloc(
        execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void remove_copy_if_bad_alloc_test()
{
    test_remove_copy_if_bad_alloc<std::random_access_iterator_tag>();
    test_remove_copy_if_bad_alloc<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_remove_copy_if_bad_alloc<std::input_iterator_tag>();
#endif
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    remove_copy_if_test();
    remove_copy_if_exception_test();
    remove_copy_if_bad_alloc_test();
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
