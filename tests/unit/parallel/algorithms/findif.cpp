//  copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_find.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(2,101);

template <typename ExPolicy, typename IteratorTag>
void test_find_if(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill vector with random values about 1
    std::fill(std::begin(c), std::end(c), dis(gen));
    c.at(c.size()/2) = 1;

    iterator index = hpx::parallel::find_if(policy,
        iterator(std::begin(c)), iterator(std::end(c)),
        [](std::size_t v) {
            return v == std::size_t(1);
        });

    base_iterator test_index = std::begin(c) + c.size()/2;

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_find_if_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill vector with random values above 1
    std::fill(std::begin(c), std::end(c), dis(gen));
    c.at(c.size()/2) = 1;

    hpx::future<iterator> f =
        hpx::parallel::find_if(p,
            iterator(std::begin(c)), iterator(std::end(c)),
        [](std::size_t v) {
            return v == std::size_t(1);
        });
    f.wait();

    //create iterator at position of value to be found
    base_iterator test_index = std::begin(c) + c.size()/2;

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_find_if()
{
    using namespace hpx::parallel;
    test_find_if(execution::seq, IteratorTag());
    test_find_if(execution::par, IteratorTag());
    test_find_if(execution::par_unseq, IteratorTag());

    test_find_if_async(execution::seq(execution::task), IteratorTag());
    test_find_if_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_if(execution_policy(execution::seq), IteratorTag());
    test_find_if(execution_policy(execution::par), IteratorTag());
    test_find_if(execution_policy(execution::par_unseq), IteratorTag());

    test_find_if(execution_policy(execution::seq(execution::task)), IteratorTag());
    test_find_if(execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void find_if_test()
{
    test_find_if<std::random_access_iterator_tag>();
    test_find_if<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_find_if<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_if_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen()+1);
    c[c.size()/2]=0;

    bool caught_exception = false;
    try {
        hpx::parallel::find_if(policy,
            decorated_iterator(
                std::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(std::end(c)),
            [](std::size_t v){return 1;});
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
void test_find_if_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen()+1);
    c[c.size()/2]=0;

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::find_if(p,
                decorated_iterator(
                    std::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(std::end(c)),
                [](std::size_t v){return 1;});
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
void test_find_if_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_find_if_exception(execution::seq, IteratorTag());
    test_find_if_exception(execution::par, IteratorTag());

    test_find_if_exception_async(execution::seq(execution::task), IteratorTag());
    test_find_if_exception_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_if_exception(execution_policy(execution::seq), IteratorTag());
    test_find_if_exception(execution_policy(execution::par), IteratorTag());

    test_find_if_exception(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_find_if_exception(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void find_if_exception_test()
{
    test_find_if_exception<std::random_access_iterator_tag>();
    test_find_if_exception<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_find_if_exception<std::input_iterator_tag>();
#endif
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_if_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(100007);
    std::iota(std::begin(c), std::end(c), gen()+1);
    c[c.size()/2]=0;

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::find_if(policy,
            decorated_iterator(
                std::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(std::end(c)),
            [](std::size_t v){return 1;});
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
void test_find_if_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen()+1);
    c[c.size()/2] = 0;

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::find_if(p,
                decorated_iterator(
                    std::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(std::end(c)),
                [](std::size_t v){return 1;});
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
void test_find_if_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_find_if_bad_alloc(execution::seq, IteratorTag());
    test_find_if_bad_alloc(execution::par, IteratorTag());

    test_find_if_bad_alloc_async(execution::seq(execution::task), IteratorTag());
    test_find_if_bad_alloc_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_if_bad_alloc(execution_policy(execution::seq), IteratorTag());
    test_find_if_bad_alloc(execution_policy(execution::par), IteratorTag());

    test_find_if_bad_alloc(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_find_if_bad_alloc(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void find_if_bad_alloc_test()
{
    test_find_if_bad_alloc<std::random_access_iterator_tag>();
    test_find_if_bad_alloc<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_find_if_bad_alloc<std::input_iterator_tag>();
#endif
}

int hpx_main(boost::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    find_if_test();
    find_if_exception_test();
    find_if_bad_alloc_test();
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
