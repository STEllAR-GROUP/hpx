//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_search.hpp>
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
void test_search1(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 3);
    // create subsequence in middle of vector
    c[c.size()/2] = 1;
    c[c.size()/2 + 1] = 2;

    std::size_t h[] = { 1, 2 };

    iterator index = hpx::parallel::search(policy,
        iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(h), std::end(h));

    base_iterator test_index = std::begin(c) + c.size()/2;

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_search1_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 3);
    // create subsequence in middle of vector
    c[c.size()/2] = 1;
    c[c.size()/2 + 1] = 2;

    std::size_t h[] = { 1, 2 };

    hpx::future<iterator> f =
        hpx::parallel::search(p,
            iterator(std::begin(c)), iterator(std::end(c)),
            std::begin(h), std::end(h));
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = std::begin(c) + c.size()/2;

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_search1()
{
    using namespace hpx::parallel;
    test_search1(execution::seq, IteratorTag());
    test_search1(execution::par, IteratorTag());
    test_search1(execution::par_unseq, IteratorTag());

    test_search1_async(execution::seq(execution::task), IteratorTag());
    test_search1_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_search1(execution_policy(execution::seq), IteratorTag());
    test_search1(execution_policy(execution::par), IteratorTag());
    test_search1(execution_policy(execution::par_unseq), IteratorTag());
    test_search1(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_search1(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void search_test1()
{
    test_search1<std::random_access_iterator_tag>();
    test_search1<std::forward_iterator_tag>();
}

template <typename ExPolicy, typename IteratorTag>
void test_search2(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values about 2
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 3);
    // create subsequence at start and end
    c[0] = 1;
    c[1] = 2;
    c[c.size()-1] = 2;
    c[c.size()-2] = 1;

    std::size_t h[] = { 1, 2 };

    iterator index = hpx::parallel::search(policy,
        iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(h), std::end(h));

    base_iterator test_index = std::begin(c);

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_search2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 3);
    // create subsequence at start and end
    c[0] = 1;
    c[1] = 2;
    c[c.size()-1] = 2;
    c[c.size()-2] = 1;

    std::size_t h[] = { 1, 2 };

    hpx::future<iterator> f =
        hpx::parallel::search(p,
            iterator(std::begin(c)), iterator(std::end(c)),
            std::begin(h), std::end(h));
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = std::begin(c);

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_search2()
{
    using namespace hpx::parallel;
    test_search2(execution::seq, IteratorTag());
    test_search2(execution::par, IteratorTag());
    test_search2(execution::par_unseq, IteratorTag());

    test_search2_async(execution::seq(execution::task), IteratorTag());
    test_search2_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_search2(execution_policy(execution::seq), IteratorTag());
    test_search2(execution_policy(execution::par), IteratorTag());
    test_search2(execution_policy(execution::par_unseq), IteratorTag());

    test_search2(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_search2(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void search_test2()
{
    test_search2<std::random_access_iterator_tag>();
    test_search2<std::forward_iterator_tag>();
}

template <typename ExPolicy, typename IteratorTag>
void test_search3(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 3);
    // create subsequence large enough to always be split into multiple partitions
    std::iota(std::begin(c), std::begin(c) + c.size()/16+1, 1);
    std::size_t sub_size = c.size()/16 + 1;
    std::vector<std::size_t> h(sub_size);
    std::iota(std::begin(h), std::end(h), 1);

    iterator index = hpx::parallel::search(policy,
        iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(h), std::end(h));

    base_iterator test_index = std::begin(c);

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_search3_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 6
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 7);
    // create subsequence large enough to always be split into multiple partitions
    std::iota(std::begin(c), std::begin(c) + c.size()/16+1, 1);
    std::size_t sub_size = c.size()/16 + 1;
    std::vector<std::size_t> h(sub_size);
    std::iota(std::begin(h), std::end(h), 1);

    // create only two partitions, splitting the desired sub sequence into
    // separate partitions.
    hpx::future<iterator> f =
        hpx::parallel::search(p,
            iterator(std::begin(c)), iterator(std::end(c)),
            std::begin(h), std::end(h));
    f.wait();

    //create iterator at position of value to be found
    base_iterator test_index = std::begin(c);

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_search3()
{
    using namespace hpx::parallel;
    test_search3(execution::seq, IteratorTag());
    test_search3(execution::par, IteratorTag());
    test_search3(execution::par_unseq, IteratorTag());

    test_search3_async(execution::seq(execution::task), IteratorTag());
    test_search3_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_search3(execution_policy(execution::seq), IteratorTag());
    test_search3(execution_policy(execution::par), IteratorTag());
    test_search3(execution_policy(execution::par_unseq), IteratorTag());

    test_search3(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_search3(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void search_test3()
{
    test_search3<std::random_access_iterator_tag>();
    test_search3<std::forward_iterator_tag>();
}

template <typename ExPolicy, typename IteratorTag>
void test_search4(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 3);
    // create subsequence in middle of vector
    c[c.size()/2] = 1;
    c[c.size()/2 + 1] = 2;

    std::size_t h[] = { 1, 2 };

    auto op =
        [](std::size_t a, std::size_t b)
        {
            return !(a != b);
        };

    iterator index = hpx::parallel::search(policy,
        iterator(std::begin(c)), iterator(std::end(c)),
        std::begin(h), std::end(h), op);

    base_iterator test_index = std::begin(c) + c.size()/2;

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_search4_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 3);
    // create subsequence in middle of vector, provide custom predicate
    // for search
    c[c.size()/2] = 1;
    c[c.size()/2 + 1] = 2;

    std::size_t h[] = { 1, 2 };

    auto op =
        [](std::size_t a, std::size_t b)
        {
            return !(a != b);
        };

    hpx::future<iterator> f =
        hpx::parallel::search(p,
            iterator(std::begin(c)), iterator(std::end(c)),
            std::begin(h), std::end(h), op);
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = std::begin(c) + c.size()/2;

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_search4()
{
    using namespace hpx::parallel;
    test_search4(execution::seq, IteratorTag());
    test_search4(execution::par, IteratorTag());
    test_search4(execution::par_unseq, IteratorTag());

    test_search4_async(execution::seq(execution::task), IteratorTag());
    test_search4_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_search4(execution_policy(execution::seq), IteratorTag());
    test_search4(execution_policy(execution::par), IteratorTag());
    test_search4(execution_policy(execution::par_unseq), IteratorTag());
    test_search4(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_search4(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void search_test4()
{
    test_search4<std::random_access_iterator_tag>();
    test_search4<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_search_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand() + 1);
    c[c.size()/2] = 1;
    c[c.size()/2+1] = 2;

    std::vector<std::size_t> h;
    h.push_back(1);
    h.push_back(2);

    bool caught_exception = false;
    try {
        hpx::parallel::search(policy,
            decorated_iterator(
                std::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c),
                [](){ throw std::runtime_error("test"); }),
            std::begin(h), std::end(h));
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
void test_search_async_exception(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand() + 1);
    c[c.size()/2] = 1;
    c[c.size()/2+1] = 2;

    std::size_t h[] = { 1, 2 };

    bool caught_exception = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::search(p,
                decorated_iterator(
                    std::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(
                    std::end(c),
                    [](){ throw std::runtime_error("test"); }),
            std::begin(h), std::end(h));
        f.get();

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<
            ExPolicy, IteratorTag
        >::call(p, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename IteratorTag>
void test_search_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_search_exception(execution::seq, IteratorTag());
    test_search_exception(execution::par, IteratorTag());

    test_search_async_exception(execution::seq(execution::task), IteratorTag());
    test_search_async_exception(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_search_exception(execution_policy(execution::par), IteratorTag());
    test_search_exception(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_search_exception(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void search_exception_test()
{
    test_search_exception<std::random_access_iterator_tag>();
    test_search_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_search_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(100007);
    std::iota(std::begin(c), std::end(c), std::rand() + 1);
    c[c.size()/2] = 0;

    std::size_t h[] = { 1, 2 };

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::search(policy,
            decorated_iterator(
                std::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(
                std::end(c),
                [](){ throw std::bad_alloc(); }),
            std::begin(h), std::end(h));
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
void test_search_async_bad_alloc(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand() + 1);
    c[c.size()/2] = 0;

    std::size_t h[] = { 1, 2 };

    bool caught_bad_alloc = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::search(p,
                decorated_iterator(
                    std::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(
                    std::end(c),
                    [](){ throw std::bad_alloc(); }),
                std::begin(h), std::end(h));

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
}

template <typename IteratorTag>
void test_search_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_search_bad_alloc(execution::par, IteratorTag());
    test_search_bad_alloc(execution::seq, IteratorTag());

    test_search_async_bad_alloc(execution::seq(execution::task), IteratorTag());
    test_search_async_bad_alloc(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_search_bad_alloc(execution_policy(execution::par), IteratorTag());
    test_search_bad_alloc(execution_policy(execution::seq), IteratorTag());
    test_search_bad_alloc(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_search_bad_alloc(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void search_bad_alloc_test()
{
    test_search_bad_alloc<std::random_access_iterator_tag>();
    test_search_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{

    unsigned int seed = (unsigned int)std::time(nullptr);
    if(vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    search_test1();
    search_test2();
    search_test3();
    search_test4();
    search_exception_test();
    search_bad_alloc_test();
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

    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
