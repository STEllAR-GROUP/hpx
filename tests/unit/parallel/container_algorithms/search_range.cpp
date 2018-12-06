//  Copyright (c) 2018 Christopher Ogle
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


struct user_defined_type_1
{
    user_defined_type_1() = default;
    user_defined_type_1(int v) : val(v){}
    unsigned int val;
};

struct user_defined_type_2
{
    user_defined_type_2() = default;
    user_defined_type_2(int v) : val(v){}
    std::size_t val;
};

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_search1(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 3);
    // create subsequence in middle of vector
    c[c.size()/2] = 1;
    c[c.size()/2 + 1] = 2;

    std::size_t h[] = { 1, 2 };

    auto index = hpx::parallel::search(policy, c, h);
    auto test_index = std::begin(c) + c.size()/2;

    HPX_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_search1_async(ExPolicy p, IteratorTag)
{
    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 3);
    // create subsequence in middle of vector
    c[c.size()/2] = 1;
    c[c.size()/2 + 1] = 2;

    std::size_t h[] = { 1, 2 };

    auto f = hpx::parallel::search(p, c, h);
    f.wait();

    // create iterator at position of value to be found
    auto test_index = std::begin(c) + c.size()/2;

    HPX_TEST(f.get() == test_index);
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

    std::vector<std::size_t> c(10007);
    // fill vector with random values about 2
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 3);
    // create subsequence at start and end
    c[0] = 1;
    c[1] = 2;
    c[c.size()-1] = 2;
    c[c.size()-2] = 1;

    std::size_t h[] = { 1, 2 };

    auto index = hpx::parallel::search(policy, c, h);

    auto test_index = std::begin(c);

    HPX_TEST(index == test_index);
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

    auto f = hpx::parallel::search(p, c, h);
    f.wait();

    // create iterator at position of value to be found
    auto test_index = std::begin(c);

    HPX_TEST(f.get() == test_index);
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

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), (std::rand() % 100) + 3);
    // create subsequence large enough to always be split into multiple partitions
    std::iota(std::begin(c), std::begin(c) + c.size()/16+1, 1);
    std::size_t sub_size = c.size()/16 + 1;
    std::vector<std::size_t> h(sub_size);
    std::iota(std::begin(h), std::end(h), 1);

    auto index = hpx::parallel::search(policy, c, h);

    auto test_index = std::begin(c);

    HPX_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_search3_async(ExPolicy p, IteratorTag)
{
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
    auto f = hpx::parallel::search(p, c, h);
    f.wait();

    //create iterator at position of value to be found
    auto test_index = std::begin(c);

    HPX_TEST(f.get() == test_index);
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

    auto index = hpx::parallel::search(policy, c, h, op);

    auto test_index = std::begin(c) + c.size()/2;

    HPX_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_search4_async(ExPolicy p, IteratorTag)
{
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

    auto f = hpx::parallel::search(p, c, h, op);
    f.wait();

    // create iterator at position of value to be found
    auto test_index = std::begin(c) + c.size()/2;

    HPX_TEST(f.get() == test_index);
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
}

void search_test4()
{
    test_search4<std::random_access_iterator_tag>();
    test_search4<std::forward_iterator_tag>();
}

template <typename ExPolicy, typename IteratorTag>
void test_search5(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    std::vector<user_defined_type_1> c(10007);

    // fill vector with random values above 2
    std::for_each(std::begin(c), std::end(c), [](user_defined_type_1 & ut1){
        ut1.val = (std::rand() % 100) + 3;
    });

    c[c.size()/2].val = 1;
    c[c.size()/2 + 1].val = 2;

    user_defined_type_2 h[] = { user_defined_type_2(1), user_defined_type_2(2) };

    auto op =
        [](std::size_t a, std::size_t b)
        {
            return (a == b);
        };

    //Provide custom projections
    auto proj1 =
        [](const user_defined_type_1 & ut1)
        {
            return ut1.val;
        };

    auto proj2 =
        [](const user_defined_type_2 & ut2)
        {
            return ut2.val;
        };

    auto index = hpx::parallel::search(policy, c, h, op, proj1, proj2);
    auto test_index = std::begin(c) + c.size()/2;

    HPX_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_search5_async(ExPolicy p, IteratorTag)
{
    std::vector<user_defined_type_1> c(10007);
    // fill vector with random values above 2
    std::for_each(std::begin(c), std::end(c), [](user_defined_type_1 & ut1){
        ut1.val = (std::rand() % 100) + 3;
    });
    // create subsequence in middle of vector,
    c[c.size()/2].val = 1;
    c[c.size()/2 + 1].val = 2;

    user_defined_type_2 h[] = { user_defined_type_2(1), user_defined_type_2(2) };

    auto op =
        [](std::size_t a, std::size_t b)
        {
            return !(a != b);
        };

    //Provide custom projections
    auto proj1 =
        [](const user_defined_type_1 & ut1)
        {
            return ut1.val;
        };

    auto proj2 =
        [](const user_defined_type_2 & ut2)
        {
            return ut2.val;
        };


    auto f = hpx::parallel::search(p, c, h, op, proj1, proj2);
    f.wait();

    // create iterator at position of value to be found
    auto test_index = std::begin(c) + c.size()/2;

    HPX_TEST(f.get() == test_index);
}

template <typename IteratorTag>
void test_search5()
{
    using namespace hpx::parallel;
    test_search5(execution::seq, IteratorTag());
    test_search5(execution::par, IteratorTag());
    test_search5(execution::par_unseq, IteratorTag());

    test_search5_async(execution::seq(execution::task), IteratorTag());
    test_search5_async(execution::par(execution::task), IteratorTag());
}

void search_test5()
{
    test_search5<std::random_access_iterator_tag>();
    test_search5<std::forward_iterator_tag>();
}


////////////////////////////////////////////////////////////////////////////
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
    search_test5();
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
