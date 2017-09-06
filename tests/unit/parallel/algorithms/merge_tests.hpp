//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_MERGE_AUG_11_2017_0540PM)
#define HPX_PARALLEL_TEST_MERGE_AUG_11_2017_0540PM

#include <hpx/include/parallel_merge.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/unused.hpp>

#include <boost/random.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////

struct throw_always
{
    template <typename T>
    bool operator()(T const&, T const&) const
    {
        throw std::runtime_error("test");
    }
};

struct throw_bad_alloc
{
    template <typename T>
    bool operator()(T const&, T const&) const
    {
        throw std::bad_alloc();
    }
};

struct user_defined_type
{
    user_defined_type() = default;
    user_defined_type(int rand_no)
      : val(rand_no),
        name(name_list[std::rand() % name_list.size()])
    {}

    bool operator<(user_defined_type const& t) const
    {
        if (this->name < t.name)
            return true;
        else if (this->name > t.name)
            return false;
        else
            return this->val < t.val;
    }

    bool operator>(user_defined_type const& t) const
    {
        if (this->name > t.name)
            return true;
        else if (this->name < t.name)
            return false;
        else
            return this->val > t.val;
    }

    bool operator==(user_defined_type const& t) const
    {
        return this->name == t.name && this->val == t.val;
    }

    user_defined_type operator+(int val) const
    {
        user_defined_type t(*this);
        t.val += val;
        return t;
    }

    static const std::vector<std::string> name_list;

    int val;
    std::string name;
};

const std::vector<std::string> user_defined_type::name_list{
    "ABB", "ABC", "ACB", "BASE", "CAA", "CAAA", "CAAB"
};

struct random_fill
{
    random_fill() = default;
    random_fill(int rand_base, int range)
      : gen(std::rand()),
        dist(rand_base - range / 2, rand_base + range / 2)
    {}

    int operator()()
    {
        return dist(gen);
    }

    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<> dist;
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag, typename DataType, typename Comp>
void test_merge(ExPolicy policy, IteratorTag, DataType, Comp comp,
    int rand_base)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    using hpx::util::get;

    std::size_t const size1 = 300007, size2 = 123456;
    std::vector<DataType> src1(size1), src2(size2),
        dest_res(size1 + size2), dest_sol(size1 + size2);

    std::generate(std::begin(src1), std::end(src1), random_fill(rand_base, 6));
    std::generate(std::begin(src2), std::end(src2), random_fill(rand_base, 8));
    std::sort(std::begin(src1), std::end(src1), comp);
    std::sort(std::begin(src2), std::end(src2), comp);

    auto result = hpx::parallel::merge(policy,
        iterator(std::begin(src1)), iterator(std::end(src1)),
        iterator(std::begin(src2)), iterator(std::end(src2)),
        iterator(std::begin(dest_res)),
        comp);
    auto solution = std::merge(
        std::begin(src1), std::end(src1),
        std::begin(src2), std::end(src2),
        std::begin(dest_sol),
        comp);

    bool equality = std::equal(
        std::begin(dest_res), get<2>(result).base(),
        std::begin(dest_sol), solution);

    HPX_TEST(equality);
}

template <typename ExPolicy, typename IteratorTag, typename DataType, typename Comp>
void test_merge_async(ExPolicy policy, IteratorTag, DataType, Comp comp,
    int rand_base)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    using hpx::util::get;

    std::size_t const size1 = 300007, size2 = 123456;
    std::vector<DataType> src1(size1), src2(size2),
        dest_res(size1 + size2), dest_sol(size1 + size2);

    std::generate(std::begin(src1), std::end(src1), random_fill(rand_base, 6));
    std::generate(std::begin(src2), std::end(src2), random_fill(rand_base, 8));
    std::sort(std::begin(src1), std::end(src1), comp);
    std::sort(std::begin(src2), std::end(src2), comp);

    auto f = hpx::parallel::merge(policy,
        iterator(std::begin(src1)), iterator(std::end(src1)),
        iterator(std::begin(src2)), iterator(std::end(src2)),
        iterator(std::begin(dest_res)),
        comp);
    auto result = f.get();
    auto solution = std::merge(
        std::begin(src1), std::end(src1),
        std::begin(src2), std::end(src2),
        std::begin(dest_sol),
        comp);

    bool equality = std::equal(
        std::begin(dest_res), get<2>(result).base(),
        std::begin(dest_sol), solution);

    HPX_TEST(equality);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_merge_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size1 = 300007, size2 = 123456;
    std::vector<int> src1(size1), src2(size2), dest(size1 + size2);

    std::generate(std::begin(src1), std::end(src1), random_fill());
    std::generate(std::begin(src2), std::end(src2), random_fill());
    std::sort(std::begin(src1), std::end(src1));
    std::sort(std::begin(src2), std::end(src2));

    bool caught_exception = false;
    try {
        auto result = hpx::parallel::merge(policy,
            iterator(std::begin(src1)), iterator(std::end(src1)),
            iterator(std::begin(src2)), iterator(std::end(src2)),
            iterator(std::begin(dest)),
            throw_always());

        HPX_UNUSED(result);
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
void test_merge_exception_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size1 = 300007, size2 = 123456;
    std::vector<int> src1(size1), src2(size2), dest(size1 + size2);

    std::generate(std::begin(src1), std::end(src1), random_fill());
    std::generate(std::begin(src2), std::end(src2), random_fill());
    std::sort(std::begin(src1), std::end(src1));
    std::sort(std::begin(src2), std::end(src2));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f = hpx::parallel::merge(policy,
            iterator(std::begin(src1)), iterator(std::end(src1)),
            iterator(std::begin(src2)), iterator(std::end(src2)),
            iterator(std::begin(dest)),
            throw_always());
        returned_from_algorithm = true;
        f.get();

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
    HPX_TEST(returned_from_algorithm);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_merge_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size1 = 300007, size2 = 123456;
    std::vector<int> src1(size1), src2(size2), dest(size1 + size2);

    std::generate(std::begin(src1), std::end(src1), random_fill());
    std::generate(std::begin(src2), std::end(src2), random_fill());
    std::sort(std::begin(src1), std::end(src1));
    std::sort(std::begin(src2), std::end(src2));

    bool caught_bad_alloc = false;
    try {
        auto result = hpx::parallel::merge(policy,
            iterator(std::begin(src1)), iterator(std::end(src1)),
            iterator(std::begin(src2)), iterator(std::end(src2)),
            iterator(std::begin(dest)),
            throw_bad_alloc());

        HPX_UNUSED(result);
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
void test_merge_bad_alloc_async(ExPolicy policy, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t const size1 = 300007, size2 = 123456;
    std::vector<int> src1(size1), src2(size2), dest(size1 + size2);

    std::generate(std::begin(src1), std::end(src1), random_fill());
    std::generate(std::begin(src2), std::end(src2), random_fill());
    std::sort(std::begin(src1), std::end(src1));
    std::sort(std::begin(src2), std::end(src2));

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f = hpx::parallel::merge(policy,
            iterator(std::begin(src1)), iterator(std::end(src1)),
            iterator(std::begin(src2)), iterator(std::end(src2)),
            iterator(std::begin(dest)),
            throw_bad_alloc());
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

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag, typename DataType>
void test_merge_etc(ExPolicy policy, IteratorTag,
    DataType, int rand_base)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef typename std::vector<DataType>::iterator base_iterator;

    using hpx::util::get;

    std::size_t const size1 = 300007, size2 = 123456;
    std::vector<DataType> src1(size1), src2(size2),
        dest_res(size1 + size2), dest_sol(size1 + size2);

    std::generate(std::begin(src1), std::end(src1), random_fill(rand_base, 6));
    std::generate(std::begin(src2), std::end(src2), random_fill(rand_base, 8));
    std::sort(std::begin(src1), std::end(src1));
    std::sort(std::begin(src2), std::end(src2));

    // Test default comparison.
    {
        typedef test::test_iterator<base_iterator, IteratorTag> iterator;

        auto result = hpx::parallel::merge(policy,
            iterator(std::begin(src1)), iterator(std::end(src1)),
            iterator(std::begin(src2)), iterator(std::end(src2)),
            iterator(std::begin(dest_res)));
        auto solution = std::merge(
            std::begin(src1), std::end(src1),
            std::begin(src2), std::end(src2),
            std::begin(dest_sol));

        bool equality = std::equal(
            std::begin(dest_res), get<2>(result).base(),
            std::begin(dest_sol), solution);

        HPX_TEST(equality);
    }

    // Test projection.
    {
        typedef test::test_iterator<base_iterator, IteratorTag> iterator;

        DataType val;
        auto result = hpx::parallel::merge(policy,
            iterator(std::begin(src1)), iterator(std::end(src1)),
            iterator(std::begin(src2)), iterator(std::end(src2)),
            iterator(std::begin(dest_res)),
            [](DataType const& a, DataType const& b) -> bool {
                return a < b;
            },
            [&val](DataType const&) -> DataType {
                // This is projection.
                return val;
            },
            [&val](DataType const&) -> DataType {
                // This is projection.
                return val + 1;
            });

        bool equality1 = std::equal(std::begin(src1), std::end(src1),
            std::begin(dest_res));
        bool equality2 = std::equal(std::begin(src2), std::end(src2),
            std::begin(dest_res) + size1);

        HPX_TEST(equality1);
        HPX_TEST(equality2);
    }

    // Test sequential_merge with input_iterator_tag.
    {
        typedef test::test_iterator<base_iterator,
            std::input_iterator_tag> input_iterator;
        typedef test::test_iterator<base_iterator,
            std::output_iterator_tag> output_iterator;

        auto result = hpx::parallel::v1::detail::sequential_merge(
            input_iterator(std::begin(src1)), input_iterator(std::end(src1)),
            input_iterator(std::begin(src2)), input_iterator(std::end(src2)),
            output_iterator(std::begin(dest_res)),
            [](DataType const& a, DataType const& b) -> bool {
                return a < b;
            },
            [](DataType const& t) -> DataType const& {
                return t;
            },
            [](DataType const& t) -> DataType const& {
                return t;
            });
        auto solution = std::merge(
            std::begin(src1), std::end(src1),
            std::begin(src2), std::end(src2),
            std::begin(dest_sol));

        bool equality = std::equal(
            std::begin(dest_res), get<2>(result).base(),
            std::begin(dest_sol), solution);

        HPX_TEST(equality);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_merge()
{
    using namespace hpx::parallel;

    int rand_base = std::rand();

    ////////// Test cases for 'int' type.
    test_merge(execution::seq, IteratorTag(), int(),
        [](const int a, const int b) -> bool {
            return a < b;
        }, rand_base);
    test_merge(execution::par, IteratorTag(), int(),
        [](const int a, const int b) -> bool {
            return a < b;
        }, rand_base);
    test_merge(execution::par_unseq, IteratorTag(), int(),
        [](const int a, const int b) -> bool {
            return a > b;
        }, rand_base);

    ////////// Test cases for user defined type.
    test_merge(execution::seq, IteratorTag(), user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a < b;
        }, rand_base);
    test_merge(execution::par, IteratorTag(), user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a > b;
        }, rand_base);
    test_merge(execution::par_unseq, IteratorTag(), user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a < b;
        }, rand_base);

    ////////// Asynchronous test cases for 'int' type.
    test_merge_async(execution::seq(execution::task), IteratorTag(), int(),
        [](const int a, const int b) -> bool {
            return a > b;
        }, rand_base);
    test_merge_async(execution::par(execution::task), IteratorTag(), int(),
        [](const int a, const int b) -> bool {
            return a > b;
        }, rand_base);

    ////////// Asynchronous test cases for user defined type.
    test_merge_async(execution::seq(execution::task), IteratorTag(),
        user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a < b;
        }, rand_base);
    test_merge_async(execution::par(execution::task), IteratorTag(),
        user_defined_type(),
        [](user_defined_type const& a, user_defined_type const& b) -> bool {
            return a < b;
        }, rand_base);

    ////////// Another test cases for justifying the implementation.
    test_merge_etc(execution::seq, IteratorTag(),
        user_defined_type(), rand_base);
    test_merge_etc(execution::par, IteratorTag(),
        user_defined_type(), rand_base);
    test_merge_etc(execution::par_unseq, IteratorTag(),
        user_defined_type(), rand_base);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_merge_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_merge_exception(execution::seq, IteratorTag());
    test_merge_exception(execution::par, IteratorTag());

    test_merge_exception_async(execution::seq(execution::task),
        IteratorTag());
    test_merge_exception_async(execution::par(execution::task),
        IteratorTag());
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_merge_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_merge_bad_alloc(execution::seq, IteratorTag());
    test_merge_bad_alloc(execution::par, IteratorTag());

    test_merge_bad_alloc_async(execution::seq(execution::task),
        IteratorTag());
    test_merge_bad_alloc_async(execution::par(execution::task),
        IteratorTag());
}

#endif
