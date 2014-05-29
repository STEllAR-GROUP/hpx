//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/algorithm.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_iterator.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_foreach1(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10000);
    hpx::parallel::for_each(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        [](std::size_t& v) {
            v = 42;
        });

    // verify values
    std::for_each(boost::begin(c), boost::end(c),
        [](std::size_t v) {
            HPX_TEST_EQ(v, std::size_t(42));
        });
}

template <typename IteratorTag>
void test_foreach1(hpx::parallel::task_execution_policy const&, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10000);

    hpx::future<void> f =
        hpx::parallel::for_each(hpx::parallel::task,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            [](std::size_t& v) {
                v = 42;
            });
    f.wait();

    // verify values
    std::for_each(boost::begin(c), boost::end(c),
        [](std::size_t v) {
            HPX_TEST_EQ(v, std::size_t(42));
        });
}

template <typename IteratorTag>
void test_foreach1()
{
    test_foreach1(hpx::parallel::seq, IteratorTag());
    test_foreach1(hpx::parallel::par, IteratorTag());
    test_foreach1(hpx::parallel::vec, IteratorTag());
    test_foreach1(hpx::parallel::task, IteratorTag());
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_foreach1<std::random_access_iterator_tag>();
    test_foreach1<std::forward_iterator_tag>();
    test_foreach1<std::input_iterator_tag>();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}


