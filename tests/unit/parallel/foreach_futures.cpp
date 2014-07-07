//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/algorithm.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each_futures(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<hpx::future<std::size_t> >::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // prepare some test data
    std::vector<std::size_t> idx = test::random_iota(1003);

    std::vector<hpx::promise<std::size_t> > p(1003);
    std::vector<hpx::future<std::size_t> > f = test::fill_with_futures(p);

    hpx::future<void> done_init = hpx::async(
        hpx::util::bind(&test::make_ready, boost::ref(p), boost::ref(idx)));

    // this is the actual test
    std::vector<std::size_t> d = test::iota(1003, std::rand());
    hpx::parallel::for_each(policy,
        iterator(boost::begin(f)), iterator(boost::end(f)),
        [&d](hpx::future<std::size_t>& fut) {
            std::size_t v = fut.get();
            HPX_TEST(v < 1003);
            d[v] = v;
        });

    done_init.wait();

    // verify values
    std::vector<std::size_t> c = test::iota(1003, 0);

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename IteratorTag>
void test_for_each_futures(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<hpx::future<std::size_t> >::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // prepare some test data
    std::vector<std::size_t> idx = test::random_iota(1003);

    std::vector<hpx::promise<std::size_t> > p(1003);
    std::vector<hpx::future<std::size_t> > f = test::fill_with_futures(p);

    hpx::future<void> done_init = hpx::async(
        hpx::util::bind(&test::make_ready, boost::ref(p), boost::ref(idx)));

    // this is the actual test
    std::vector<std::size_t> d = test::iota(1003, std::rand());
    hpx::future<void> fut =
        hpx::parallel::for_each(hpx::parallel::task,
            iterator(boost::begin(f)), iterator(boost::end(f)),
            [&d](hpx::future<std::size_t>& fut) {
                std::size_t v = fut.get();
                HPX_TEST(v < 1003);
                d[v] = v;
            });
    fut.wait();

    done_init.wait();

    // verify values
    std::vector<std::size_t> c = test::iota(1003, 0);

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename IteratorTag>
void test_for_each_futures()
{
    using namespace hpx::parallel;

    test_for_each_futures(seq, IteratorTag());
    test_for_each_futures(par, IteratorTag());
    test_for_each_futures(par_vec, IteratorTag());
    test_for_each_futures(task, IteratorTag());

    test_for_each_futures(execution_policy(seq), IteratorTag());
    test_for_each_futures(execution_policy(par), IteratorTag());
    test_for_each_futures(execution_policy(par_vec), IteratorTag());
    test_for_each_futures(execution_policy(task), IteratorTag());
}

void for_each_futures_test()
{
    test_for_each_futures<std::random_access_iterator_tag>();
    test_for_each_futures<std::forward_iterator_tag>();
    test_for_each_futures<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    for_each_futures_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}


