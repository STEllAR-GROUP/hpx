//  Copyright (c) 2006-2018 Maxim Khizhinsky
//  Copyright (c) 2020 Weile Wei
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/thread_support.hpp>

#include <cds/container/fcdeque.h>
#include <cds/gc/hp.h>    // for cds::HP (Hazard Pointer) SMR
#include <cds/init.h>     // for cds::Initialize and cds::Terminate

#include <cstddef>
#include <deque>

template <class Deque>
void run(Deque& dq)
{
    std::size_t const c_nSize = 100;
    int total_sum = 0;

    // push_front/pop_front
    for (int i = 0; i < static_cast<int>(c_nSize); ++i)
    {
        HPX_TEST(dq.push_front(i));
        total_sum += i;
    }
    HPX_TEST(dq.size() == c_nSize);

    int sum = 0;
    dq.apply([&sum](typename Deque::deque_type const& d) {
        for (auto const& el : d)
            sum += el;
    });
    HPX_TEST(sum == total_sum);

    std::size_t nCount = 0;
    int val;
    while (!dq.empty())
    {
        HPX_TEST(dq.pop_front(val));
        ++nCount;
        HPX_TEST(static_cast<int>(c_nSize - nCount) == val);
    }
    HPX_TEST(nCount == c_nSize);

    // push_back/pop_back
    for (int i = 0; i < static_cast<int>(c_nSize); ++i)
        HPX_TEST(dq.push_back(i));
    HPX_TEST(dq.size() == c_nSize);

    nCount = 0;
    while (!dq.empty())
    {
        HPX_TEST(dq.pop_back(val));
        ++nCount;
        HPX_TEST(static_cast<int>(c_nSize - nCount) == val);
    }
    HPX_TEST(nCount == c_nSize);

    // push_back/pop_front
    for (int i = 0; i < static_cast<int>(c_nSize); ++i)
        HPX_TEST(dq.push_back(i));
    HPX_TEST(dq.size() == c_nSize);

    nCount = 0;
    while (!dq.empty())
    {
        HPX_TEST(dq.pop_front(val));
        HPX_TEST(static_cast<int>(nCount) == val);
        ++nCount;
    }
    HPX_TEST(nCount == c_nSize);

    // push_front/pop_back
    for (int i = 0; i < static_cast<int>(c_nSize); ++i)
        HPX_TEST(dq.push_front(i));
    HPX_TEST(dq.size() == c_nSize);

    nCount = 0;
    while (!dq.empty())
    {
        HPX_TEST(dq.pop_back(val));
        HPX_TEST(static_cast<int>(nCount) == val);
        ++nCount;
    }
    HPX_TEST(nCount == c_nSize);

    // clear
    for (int i = 0; i < static_cast<int>(c_nSize); ++i)
        HPX_TEST(dq.push_front(i));
    HPX_TEST(dq.size() == c_nSize);

    HPX_TEST(dq.empty() == false);
    dq.clear();
    HPX_TEST(dq.empty());
}

int hpx_main(int, char**)
{
    using std_deque_type = cds::container::FCDeque<int>;

    using std_empty_wait_strategy_deque_type =
        cds::container::FCDeque<int, std::deque<int>,
            cds::container::fcdeque::make_traits<cds::opt::wait_strategy<
                cds::algo::flat_combining::wait_strategy::empty>>::type>;

    using std_multi_mutex_multi_condvar_deque_type =
        cds::container::FCDeque<int, std::deque<int>,
            cds::container::fcdeque::make_traits<
                cds::opt::wait_strategy<cds::algo::flat_combining::
                        wait_strategy::multi_mutex_multi_condvar<>>>::type>;

    using std_elimination_deque_type =
        cds::container::FCDeque<int, std::deque<int>,
            cds::container::fcdeque::make_traits<
                cds::opt::enable_elimination<true>>::type>;

    using std_elimination_single_mutex_single_condvar_deque_type =
        cds::container::FCDeque<int, std::deque<int>,
            cds::container::fcdeque::make_traits<
                cds::opt::enable_elimination<true>,
                cds::opt::wait_strategy<cds::algo::flat_combining::
                        wait_strategy::single_mutex_single_condvar<3>>>::type>;

    using std_statistics_deque_type =
        cds::container::FCDeque<int, std::deque<int>,
            cds::container::fcdeque::make_traits<
                cds::opt::stat<cds::container::fcdeque::stat<>>>::type>;

    using std_stat_single_mutex_multi_condvar_deque_type =
        cds::container::FCDeque<int, std::deque<int>,
            cds::container::fcdeque::make_traits<
                cds::opt::stat<cds::container::fcdeque::stat<>>,
                cds::opt::wait_strategy<cds::algo::flat_combining::
                        wait_strategy::single_mutex_multi_condvar<2>>>::type>;

    struct hpx_deque_traits
      : public cds::container::fcdeque::make_traits<
            cds::opt::enable_elimination<true>>::type
    {
        using lock_type = hpx::lcos::local::mutex;
    };
    using hpx_mutex_deque_type =
        cds::container::FCDeque<int, std::deque<int>, hpx_deque_traits>;

    // create different deque types
    std_deque_type std_dq;
    std_empty_wait_strategy_deque_type std_empty_wait_strategy_deque;
    std_multi_mutex_multi_condvar_deque_type
        std_multi_mutex_multi_condvar_deque;
    std_elimination_deque_type std_elimination_deque;
    std_elimination_single_mutex_single_condvar_deque_type
        std_elimination_single_mutex_single_condvar_deque;
    std_statistics_deque_type std_statistics_deque;
    std_stat_single_mutex_multi_condvar_deque_type
        std_stat_single_mutex_multi_condvar_deque;
    hpx_mutex_deque_type hpx_mutex_deque;

    // execute different deque objects
    run(std_dq);
    run(std_empty_wait_strategy_deque);
    run(std_multi_mutex_multi_condvar_deque);
    run(std_elimination_deque);
    run(std_elimination_single_mutex_single_condvar_deque);
    run(std_statistics_deque);
    run(std_stat_single_mutex_multi_condvar_deque);
    run(hpx_mutex_deque);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
