//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos_local.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/pack_traversal/unwrap.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
#include <boost/container/small_vector.hpp>

namespace hpx { namespace traits {
    // support unwrapping of boost::container::small_vector
    // Note: small_vector's allocator support is not 100% conforming
    template <typename NewType, typename OldType, std::size_t Size,
        typename OldAllocator>
    struct pack_traversal_rebind_container<NewType,
        boost::container::small_vector<OldType, Size, OldAllocator>>
    {
        using NewAllocator = typename std::allocator_traits<
            OldAllocator>::template rebind_alloc<NewType>;

        static boost::container::small_vector<NewType, Size, NewAllocator> call(
            boost::container::small_vector<OldType, Size, OldAllocator> const&)
        {
            // Create a new version of the container with a new allocator
            // instance
            return boost::container::small_vector<NewType, Size,
                NewAllocator>();
        }
    };
}}    // namespace hpx::traits

template <typename T>
using small_vector =
    boost::container::small_vector<T, 3, boost::container::new_allocator<T>>;

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::uint32_t> void_f_count;
std::atomic<std::uint32_t> int_f_count;

void void_f()
{
    ++void_f_count;
}
int int_f()
{
    ++int_f_count;
    return 42;
}

std::atomic<std::uint32_t> void_f1_count;
std::atomic<std::uint32_t> int_f1_count;

void void_f1(int)
{
    ++void_f1_count;
}
int int_f1(int i)
{
    ++int_f1_count;
    return i + 42;
}

std::atomic<std::uint32_t> int_f2_count;
int int_f2(int l, int r)
{
    ++int_f2_count;
    return l + r;
}

std::atomic<std::uint32_t> int_f_vector_count;

int int_f_vector(small_vector<int> const& vf)
{
    int sum = 0;
    for (int f : vf)
    {
        sum += f;
    }
    return sum;
}

void function_pointers(std::uint32_t num)
{
    void_f_count.store(0);
    int_f_count.store(0);
    void_f1_count.store(0);
    int_f1_count.store(0);
    int_f2_count.store(0);

    hpx::future<void> f1 =
        hpx::dataflow(hpx::util::unwrapping(&void_f1), hpx::async(&int_f));
    hpx::future<int> f2 = hpx::dataflow(hpx::util::unwrapping(&int_f1),
        hpx::dataflow(
            hpx::util::unwrapping(&int_f1), hpx::make_ready_future(42)));

    hpx::future<int> f3 = hpx::dataflow(hpx::util::unwrapping(&int_f2),
        hpx::dataflow(
            hpx::util::unwrapping(&int_f1), hpx::make_ready_future(42)),
        hpx::dataflow(
            hpx::util::unwrapping(&int_f1), hpx::make_ready_future(37)));

    int_f_vector_count.store(0);

    small_vector<hpx::future<int>> vf;
    vf.resize(num);
    for (std::uint32_t i = 0; i < num; ++i)
    {
        vf[i] = hpx::dataflow(
            hpx::util::unwrapping(&int_f1), hpx::make_ready_future(42));
    }
    hpx::future<int> f4 =
        hpx::dataflow(hpx::util::unwrapping(&int_f_vector), std::move(vf));

    hpx::future<int> f5 = hpx::dataflow(hpx::util::unwrapping(&int_f1),
        hpx::dataflow(
            hpx::util::unwrapping(&int_f1), hpx::make_ready_future(42)),
        hpx::dataflow(
            hpx::util::unwrapping(&void_f), hpx::make_ready_future()));

    f1.wait();
    HPX_TEST_EQ(f2.get(), 126);
    HPX_TEST_EQ(f3.get(), 163);
    HPX_TEST_EQ(f4.get(), int(num * 84));
    HPX_TEST_EQ(f5.get(), 126);
    HPX_TEST_EQ(void_f_count, 1u);
    HPX_TEST_EQ(int_f_count, 1u);
    HPX_TEST_EQ(void_f1_count, 1u);
    HPX_TEST_EQ(int_f1_count, 6u + num);
    HPX_TEST_EQ(int_f2_count, 1u);
}

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::uint32_t> future_void_f1_count;
std::atomic<std::uint32_t> future_void_f2_count;

void future_void_f1(hpx::future<void> f1)
{
    HPX_TEST(f1.is_ready());
    ++future_void_f1_count;
}
void future_void_sf1(hpx::shared_future<void> f1)
{
    HPX_TEST(f1.is_ready());
    ++future_void_f1_count;
}
void future_void_f2(hpx::future<void> f1, hpx::future<void> f2)
{
    HPX_TEST(f1.is_ready());
    HPX_TEST(f2.is_ready());
    ++future_void_f2_count;
}

std::atomic<std::uint32_t> future_int_f1_count;

int future_int_f1(hpx::future<void> f1)
{
    HPX_TEST(f1.is_ready());
    ++future_int_f1_count;
    return 1;
}

std::atomic<std::uint32_t> future_int_f_vector_count;

int future_int_f_vector(small_vector<hpx::future<int>>& vf)
{
    ++future_int_f_vector_count;

    int sum = 0;
    for (hpx::future<int>& f : vf)
    {
        HPX_TEST(f.is_ready());
        sum += f.get();
    }
    return sum;
}

void future_function_pointers(std::uint32_t num)
{
    future_int_f1_count.store(0);
    future_int_f_vector_count.store(0);

    future_int_f_vector_count.store(0);
    small_vector<hpx::future<int>> vf;
    vf.resize(num);
    for (std::uint32_t i = 0; i < num; ++i)
    {
        vf[i] = hpx::dataflow(&future_int_f1, hpx::make_ready_future());
    }
    hpx::future<int> f5 = hpx::dataflow(&future_int_f_vector, std::ref(vf));

    HPX_TEST_EQ(f5.get(), int(num));
    HPX_TEST_EQ(future_int_f1_count, num);
    HPX_TEST_EQ(future_int_f_vector_count, 1u);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    function_pointers(3);
    function_pointers(3);
    future_function_pointers(10);
    future_function_pointers(10);
    return hpx::util::report_errors();
}
