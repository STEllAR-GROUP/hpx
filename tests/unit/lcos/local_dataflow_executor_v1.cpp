//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/unwrap.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::dataflow;
using hpx::util::bind;

using hpx::async;
using hpx::future;
using hpx::shared_future;

using hpx::make_ready_future;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;
using hpx::util::unwrapping;

///////////////////////////////////////////////////////////////////////////////

boost::atomic<std::uint32_t> void_f_count;
boost::atomic<std::uint32_t> int_f_count;

void void_f() {++void_f_count;}
int int_f() {++int_f_count; return 42; }

boost::atomic<std::uint32_t> void_f1_count;
boost::atomic<std::uint32_t> int_f1_count;

void void_f1(int) {++void_f1_count;}
int int_f1(int i) {++int_f1_count; return i+42; }

boost::atomic<std::uint32_t> int_f2_count;
int int_f2(int l, int r) {++int_f2_count; return l + r; }

boost::atomic<std::uint32_t> int_f_vector_count;

int int_f_vector(std::vector<int> const & vf)
{
    int sum = 0;
    for (int f : vf)
    {
        sum += f;
    }
    return sum;
}

template <typename Executor>
void function_pointers(Executor& exec)
{
    void_f_count.store(0);
    int_f_count.store(0);
    void_f1_count.store(0);
    int_f1_count.store(0);
    int_f2_count.store(0);

    future<void> f1 = dataflow(exec, unwrapping(&void_f1), async(&int_f));
    future<int>
        f2 = dataflow(exec,
            unwrapping(&int_f1)
          , dataflow(exec,
                unwrapping(&int_f1)
              , make_ready_future(42))
        );
    future<int>
        f3 = dataflow(exec,
            unwrapping(&int_f2)
          , dataflow(exec,
                unwrapping(&int_f1)
              , make_ready_future(42)
            )
          , dataflow(exec,
                unwrapping(&int_f1)
              , make_ready_future(37)
            )
        );

    int_f_vector_count.store(0);
    std::vector<future<int> > vf;
    for(std::size_t i = 0; i < 10; ++i)
    {
        vf.push_back(dataflow(exec, unwrapping(&int_f1), make_ready_future(42)));
    }
    future<int> f4 = dataflow(exec, unwrapping(&int_f_vector), std::move(vf));

    future<int>
        f5 = dataflow(exec,
            unwrapping(&int_f1)
          , dataflow(exec,
                unwrapping(&int_f1)
              , make_ready_future(42))
          , dataflow(exec,
                unwrapping(&void_f)
              , make_ready_future())
        );

    f1.wait();
    HPX_TEST_EQ(f2.get(), 126);
    HPX_TEST_EQ(f3.get(), 163);
    HPX_TEST_EQ(f4.get(), 10 * 84);
    HPX_TEST_EQ(f5.get(), 126);
    HPX_TEST_EQ(void_f_count, 1u);
    HPX_TEST_EQ(int_f_count, 1u);
    HPX_TEST_EQ(void_f1_count, 1u);
    HPX_TEST_EQ(int_f1_count, 16u);
    HPX_TEST_EQ(int_f2_count, 1u);
}

///////////////////////////////////////////////////////////////////////////////

boost::atomic<std::uint32_t> future_void_f1_count;
boost::atomic<std::uint32_t> future_void_f2_count;

void future_void_f1(future<void> f1)
{
    HPX_TEST(f1.is_ready());
    ++future_void_f1_count;
}

void future_void_sf1(shared_future<void> f1)
{
    HPX_TEST(f1.is_ready());
    ++future_void_f1_count;
}

void future_void_f2(future<void> f1, future<void> f2)
{
    HPX_TEST(f1.is_ready());
    HPX_TEST(f2.is_ready());
    ++future_void_f2_count;
}

boost::atomic<std::uint32_t> future_int_f1_count;
boost::atomic<std::uint32_t> future_int_f2_count;

int future_int_f1(future<void> f1)
{
    HPX_TEST(f1.is_ready());
    ++future_int_f1_count;
    return 1;
}

int future_int_f2(future<int> f1, future<int> f2)
{
    HPX_TEST(f1.is_ready()); HPX_TEST(f2.is_ready());
    ++future_int_f2_count;
    return f1.get() + f2.get();
}

boost::atomic<std::uint32_t> future_int_f_vector_count;

int future_int_f_vector(std::vector<future<int> >& vf)
{
    int sum = 0;
    for (future<int>& f : vf)
    {
        HPX_TEST(f.is_ready());
        sum += f.get();
    }
    return sum;
}

template <typename Executor>
void future_function_pointers(Executor& exec)
{
    future_void_f1_count.store(0);
    future_void_f2_count.store(0);
    future_int_f1_count.store(0);
    future_int_f2_count.store(0);

    future<void> f1
        = dataflow(exec,
            &future_void_f1
          , async(&future_void_sf1, shared_future<void>(make_ready_future()))
        );

    f1.wait();

    HPX_TEST_EQ(future_void_f1_count, 2u);
    future_void_f1_count.store(0);

    future<void> f2 = dataflow(exec,
        &future_void_f2
      , async(&future_void_sf1, shared_future<void>(make_ready_future()))
      , async(&future_void_sf1, shared_future<void>(make_ready_future()))
    );

    f2.wait();
    HPX_TEST_EQ(future_void_f1_count, 2u);
    HPX_TEST_EQ(future_void_f2_count, 1u);

    future_void_f1_count.store(0);
    future_void_f2_count.store(0);
    future_int_f1_count.store(0);
    future_int_f2_count.store(0);

    future<int> f3 = dataflow(exec,
        &future_int_f1
      , make_ready_future()
    );

    HPX_TEST_EQ(f3.get(), 1);
    HPX_TEST_EQ(future_int_f1_count, 1u);
    future_int_f1_count.store(0);

    future<int> f4 = dataflow(exec,
        &future_int_f2
      , dataflow(exec, &future_int_f1, make_ready_future())
      , dataflow(exec, &future_int_f1, make_ready_future())
    );

    HPX_TEST_EQ(f4.get(), 2);
    HPX_TEST_EQ(future_int_f1_count, 2u);
    HPX_TEST_EQ(future_int_f2_count, 1u);
    future_int_f1_count.store(0);
    future_int_f2_count.store(0);

    future_int_f_vector_count.store(0);
    std::vector<future<int> > vf;
    for(std::size_t i = 0; i < 10; ++i)
    {
        vf.push_back(dataflow(exec, &future_int_f1, make_ready_future()));
    }
    future<int> f5 = dataflow(exec, &future_int_f_vector, std::ref(vf));

    HPX_TEST_EQ(f5.get(), 10);
}

///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::uint32_t> void_f4_count;
boost::atomic<std::uint32_t> int_f4_count;

void void_f4(int) { ++void_f4_count; }
int int_f4(int i) { ++int_f4_count; return i+42; }

boost::atomic<std::uint32_t> void_f5_count;
boost::atomic<std::uint32_t> int_f5_count;

void void_f5(int, hpx::future<int>) { ++void_f5_count; }
int int_f5(int i, hpx::future<int> j) { ++int_f5_count; return i+j.get()+42; }

template <typename Executor>
void plain_arguments(Executor& exec)
{
    void_f4_count.store(0);
    int_f4_count.store(0);

    {
        future<void> f1 = dataflow(exec, &void_f4, 42);
        future<int> f2 = dataflow(exec, &int_f4, 42);

        f1.wait();
        HPX_TEST_EQ(void_f4_count, 1u);

        HPX_TEST_EQ(f2.get(), 84);
        HPX_TEST_EQ(int_f4_count, 1u);
    }

    void_f5_count.store(0);
    int_f5_count.store(0);

    {
        future<void> f1 = dataflow(exec, &void_f5, 42, async(&int_f));
        future<int> f2 = dataflow(exec, &int_f5, 42, async(&int_f));

        f1.wait();
        HPX_TEST_EQ(void_f5_count, 1u);

        HPX_TEST_EQ(f2.get(), 126);
        HPX_TEST_EQ(int_f5_count, 1u);
    }
}

template <typename Executor>
void plain_deferred_arguments(Executor& exec)
{
    void_f5_count.store(0);
    int_f5_count.store(0);

    {
        future<void> f1 = dataflow(
            exec, &void_f5, 42, async(hpx::launch::deferred, &int_f));
        future<int> f2 = dataflow(
            exec, &int_f5, 42, async(hpx::launch::deferred, &int_f));

        f1.wait();
        HPX_TEST_EQ(void_f5_count, 1u);

        HPX_TEST_EQ(f2.get(), 126);
        HPX_TEST_EQ(int_f5_count, 1u);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    {
        hpx::parallel::sequential_executor exec;
        function_pointers(exec);
        future_function_pointers(exec);
        plain_arguments(exec);
        plain_deferred_arguments(exec);
    }

    {
        hpx::parallel::parallel_executor exec;
        function_pointers(exec);
        future_function_pointers(exec);
        plain_arguments(exec);
        plain_deferred_arguments(exec);
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv, cfg), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}
