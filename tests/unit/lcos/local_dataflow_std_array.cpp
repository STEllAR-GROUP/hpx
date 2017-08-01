//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/unwrap.hpp>

#include <boost/atomic.hpp>

#include <array>
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

int int_f_vector(std::array<int, 10> const & vf)
{
    int sum = 0;
    for (int f : vf)
    {
        sum += f;
    }
    return sum;
}

void function_pointers()
{
    void_f_count.store(0);
    int_f_count.store(0);
    void_f1_count.store(0);
    int_f1_count.store(0);
    int_f2_count.store(0);

    future<void> f1 = dataflow(unwrapping(&void_f1), async(&int_f));
    future<int>
        f2 = dataflow(
            unwrapping(&int_f1)
          , dataflow(
                unwrapping(&int_f1)
              , make_ready_future(42))
        );
    future<int>
        f3 = dataflow(
            unwrapping(&int_f2)
          , dataflow(
                unwrapping(&int_f1)
              , make_ready_future(42)
            )
          , dataflow(
                unwrapping(&int_f1)
              , make_ready_future(37)
            )
        );

    int_f_vector_count.store(0);
    std::array<future<int>, 10> vf;
    for(std::size_t i = 0; i < 10; ++i)
    {
        vf[i] = dataflow(unwrapping(&int_f1), make_ready_future(42));
    }
    future<int> f4 = dataflow(unwrapping(&int_f_vector), std::move(vf));

    future<int>
        f5 = dataflow(
            unwrapping(&int_f1)
          , dataflow(
                unwrapping(&int_f1)
              , make_ready_future(42))
          , dataflow(
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
    { HPX_TEST(f1.is_ready()); ++future_void_f1_count;}
void future_void_sf1(shared_future<void> f1)
    { HPX_TEST(f1.is_ready());++future_void_f1_count;}
void future_void_f2(future<void> f1, future<void> f2)
    { HPX_TEST(f1.is_ready()); HPX_TEST(f2.is_ready()); ++future_void_f2_count;}

boost::atomic<std::uint32_t> future_int_f1_count;

int future_int_f1(future<void> f1) { HPX_TEST(f1.is_ready());
    ++future_int_f1_count; return 1;}

boost::atomic<std::uint32_t> future_int_f_vector_count;

int future_int_f_vector(std::array<future<int>, 10>& vf)
{
    ++future_int_f_vector_count;

    int sum = 0;
    for (future<int>& f : vf)
    {
        HPX_TEST(f.is_ready());
        sum += f.get();
    }
    return sum;
}

void future_function_pointers()
{
    future_int_f1_count.store(0);
    future_int_f_vector_count.store(0);

    future_int_f_vector_count.store(0);
    std::array<future<int>, 10> vf;
    for(std::size_t i = 0; i < 10; ++i)
    {
        vf[i] = dataflow(&future_int_f1, make_ready_future());
    }
    future<int> f5 = dataflow(&future_int_f_vector, std::ref(vf));

    HPX_TEST_EQ(f5.get(), 10);
    HPX_TEST_EQ(future_int_f1_count, 10u);
    HPX_TEST_EQ(future_int_f_vector_count, 1u);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    function_pointers();
    future_function_pointers();

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

#endif
