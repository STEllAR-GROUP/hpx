//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/unwrap.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::lcos::local::dataflow;
using hpx::util::bind;

using hpx::async;
using hpx::future;

using hpx::make_ready_future;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;
using hpx::util::unwrap;

///////////////////////////////////////////////////////////////////////////////

boost::atomic<boost::uint32_t> void_f_count;
boost::atomic<boost::uint32_t> int_f_count;

void void_f() {++void_f_count;}
int int_f() {++int_f_count; return 42; }

boost::atomic<boost::uint32_t> void_f1_count;
boost::atomic<boost::uint32_t> int_f1_count;

void void_f1(int) {++void_f1_count;}
int int_f1(int i) {++int_f1_count; return i+42; }

boost::atomic<boost::uint32_t> int_f2_count;
int int_f2(int l, int r) {++int_f2_count; return l + r; }

void function_pointers()
{
    void_f_count.store(0);
    int_f_count.store(0);
    void_f1_count.store(0);
    int_f1_count.store(0);
    int_f2_count.store(0);

    future<void> f1 = dataflow(unwrap(&void_f1), async(bind(&int_f)));
    future<int>
        f2 = dataflow(
            unwrap(&int_f1)
          , dataflow(
                unwrap(&int_f1)
              , make_ready_future(42))
        );
    future<int>
        f3 = dataflow(
            unwrap(&int_f2)
          , dataflow(
                unwrap(&int_f1)
              , make_ready_future(42)
            )
          , dataflow(
                unwrap(&int_f1)
              , make_ready_future(37)
            )
        );

    hpx::wait(f1);
    HPX_TEST_EQ(f2.get(), 126);
    HPX_TEST_EQ(f3.get(), 163);
    HPX_TEST_EQ(void_f_count, 0u);
    HPX_TEST_EQ(int_f_count, 1u);
    HPX_TEST_EQ(void_f1_count, 1u);
    HPX_TEST_EQ(int_f1_count, 4u);
    HPX_TEST_EQ(int_f2_count, 1u);
}

///////////////////////////////////////////////////////////////////////////////

boost::atomic<boost::uint32_t> future_void_f1_count;
boost::atomic<boost::uint32_t> future_void_f2_count;

void future_void_f1(future<void>) {++future_void_f1_count;}
void future_void_f2(future<void>, future<void>) {++future_void_f2_count;}

boost::atomic<boost::uint32_t> future_int_f1_count;
boost::atomic<boost::uint32_t> future_int_f2_count;

int future_int_f1(future<void>) {++future_int_f1_count; return 1;}
int future_int_f2(future<int> f1, future<int> f2)
{
    ++future_int_f2_count;
    return f1.get() + f2.get();
}

void future_function_pointers()
{
    future_void_f1_count.store(0);
    future_void_f2_count.store(0);

    future<void> f1
        = dataflow(
            &future_void_f1, async(bind(&future_void_f1, make_ready_future()))
        );
    
    hpx::wait(f1);

    HPX_TEST_EQ(future_void_f1_count, 2u);
    future_void_f1_count.store(0);

    future<void> f2 = dataflow(
        &future_void_f2
      , async(bind(&future_void_f1, make_ready_future()))
      , async(bind(&future_void_f1, make_ready_future()))
    );

    hpx::wait(f2);
    HPX_TEST_EQ(future_void_f1_count, 2u);
    HPX_TEST_EQ(future_void_f2_count, 1u);
    future_void_f1_count.store(0);
    future_void_f2_count.store(0);

    future<int> f3 = dataflow(
        &future_int_f1
      , make_ready_future()
    );

    HPX_TEST_EQ(f3.get(), 1);
    HPX_TEST_EQ(future_int_f1_count, 1u);
    future_int_f1_count.store(0);

    future<int> f4 = dataflow(
        &future_int_f2
      , dataflow(&future_int_f1, make_ready_future())
      , dataflow(&future_int_f1, make_ready_future())
    );

    HPX_TEST_EQ(f4.get(), 2);
    HPX_TEST_EQ(future_int_f1_count, 2u);
    HPX_TEST_EQ(future_int_f2_count, 1u);
    future_int_f1_count.store(0);
    future_int_f2_count.store(0);
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
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv, cfg), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}
