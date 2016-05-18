////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/unwrapped.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/assign.hpp>
#include <boost/atomic.hpp>

#include <numeric>
#include <string>
#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

using hpx::actions::action;

using hpx::util::unwrapped;
using hpx::util::unwrapped2;
using hpx::async;
using hpx::lcos::future;

using hpx::find_here;

using hpx::naming::id_type;


///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::size_t> void_counter;

void null_thread()
{
    ++void_counter;
}

HPX_PLAIN_ACTION(null_thread, null_action);

///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::size_t> result_counter;

bool null_result_thread()
{
    ++result_counter;
    return true;
}

HPX_PLAIN_ACTION(null_result_thread, null_result_action);

///////////////////////////////////////////////////////////////////////////////
int increment(int c)
{
    return c + 1;
}

int accumulate(std::vector<int> cs)
{
    return std::accumulate(cs.begin(), cs.end(), 0);
}

int add(int c1, int c2)
{
    return c1 + c2;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        boost::atomic<std::size_t> count(0);

        ///////////////////////////////////////////////////////////////////////
        id_type const here_ = find_here();

        ///////////////////////////////////////////////////////////////////////
        // Sync wait, single future, void return.
        {
            unwrapped(async<null_action>(here_));

            HPX_TEST_EQ(1U, void_counter.load());

            void_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Sync wait, single future, non-void return.
        {
            HPX_TEST_EQ(true, unwrapped(async<null_result_action>(here_)));
            HPX_TEST_EQ(1U, result_counter.load());

            result_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Sync wait, multiple futures, void return.
        {
            unwrapped(async<null_action>(here_)
               , async<null_action>(here_)
               , async<null_action>(here_));

            HPX_TEST_EQ(3U, void_counter.load());

            void_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Sync wait, multiple futures, non-void return.
        {
            hpx::util::tuple<bool, bool, bool> r
                = unwrapped(async<null_result_action>(here_)
                     , async<null_result_action>(here_)
                     , async<null_result_action>(here_));

            HPX_TEST_EQ(true, hpx::util::get<0>(r));
            HPX_TEST_EQ(true, hpx::util::get<1>(r));
            HPX_TEST_EQ(true, hpx::util::get<2>(r));
            HPX_TEST_EQ(3U, result_counter.load());

            result_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Sync wait, vector of futures, void return.
        {
            std::vector<future<void> > futures;
            futures.reserve(64);

            for (std::size_t i = 0; i < 64; ++i)
                futures.push_back(async<null_action>(here_));

            unwrapped(futures);

            HPX_TEST_EQ(64U, void_counter.load());

            void_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Sync wait, vector of futures, non-void return.
        {
            std::vector<future<bool> > futures;
            futures.reserve(64);

            std::vector<bool> values;
            values.reserve(64);

            for (std::size_t i = 0; i < 64; ++i)
                futures.push_back(async<null_result_action>(here_));

            values = unwrapped(futures);

            HPX_TEST_EQ(64U, result_counter.load());

            for (std::size_t i = 0; i < 64; ++i)
                HPX_TEST_EQ(true, values[i]);

            result_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Sync wait, vector of futures, non-void return ignored.
        {
            std::vector<future<bool> > futures;
            futures.reserve(64);

            for (std::size_t i = 0; i < 64; ++i)
                futures.push_back(async<null_result_action>(here_));

            unwrapped(futures);

            HPX_TEST_EQ(64U, result_counter.load());

            result_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Functional wrapper, single future
        {
            future<int> future = hpx::make_ready_future(42);

            HPX_TEST_EQ(unwrapped(&increment)(future), 42 + 1);
        }

        ///////////////////////////////////////////////////////////////////////
        // Functional wrapper, vector of future
        {
            std::vector<future<int> > futures;
            futures.reserve(64);

            for (std::size_t i = 0; i < 64; ++i)
                futures.push_back(hpx::make_ready_future(42));

            HPX_TEST_EQ(unwrapped(&accumulate)(futures), 42 * 64);
        }

        ///////////////////////////////////////////////////////////////////////
        // Functional wrapper, tuple of future
        {
            hpx::util::tuple<future<int>, future<int> > tuple =
                hpx::util::forward_as_tuple(
                    hpx::make_ready_future(42), hpx::make_ready_future(42));

            HPX_TEST_EQ(unwrapped(&add)(tuple), 42 + 42);
        }

        ///////////////////////////////////////////////////////////////////////
        // Functional wrapper, future of tuple of future
        {
            hpx::future<
                hpx::util::tuple<future<int>, future<int> >
            > tuple_future =
                hpx::make_ready_future(hpx::util::make_tuple(
                    hpx::make_ready_future(42), hpx::make_ready_future(42)));

            HPX_TEST_EQ(unwrapped2(&add)(tuple_future), 42 + 42);
        }
    }

    finalize();

    return report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}

