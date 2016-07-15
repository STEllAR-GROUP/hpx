////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/lcos/wait_each.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <string>
#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

using hpx::actions::action;

using hpx::lcos::wait_each;
using hpx::async;
using hpx::lcos::future;

using hpx::find_here;

using hpx::naming::id_type;

///////////////////////////////////////////////////////////////////////////////
struct callback
{
  private:
    mutable boost::atomic<std::size_t> * calls_;

  public:
    callback(boost::atomic<std::size_t> & calls) : calls_(&calls) {}

    template <
        typename T
    >
    void operator()(
        T const&
        ) const
    {
        ++(*calls_);
    }

    std::size_t count() const
    {
        return *calls_;
    }

    void reset()
    {
        calls_->store(0);
    }
};

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
int hpx_main(
    variables_map& vm
    )
{
    {
        boost::atomic<std::size_t> count(0);
        callback cb(count);

        ///////////////////////////////////////////////////////////////////////
        HPX_SANITY_EQ(0U, cb.count());

        cb(0);

        HPX_SANITY_EQ(1U, cb.count());

        cb.reset();

        HPX_SANITY_EQ(0U, cb.count());

        ///////////////////////////////////////////////////////////////////////
        id_type const here_ = find_here();

        ///////////////////////////////////////////////////////////////////////
        // Async wait, single future, void return.
        {
            wait_each(cb, async<null_action>(here_));

            HPX_TEST_EQ(1U, cb.count());
            HPX_TEST_EQ(1U, void_counter.load());

            cb.reset();
            void_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Async wait, single future, non-void return.
        {
            wait_each(cb, async<null_result_action>(here_));

            HPX_TEST_EQ(1U, cb.count());
            HPX_TEST_EQ(1U, result_counter.load());

            cb.reset();
            result_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Async wait, vector of futures, void return.
        {
            std::vector<future<void> > futures;
            futures.reserve(64);

            for (std::size_t i = 0; i < 64; ++i)
                futures.push_back(async<null_action>(here_));

            wait_each(cb, futures);

            HPX_TEST_EQ(64U, cb.count());
            HPX_TEST_EQ(64U, void_counter.load());

            cb.reset();
            void_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Async wait, vector of futures, non-void return.
        {
            std::vector<future<bool> > futures;
            futures.reserve(64);

            for (std::size_t i = 0; i < 64; ++i)
                futures.push_back(async<null_result_action>(here_));

            wait_each(cb, futures);

            HPX_TEST_EQ(64U, cb.count());
            HPX_TEST_EQ(64U, result_counter.load());

            cb.reset();
            result_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Async wait, single future, deferred.
        {
            wait_each(cb, async(hpx::launch::deferred, &null_thread));

            HPX_TEST_EQ(1U, cb.count());
            HPX_TEST_EQ(1U, void_counter.load());

            cb.reset();
            void_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Async wait, vector of futures, deferred.
        {
            std::vector<future<void> > futures;
            futures.reserve(64);

            for (std::size_t i = 0; i < 64; ++i)
                futures.push_back(async(
                    i % 3 ? hpx::launch::async : hpx::launch::deferred, &null_thread));

            wait_each(cb, futures);

            HPX_TEST_EQ(64U, cb.count());
            HPX_TEST_EQ(64U, void_counter.load());

            cb.reset();
            void_counter.store(0);
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
    std::vector<std::string> const cfg = {
        "hpx.os_threads=" +
            std::to_string(hpx::threads::hardware_concurrency())
    };

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}
