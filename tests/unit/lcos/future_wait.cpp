////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

using hpx::actions::plain_action0;
using hpx::actions::plain_result_action0;

using hpx::lcos::wait;
using hpx::async;
using hpx::lcos::future;

using hpx::find_here;

using hpx::naming::id_type;

///////////////////////////////////////////////////////////////////////////////
struct callback
{
  private:
    mutable std::size_t calls_;

  public:
    callback() : calls_(0) {} 

    template <
        typename T
    >
    void operator()(
        std::size_t 
      , T const& 
        ) const
    {
        ++calls_;
    }

    void operator()(
        std::size_t 
        ) const
    {
        ++calls_;
    }

    std::size_t count() const
    {
        return calls_;
    }

    void reset()
    {
        calls_ = 0;
    } 
};

///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::size_t> void_counter; 

void null_thread()
{
    ++void_counter;
}

typedef plain_action0<null_thread> null_action;

HPX_REGISTER_PLAIN_ACTION(null_action);

///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::size_t> result_counter; 

bool null_result_thread()
{
    ++result_counter;
    return true;
}

typedef plain_result_action0<bool, null_result_thread> null_result_action;

HPX_REGISTER_PLAIN_ACTION(null_result_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        callback cb;

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
            wait(async<null_action>(here_), cb);

            HPX_TEST_EQ(1U, cb.count());
            HPX_TEST_EQ(1U, void_counter.load());

            cb.reset();
            void_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Async wait, single future, non-void return.
        {
            wait(async<null_result_action>(here_), cb);

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

            wait(futures, cb);

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

            wait(futures, cb);

            HPX_TEST_EQ(64U, cb.count());
            HPX_TEST_EQ(64U, result_counter.load());

            cb.reset();
            result_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Sync wait, single future, void return.
        {
            wait(async<null_action>(here_));

            HPX_TEST_EQ(1U, void_counter.load());

            void_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Sync wait, single future, non-void return.
        {
            HPX_TEST_EQ(true, wait(async<null_result_action>(here_)));
            HPX_TEST_EQ(1U, result_counter.load());

            result_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Sync wait, multiple futures, void return.
        {
            wait(async<null_action>(here_)
               , async<null_action>(here_)
               , async<null_action>(here_));

            HPX_TEST_EQ(3U, void_counter.load());

            void_counter.store(0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Sync wait, multiple futures, non-void return.
        {
            boost::tuple<bool, bool, bool> r 
                = wait(async<null_result_action>(here_)
                     , async<null_result_action>(here_)
                     , async<null_result_action>(here_));

            HPX_TEST_EQ(true, boost::get<0>(r));
            HPX_TEST_EQ(true, boost::get<1>(r));
            HPX_TEST_EQ(true, boost::get<2>(r));
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

            wait(futures);

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

            wait(futures, values);

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

            wait(futures);

            HPX_TEST_EQ(64U, result_counter.load());

            result_counter.store(0);
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
        boost::lexical_cast<std::string>(hpx::thread::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}

