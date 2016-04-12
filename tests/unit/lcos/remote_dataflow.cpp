//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/unwrapped.hpp>

#include <boost/atomic.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
boost::atomic<boost::uint32_t> void_f_count;

void void_f(hpx::future<int> &&) { ++void_f_count; }
HPX_PLAIN_ACTION(void_f);

hpx::id_type id_f() { return hpx::find_here(); }
HPX_PLAIN_ACTION(id_f);

void plain_actions(hpx::id_type const& there)
{
    hpx::id_type here = hpx::find_here();

    {
        void_f_count.store(0);

        hpx::future<void> f1 = hpx::dataflow(void_f_action(), here,
            hpx::make_ready_future(42));
        hpx::future<void> f2 = hpx::dataflow<void_f_action>(here,
            hpx::make_ready_future(42));

        f1.get();
        f2.get();

        HPX_TEST_EQ(void_f_count, 2u);
    }

    {
        hpx::future<hpx::id_type> f1 = hpx::dataflow(id_f_action(), there);
        hpx::future<hpx::id_type> f2 = hpx::dataflow<id_f_action>(there);

        HPX_TEST(there == f1.get());
        HPX_TEST(there == f2.get());
    }

    hpx::launch policies[] =
    {
        hpx::launch::async,
//        hpx::launch::deferred,    // FIXME: enable once #1523 has been fixed
        hpx::launch::sync
    };

    for (int i = 0; i != 2; ++i)
    {
        {
            void_f_count.store(0);

            hpx::future<void> f1 = hpx::dataflow(policies[i],
                void_f_action(), here, hpx::make_ready_future(42));
            hpx::future<void> f2 = hpx::dataflow<void_f_action>(policies[i],
                here, hpx::make_ready_future(42));

            f1.get();
            f2.get();

            HPX_TEST_EQ(void_f_count, 2u);
        }

        {
            hpx::future<hpx::id_type> f1 = hpx::dataflow(
                policies[i], id_f_action(), there);
            hpx::future<hpx::id_type> f2 = hpx::dataflow<id_f_action>(
                policies[i], there);

            HPX_TEST(there == f1.get());
            HPX_TEST(there == f2.get());
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    for (hpx::id_type const& id : hpx::find_all_localities())
    {
        plain_actions(id);
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
      "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}
