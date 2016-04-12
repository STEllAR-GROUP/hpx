//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/thread/locks.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
bool root_locality = false;
boost::int32_t final_result;
hpx::util::spinlock result_mutex;

void receive_result(boost::int32_t i)
{
    boost::lock_guard<hpx::util::spinlock> l(result_mutex);
    if (i > final_result)
        final_result = i;
}
HPX_PLAIN_ACTION(receive_result);

///////////////////////////////////////////////////////////////////////////////
boost::atomic<boost::int32_t> accumulator;

void increment(hpx::id_type const& there, boost::int32_t i)
{
    accumulator += i;
    hpx::apply(receive_result_action(), there, accumulator.load());
}
HPX_PLAIN_ACTION(increment);

void increment_with_future(hpx::id_type const& there,
    hpx::shared_future<boost::int32_t> fi)
{
    accumulator += fi.get();
    hpx::apply(receive_result_action(), there, accumulator.load());
}
HPX_PLAIN_ACTION(increment_with_future);

///////////////////////////////////////////////////////////////////////////////
struct increment_server
  : hpx::components::managed_component_base<increment_server>
{
    void call(hpx::id_type const& there, boost::int32_t i) const
    {
        accumulator += i;
        hpx::apply(receive_result_action(), there, accumulator.load());
    }

    HPX_DEFINE_COMPONENT_ACTION(increment_server, call);
};

typedef hpx::components::managed_component<increment_server> server_type;
HPX_REGISTER_COMPONENT(server_type, increment_server);

typedef increment_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::id_type here = hpx::find_here();
    hpx::id_type there = here;
    root_locality = true;
    if (hpx::get_num_localities_sync() > 1)
    {
        std::vector<hpx::id_type> localities = hpx::find_remote_localities();
        there = localities[0];
    }

    {
        increment_action inc;

        using hpx::util::placeholders::_1;

        hpx::apply(inc, there, here, 1);
        hpx::apply(hpx::util::bind(inc, there, here, 1));
        hpx::apply(hpx::util::bind(inc, there, here, _1), 1);
    }

    {
        increment_with_future_action inc;
        hpx::promise<boost::int32_t> p;
        hpx::shared_future<boost::int32_t> f = p.get_future();

        using hpx::util::placeholders::_1;

        hpx::apply(inc, there, here, f);
        hpx::apply(hpx::util::bind(inc, there, here, f));
        hpx::apply(hpx::util::bind(inc, there, here, _1), f);

        p.set_value(1);
    }

    {
        hpx::future<hpx::id_type> inc_f =
            hpx::components::new_<increment_server>(there);
        hpx::id_type inc = inc_f.get();

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;
        using hpx::util::placeholders::_3;

        call_action call;
        hpx::apply(call, inc, here, 1);
        hpx::apply(hpx::util::bind(call, inc, here, 1));
        hpx::apply(hpx::util::bind(call, inc, here, _1), 1);
        hpx::apply(hpx::util::bind(call, _1, here, 1), inc);
        hpx::apply(hpx::util::bind(call, _1, _2, 1), inc, here);
        hpx::apply(hpx::util::bind(call, _1, _2, _3), inc, here, 1);
    }

    {
        hpx::future<hpx::id_type> inc_f =
            hpx::components::new_<increment_server>(there);
        hpx::id_type inc = inc_f.get();

        hpx::apply<call_action>(inc, here, 1);
    }


    // Let finalize wait for every "apply" to be finished
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    accumulator.store(0);

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    // After hpx::init returns, all actions should have been executed
    // The final result is only accumulated on the root locality
    if(root_locality)
        HPX_TEST_EQ(final_result, 13);

    return hpx::util::report_errors();
}

