//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
boost::int32_t final_result;
hpx::util::spinlock result_mutex;

void receive_result(boost::int32_t i)
{
    hpx::util::spinlock::scoped_lock l(result_mutex);
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

///////////////////////////////////////////////////////////////////////////////
struct increment_server
  : hpx::components::managed_component_base<increment_server>
{
    void call(hpx::id_type const& there, boost::int32_t i) const
    {
        accumulator += i;
        hpx::apply(receive_result_action(), there, accumulator.load());
    }

    HPX_DEFINE_COMPONENT_CONST_ACTION(increment_server, call);
};

typedef hpx::components::managed_component<increment_server> server_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(server_type, increment_server);

typedef increment_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::id_type here = hpx::find_here();
    hpx::id_type there = here;
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

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    accumulator.store(0);

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    HPX_TEST_EQ(final_result, 10);

    return hpx::util::report_errors();
}

