//  Copyright (c) 2007-2014 Hartmut Kaiser
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
hpx::lcos::local::condition_variable result_cv;

void receive_result(boost::int32_t i)
{
    hpx::util::spinlock::scoped_lock l(result_mutex);
    if (i > final_result)
        final_result = i;
    result_cv.notify_one();
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

    HPX_DEFINE_COMPONENT_ACTION(increment_server, call);
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
        hpx::apply_colocated(inc, there, here, 1);
    }

    {
        hpx::future<hpx::id_type> inc_f =
            hpx::components::new_<increment_server>(there);
        hpx::id_type where = inc_f.get();

        increment_action inc;
        hpx::apply_colocated(inc, where, here, 1);
    }

    {
        hpx::future<hpx::id_type> inc_f =
            hpx::components::new_<increment_server>(there);
        hpx::id_type where = inc_f.get();

        hpx::apply_colocated<increment_action>(where, here, 1);
    }

    // finalize will synchronize will all pending operations
    int result = hpx::finalize();

    hpx::util::spinlock::scoped_lock l(result_mutex);
    result_cv.wait_for(result_mutex, boost::chrono::seconds(1),
        hpx::util::bind(std::equal_to<boost::int32_t>(), boost::ref(final_result), 3));

    HPX_TEST_EQ(final_result, 3);

    return result;
}

int main(int argc, char* argv[])
{
    accumulator.store(0);

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

