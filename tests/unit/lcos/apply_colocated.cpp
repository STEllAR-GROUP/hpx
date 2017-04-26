//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <cstdint>
#include <mutex>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
bool on_shutdown_executed = false;
std::uint32_t locality_id = std::uint32_t(-1);

std::int32_t final_result = 0;
hpx::util::spinlock result_mutex;

void receive_result(std::int32_t i)
{
    std::lock_guard<hpx::util::spinlock> l(result_mutex);
    if (i > final_result)
        final_result = i;
}
HPX_PLAIN_ACTION(receive_result);

///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::int32_t> accumulator;

void increment(hpx::id_type const& there, std::int32_t i)
{
    locality_id = hpx::get_locality_id();

    accumulator += i;
    hpx::apply(receive_result_action(), there, accumulator.load());
}
HPX_PLAIN_ACTION(increment);

///////////////////////////////////////////////////////////////////////////////
struct increment_server
  : hpx::components::managed_component_base<increment_server>
{
    void call(hpx::id_type const& there, std::int32_t i) const
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
void on_shutdown()
{
    std::lock_guard<hpx::util::spinlock> l(result_mutex);
    HPX_TEST_EQ(final_result, 3);

    on_shutdown_executed = true;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    locality_id = hpx::get_locality_id();

    hpx::id_type here = hpx::find_here();
    hpx::id_type there = here;
    if (hpx::get_num_localities(hpx::launch::sync) > 1)
    {
        std::vector<hpx::id_type> localities = hpx::find_remote_localities();
        there = localities[0];
    }

    {
        increment_action inc;
        hpx::apply(inc, hpx::colocated(there), here, 1);
    }

    {
        hpx::future<hpx::id_type> inc_f =
            hpx::components::new_<increment_server>(there);
        hpx::id_type where = inc_f.get();

        increment_action inc;
        hpx::apply(inc, hpx::colocated(where), here, 1);
    }

    {
        hpx::future<hpx::id_type> inc_f =
            hpx::components::new_<increment_server>(there);
        hpx::id_type where = inc_f.get();

        hpx::apply<increment_action>(hpx::colocated(where), here, 1);
    }

    // register function which will verify final result
    hpx::register_shutdown_function(on_shutdown);

    HPX_TEST_EQ(hpx::finalize(), 0);

    return 0;
}

int main(int argc, char* argv[])
{
    accumulator.store(0);

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    HPX_TEST_NEQ(std::uint32_t(-1), locality_id);
    HPX_TEST(on_shutdown_executed || 0 != locality_id);

    return hpx::util::report_errors();
}

