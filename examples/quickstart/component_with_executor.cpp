//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/thread_executors.hpp>

///////////////////////////////////////////////////////////////////////////////
// Define a base component which exposes the required interface
struct hello_world_server
  : hpx::components::component_base<hello_world_server>
{
    hello_world_server()
      : sched_(hpx::get_num_worker_threads())   // run on all available cores
    {}

    void print() const
    {
        hpx::cout << "hello world\n" << hpx::flush;
    }

    HPX_DEFINE_COMPONENT_ACTION(hello_world_server, print, print_action);

    ///////////////////////////////////////////////////////////////////////////
    // wrap given function into a nullary function as expected by the executor
    static void func(hpx::threads::thread_function_type f)
    {
        f(hpx::threads::wait_signaled);
    }

    /// This is the default hook implementation for schedule_thread which
    /// forwards to the default scheduler.
    static void schedule_thread(hpx::naming::address::address_type lva,
        hpx::threads::thread_init_data& data,
        hpx::threads::thread_state_enum initial_state)
    {
        char const* desc = 0;
#ifdef HPX_HAVE_THREAD_DESCRIPTION
        desc = data.description;
#endif

        hpx::get_lva<hello_world_server>::call(lva)->sched_.add(
            hpx::util::bind(
                hpx::util::one_shot(&hello_world_server::func),
                std::move(data.func)),
            desc, initial_state);
    }

private:
    hpx::threads::executors::local_priority_queue_executor sched_;
};

typedef hpx::components::component<hello_world_server> server_type;
HPX_REGISTER_COMPONENT(server_type, hello_world_server);

typedef hello_world_server::print_action print_action;
HPX_REGISTER_ACTION_DECLARATION(print_action);
HPX_REGISTER_ACTION(print_action);

///////////////////////////////////////////////////////////////////////////////
struct hello_world
  : hpx::components::client_base<
        hello_world, hpx::components::stub_base<hello_world_server> >
{
    typedef hpx::components::client_base<
        hello_world, hpx::components::stub_base<hello_world_server>
    > base_type;

    hello_world(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    void print()
    {
        hpx::async<print_action>(this->get_id()).get();
    }
};

///////////////////////////////////////////////////////////////////////////////
int main()
{
    hello_world hw = hpx::new_<hello_world_server>(hpx::find_here());
    hw.print();

    return 0;
}
