//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_executors.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
// Define a base component which exposes the required interface
struct hello_world_server
  : hpx::components::executor_component<
        hpx::parallel::local_priority_queue_executor,
        hpx::components::component_base<hello_world_server> >
{
    typedef hpx::parallel::local_priority_queue_executor executor_type;
    typedef hpx::components::executor_component<
            executor_type, hpx::components::component_base<hello_world_server>
        > base_type;

    // run on all available cores
    hello_world_server()
      : base_type(executor_type(hpx::get_num_worker_threads()))
    {}

    void print() const
    {
        hpx::cout << "hello world\n" << hpx::flush;
    }

    HPX_DEFINE_COMPONENT_ACTION(hello_world_server, print, print_action);
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
