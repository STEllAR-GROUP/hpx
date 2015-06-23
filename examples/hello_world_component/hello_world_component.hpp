//  Copyright (c) 2012 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//[hello_world_hpp_getting_started
#if !defined(HELLO_WORLD_COMPONENT_HPP)
#define HELLO_WORLD_COMPONENT_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

namespace examples { namespace server
{

struct HPX_COMPONENT_EXPORT hello_world
  : hpx::components::managed_component_base<hello_world>
{
    void invoke();
    HPX_DEFINE_COMPONENT_ACTION(hello_world, invoke);
};

}

namespace stubs
{

struct hello_world : hpx::components::stub_base<server::hello_world>
{
    static void invoke(hpx::naming::id_type const& gid)
    {
        hpx::async<server::hello_world::invoke_action>(gid).get();
    }
};

}

struct hello_world
  : hpx::components::client_base<hello_world, stubs::hello_world>
{
    typedef hpx::components::client_base<hello_world, stubs::hello_world>
        base_type;

    hello_world(hpx::future<id_type> gid)
      : base_type(std::move(gid))
    {}

    void invoke()
    {
        this->base_type::invoke(this->get_id());
    }
};

}

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::hello_world::invoke_action, hello_world_invoke_action);

#endif // HELLO_WORLD_COMPONENT_HPP
//]

