//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/components/remote_object/new_impl.hpp>
#include <hpx/components/remote_object/stubs/remote_object.hpp>

HPX_REGISTER_ACTION(hpx::components::server::create_component_action0<hpx::components::server::remote_object>,
    hpx_components_server_create_component_action0_hpx_components_server_remote_object
)

HPX_REGISTER_ACTION(hpx::components::server::remote_object_apply_action1<hpx::util::function<void(void**)> >,
    hpx_components_server_remote_object_apply_action1_hpx_util_function_void_void__)

namespace hpx { namespace components { namespace remote_object
{
    HPX_COMPONENT_EXPORT naming::id_type
    new_impl(
        naming::id_type const & target_id
      , util::function<void(void**)> const & ctor
      , util::function<void(void**)> const & dtor
    )
    {
        naming::id_type object_id
            = stubs::remote_object::create(target_id);

        lcos::future<void> apply_promise
            = stubs::remote_object::apply_async(object_id, ctor);
        lcos::future<void> dtor_promise
            = stubs::remote_object::set_dtor_async(object_id, dtor);

        apply_promise.get();
        dtor_promise.get();
        return object_id;
    }
}}}

/*
*/
