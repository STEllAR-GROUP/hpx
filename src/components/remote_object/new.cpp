//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components/remote_object/new_impl.hpp>
#include <hpx/components/remote_object/stubs/remote_object.hpp>

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
            = stubs::remote_object::create_sync(target_id);

        lcos::promise<void> apply_promise
            = stubs::remote_object::apply_async(object_id, ctor);
        lcos::promise<void> dtor_promise
            = stubs::remote_object::set_dtor_async(object_id, dtor);

        apply_promise.get();
        dtor_promise.get();
        return object_id;
    }
}}}

