//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components/remote_object/new.hpp>

namespace hpx { namespace components { namespace remote_object
{
    naming::id_type
    new_impl(naming::id_type const & target_id, util::function<void(void**)> f)
    {
        naming::id_type object_id
            = stubs::remote_object::create_sync(target_id);
        
        stubs::remote_object::apply(object_id, f);

        return object_id;
    }
}}}

