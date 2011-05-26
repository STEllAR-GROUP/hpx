////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_4903DDAF_1B6D_461B_842B_9D8D4165F024)
#define HPX_4903DDAF_1B6D_461B_842B_9D8D4165F024

#include <hpx/runtime/agas/namespace/component_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/user_component.hpp>

namespace hpx { namespace agas 
{

template <typename Database>
struct user_component_namespace : component_namespace_base<
    components::client_base<
        user_component_namespace<Database>,
        stubs::user_component_namespace<Database>
    >,
    server::user_component_namespace<Database>
> {
    typedef component_namespace_base< 
        components::client_base<
            user_component_namespace<Database>,
            stubs::user_component_namespace<Database>
        >,
        server::user_component_namespace<Database>
    > base_type;

    explicit user_component_namespace(naming::id_type const& id
                                        = naming::invalid_id)
        : base_type(id) {}
};

}}

#endif // HPX_4903DDAF_1B6D_461B_842B_9D8D4165F024

