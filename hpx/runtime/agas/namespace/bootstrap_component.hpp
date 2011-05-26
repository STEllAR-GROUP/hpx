////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_9E0F4EA1_8215_423F_B8FA_5E23CDA0A69E)
#define HPX_9E0F4EA1_8215_423F_B8FA_5E23CDA0A69E

#include <hpx/runtime/agas/namespace/component_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/bootstrap_component.hpp>

namespace hpx { namespace agas 
{

template <typename Database>
struct bootstrap_component_namespace : component_namespace_base<
    components::client_base<
        bootstrap_component_namespace<Database>,
        stubs::bootstrap_component_namespace<Database>
    >,
    server::bootstrap_component_namespace<Database>
> {
    typedef component_namespace_base< 
        components::client_base<
            bootstrap_component_namespace<Database>,
            stubs::bootstrap_component_namespace<Database>
        >,
        server::bootstrap_component_namespace<Database>
    > base_type;

    explicit bootstrap_component_namespace(naming::id_type const& id
                                             = naming::invalid_id)
        : base_type(id) {}
};

}}

#endif // HPX_9E0F4EA1_8215_423F_B8FA_5E23CDA0A69E

