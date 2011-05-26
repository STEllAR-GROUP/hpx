////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D31F64EE_9A26_4C62_A419_C15E01FA690E)
#define HPX_D31F64EE_9A26_4C62_A419_C15E01FA690E

#include <hpx/runtime/agas/namespace/primary_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/user_primary.hpp>

namespace hpx { namespace agas 
{

template <typename Database, typename Protocol>
struct user_primary_namespace : primary_namespace_base<
    components::client_base<
        user_primary_namespace<Database, Protocol>,
        stubs::user_primary_namespace<Database, Protocol>
    >,
    server::user_primary_namespace<Database, Protocol>
> {        
    typedef primary_namespace_base<
        components::client_base<
            user_primary_namespace<Database, Protocol>,
            stubs::user_primary_namespace<Database, Protocol>
        >,
        server::user_primary_namespace<Database, Protocol>
    > base_type;

    explicit user_primary_namespace(naming::id_type const& id
                                      = naming::invalid_id)
        : base_type(id) {}
};

}}

#endif // HPX_D31F64EE_9A26_4C62_A419_C15E01FA690E

