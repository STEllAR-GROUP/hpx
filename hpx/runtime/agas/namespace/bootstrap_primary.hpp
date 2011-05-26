////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_56E6FCF2_6E1E_4059_B351_220E476E3B7B)
#define HPX_56E6FCF2_6E1E_4059_B351_220E476E3B7B

#include <hpx/runtime/agas/namespace/primary_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/bootstrap_primary.hpp>

namespace hpx { namespace agas 
{

template <typename Database, typename Protocol>
struct bootstrap_primary_namespace : primary_namespace_base<
    components::client_base<
        bootstrap_primary_namespace<Database, Protocol>,
        stubs::bootstrap_primary_namespace<Database, Protocol>
    >,
    server::bootstrap_primary_namespace<Database, Protocol>
> {        
    typedef primary_namespace_base<
        components::client_base<
            bootstrap_primary_namespace<Database, Protocol>,
            stubs::bootstrap_primary_namespace<Database, Protocol>
        >,
        server::bootstrap_primary_namespace<Database, Protocol>
    > base_type;

    explicit bootstrap_primary_namespace(naming::id_type const& id
                                            = naming::invalid_id)
      : base_type(id) {}
};

}}

#endif // HPX_56E6FCF2_6E1E_4059_B351_220E476E3B7B

