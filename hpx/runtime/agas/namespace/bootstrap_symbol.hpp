////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D0B08B2A_F84D_455C_88C3_D3F43CB00377)
#define HPX_D0B08B2A_F84D_455C_88C3_D3F43CB00377

#include <hpx/runtime/agas/namespace/symbol_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/bootstrap_symbol.hpp>

namespace hpx { namespace agas 
{

template <typename Database>
struct bootstrap_symbol_namespace : symbol_namespace_base<
    components::client_base<
        bootstrap_symbol_namespace<Database>,
        stubs::bootstrap_symbol_namespace<Database>
    >,
    server::bootstrap_symbol_namespace<Database>
> {
    typedef symbol_namespace_base< 
        components::client_base<
            bootstrap_symbol_namespace<Database>,
            stubs::bootstrap_symbol_namespace<Database>
        >,
        server::bootstrap_symbol_namespace<Database>
    > base_type;

    explicit bootstrap_symbol_namespace(naming::id_type const& id
                                             = naming::invalid_id)
        : base_type(id) {}
};

}}

#endif // HPX_D0B08B2A_F84D_455C_88C3_D3F43CB00377

