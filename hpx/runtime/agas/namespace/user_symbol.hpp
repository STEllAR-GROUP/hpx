////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_CC982BE7_3A14_4A5F_B40A_403139722FAD)
#define HPX_CC982BE7_3A14_4A5F_B40A_403139722FAD

#include <hpx/runtime/agas/namespace/symbol_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/user_symbol.hpp>

namespace hpx { namespace agas 
{

template <typename Database>
struct user_symbol_namespace : symbol_namespace_base<
    components::client_base<
        user_symbol_namespace<Database>,
        stubs::user_symbol_namespace<Database>
    >,
    server::user_symbol_namespace<Database>
> {
    typedef symbol_namespace_base< 
        components::client_base<
            user_symbol_namespace<Database>,
            stubs::user_symbol_namespace<Database>
        >,
        server::user_symbol_namespace<Database>
    > base_type;

    explicit user_symbol_namespace(naming::id_type const& id
                                      = naming::invalid_id)
        : base_type(id) {}
};

}}

#endif // HPX_CC982BE7_3A14_4A5F_B40A_403139722FAD

