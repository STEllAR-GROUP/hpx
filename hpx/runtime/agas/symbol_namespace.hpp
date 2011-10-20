////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7)
#define HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7

#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/stubs/symbol_namespace.hpp>

namespace hpx { namespace agas
{

struct symbol_namespace :
    components::client_base<symbol_namespace, stubs::symbol_namespace>
{
    // {{{ nested types
    typedef components::client_base<
        symbol_namespace, stubs::symbol_namespace
    > base_type;

    typedef server::symbol_namespace server_type;

    typedef server_type::iterate_names_function_type
        iterate_names_function_type;
    // }}}

    symbol_namespace()
      : base_type(bootstrap_symbol_namespace_id())
    {}

    explicit symbol_namespace(naming::id_type const& id)
      : base_type(id)
    {}

    lcos::promise<response> service_async(
        request const& req
        )
    {
        return this->base_type::service_async(this->gid_, req);
    }

    response service(
        request const& req
      , error_code& ec = throws
        )
    {
        return this->base_type::service(this->gid_, req, ec);
    }
};

}}

#endif // HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7

