////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_60B7914E_21A5_4977_AA9C_8E66C44EE0FB)
#define HPX_60B7914E_21A5_4977_AA9C_8E66C44EE0FB

#include <boost/utility/binary.hpp>

namespace hpx { namespace agas
{

enum namespace_action_code
{
    invalid_request                         = 0,
    primary_ns_service                      = BOOST_BINARY_U(1000000),
    primary_ns_bulk_service                 = BOOST_BINARY_U(1000001),
    primary_ns_route                        = BOOST_BINARY_U(1000010),
    primary_ns_allocate                     = BOOST_BINARY_U(1000011),
    primary_ns_bind_gid                     = BOOST_BINARY_U(1000100),
    primary_ns_resolve_gid                  = BOOST_BINARY_U(1000101),
    primary_ns_resolve_locality             = BOOST_BINARY_U(1000110),
    primary_ns_free                         = BOOST_BINARY_U(1000111),
    primary_ns_unbind_gid                   = BOOST_BINARY_U(1001000),
    primary_ns_change_credit_non_blocking   = BOOST_BINARY_U(1001001),
    primary_ns_change_credit_sync           = BOOST_BINARY_U(1001010),
    primary_ns_localities                   = BOOST_BINARY_U(1001011),
    primary_ns_statistics_counter           = BOOST_BINARY_U(1001100),
    component_ns_service                    = BOOST_BINARY_U(0100000),
    component_ns_bulk_service               = BOOST_BINARY_U(0100001),
    component_ns_bind_prefix                = BOOST_BINARY_U(0100010),
    component_ns_bind_name                  = BOOST_BINARY_U(0100011),
    component_ns_resolve_id                 = BOOST_BINARY_U(0100100),
    component_ns_unbind                     = BOOST_BINARY_U(0100101),
    component_ns_iterate_types              = BOOST_BINARY_U(0100110),
    component_ns_statistics_counter         = BOOST_BINARY_U(0100111),
    symbol_ns_service                       = BOOST_BINARY_U(0010000),
    symbol_ns_bulk_service                  = BOOST_BINARY_U(0010001),
    symbol_ns_bind                          = BOOST_BINARY_U(0010010),
    symbol_ns_resolve                       = BOOST_BINARY_U(0010011),
    symbol_ns_unbind                        = BOOST_BINARY_U(0010100),
    symbol_ns_iterate_names                 = BOOST_BINARY_U(0010101),
    symbol_ns_statistics_counter            = BOOST_BINARY_U(0010110)
};

namespace detail
{
    struct counter_service_data
    {
        char const* const name_;              // name of performance counter
        namespace_action_code code_;          // corresponding AGAS service
        namespace_action_code service_code_;  // corresponding AGAS component
    };

    static counter_service_data const counter_services[] =
    {
        {   "count/bind_prefix"
          , component_ns_bind_prefix
          , component_ns_statistics_counter }
      , {   "count/bind_name"
          , component_ns_bind_name
          , component_ns_statistics_counter }
      , {   "count/resolve_id"
          , component_ns_resolve_id
          , component_ns_statistics_counter }
      , {   "count/unbind"
          , component_ns_unbind
          , component_ns_statistics_counter }
      , {   "count/iterate_types"
          , component_ns_iterate_types
          , component_ns_statistics_counter }
    };

    ///////////////////////////////////////////////////////////////////////////
    // get action code from counter type
    namespace_action_code retrieve_action_code(
        std::string const& name
      , error_code& ec = throws
        );

    // get service action code from counter type
    namespace_action_code retrieve_action_service_code(
        std::string const& name
      , error_code& ec = throws
        );
}
}}

#endif // HPX_60B7914E_21A5_4977_AA9C_8E66C44EE0FB

