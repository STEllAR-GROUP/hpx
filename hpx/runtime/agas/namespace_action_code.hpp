////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_60B7914E_21A5_4977_AA9C_8E66C44EE0FB)
#define HPX_60B7914E_21A5_4977_AA9C_8E66C44EE0FB

#include <boost/utility/binary.hpp>

namespace hpx { namespace agas
{

// Base name used to register AGAS service instances
char const* const service_name = "/0/agas/";
char const* const performance_counter_basename = "/agas/";

enum namespace_action_code
{
    invalid_request                         = 0,

    locality_ns_service                     = BOOST_BINARY_U(1100000),
    locality_ns_bulk_service                = BOOST_BINARY_U(1100001),
    locality_ns_allocate                    = BOOST_BINARY_U(1100010),
    locality_ns_free                        = BOOST_BINARY_U(1100011),
    locality_ns_localities                  = BOOST_BINARY_U(1100100),
    locality_ns_num_localities              = BOOST_BINARY_U(1100101),
    locality_ns_num_threads                 = BOOST_BINARY_U(1100110),
    locality_ns_statistics_counter          = BOOST_BINARY_U(1100111),
    locality_ns_resolve_locality            = BOOST_BINARY_U(1101000),
    locality_ns_resolved_localities         = BOOST_BINARY_U(1101001),

    primary_ns_service                      = BOOST_BINARY_U(1000000),
    primary_ns_bulk_service                 = BOOST_BINARY_U(1000001),
    primary_ns_route                        = BOOST_BINARY_U(1000010),
    primary_ns_bind_gid                     = BOOST_BINARY_U(1000011),
    primary_ns_resolve_gid                  = BOOST_BINARY_U(1000100),
    primary_ns_unbind_gid                   = BOOST_BINARY_U(1000101),
    primary_ns_increment_credit             = BOOST_BINARY_U(1000110),
    primary_ns_decrement_credit             = BOOST_BINARY_U(1000111),
    primary_ns_allocate                     = BOOST_BINARY_U(1001000),
    primary_ns_statistics_counter           = BOOST_BINARY_U(1001001),

    component_ns_service                    = BOOST_BINARY_U(0100000),
    component_ns_bulk_service               = BOOST_BINARY_U(0100001),
    component_ns_bind_prefix                = BOOST_BINARY_U(0100010),
    component_ns_bind_name                  = BOOST_BINARY_U(0100011),
    component_ns_resolve_id                 = BOOST_BINARY_U(0100100),
    component_ns_unbind_name                = BOOST_BINARY_U(0100101),
    component_ns_iterate_types              = BOOST_BINARY_U(0100110),
    component_ns_get_component_type_name    = BOOST_BINARY_U(0100111),
    component_ns_num_localities             = BOOST_BINARY_U(0101000),
    component_ns_statistics_counter         = BOOST_BINARY_U(0101001),

    symbol_ns_service                       = BOOST_BINARY_U(0010000),
    symbol_ns_bulk_service                  = BOOST_BINARY_U(0010001),
    symbol_ns_bind                          = BOOST_BINARY_U(0010010),
    symbol_ns_resolve                       = BOOST_BINARY_U(0010011),
    symbol_ns_unbind                        = BOOST_BINARY_U(0010100),
    symbol_ns_iterate_names                 = BOOST_BINARY_U(0010101),
    symbol_ns_on_event                      = BOOST_BINARY_U(0010110),
    symbol_ns_statistics_counter            = BOOST_BINARY_U(0010111)
};

namespace detail
{
    enum counter_target
    {
        counter_target_invalid = -1
      , counter_target_count = 0
      , counter_target_time = 1
    };

    struct counter_service_data
    {
        char const* const name_;              // name of performance counter
        char const* const uom_;               // unit of measure of performance counter
        counter_target target_;               // target type of the counter
        namespace_action_code code_;          // corresponding AGAS service
        namespace_action_code service_code_;  // corresponding AGAS component
    };

    // counter description data for component namespace components
    static counter_service_data const component_namespace_services[] =
    {
        // counters exposing overall API invocation count and timings
        {   "component/count"
          , ""
          , counter_target_count
          , component_ns_statistics_counter
          , component_ns_statistics_counter }
      , {   "component/time"
          , "ns"
          , counter_target_time
          , component_ns_statistics_counter
          , component_ns_statistics_counter }
        // counters exposing API invocation counts
      , {   "count/bind_prefix"
          , ""
          , counter_target_count
          , component_ns_bind_prefix
          , component_ns_statistics_counter }
      , {   "count/bind_name"
          , ""
          , counter_target_count
          , component_ns_bind_name
          , component_ns_statistics_counter }
      , {   "count/resolve_id"
          , ""
          , counter_target_count
          , component_ns_resolve_id
          , component_ns_statistics_counter }
      , {   "count/unbind_name"
          , ""
          , counter_target_count
          , component_ns_unbind_name
          , component_ns_statistics_counter }
      , {   "count/iterate_types"
          , ""
          , counter_target_count
          , component_ns_iterate_types
          , component_ns_statistics_counter }
      , {   "count/get_component_typename"
          , ""
          , counter_target_count
          , component_ns_get_component_type_name
          , component_ns_statistics_counter }
      , {   "count/num_localities_type"
          , ""
          , counter_target_time
          , component_ns_num_localities
          , component_ns_statistics_counter }
      // counters exposing API timings
      , {   "time/bind_prefix"
          , "ns"
          , counter_target_time
          , component_ns_bind_prefix
          , component_ns_statistics_counter }
      , {   "time/bind_name"
          , "ns"
          , counter_target_time
          , component_ns_bind_name
          , component_ns_statistics_counter }
      , {   "time/resolve_id"
          , "ns"
          , counter_target_time
          , component_ns_resolve_id
          , component_ns_statistics_counter }
      , {   "time/unbind_name"
          , "ns"
          , counter_target_time
          , component_ns_unbind_name
          , component_ns_statistics_counter }
      , {   "time/iterate_types"
          , "ns"
          , counter_target_time
          , component_ns_iterate_types
          , component_ns_statistics_counter }
      , {   "time/get_component_typename"
          , "ns"
          , counter_target_time
          , component_ns_get_component_type_name
          , component_ns_statistics_counter }
      , {   "time/num_localities_type"
          , "ns"
          , counter_target_time
          , component_ns_num_localities
          , component_ns_statistics_counter }
    };
    static std::size_t const num_component_namespace_services =
        sizeof(component_namespace_services)/sizeof(component_namespace_services[0]);

    // counter description data for localities namespace components
    static counter_service_data const locality_namespace_services[] =
    {
        // counters exposing overall API invocation count and timings
        {   "locality/count"
          , ""
          , counter_target_count
          , locality_ns_statistics_counter
          , locality_ns_statistics_counter }
      , {   "locality/time"
          , "ns"
          , counter_target_time
          , locality_ns_statistics_counter
          , locality_ns_statistics_counter }
        // counters exposing API invocation counts
      , {   "count/free"
          , ""
          , counter_target_count
          , locality_ns_free
          , locality_ns_statistics_counter }
      , {   "count/localities"
          , ""
          , counter_target_count
          , locality_ns_localities
          , locality_ns_statistics_counter }
      , {   "count/num_localities"
          , ""
          , counter_target_count
          , locality_ns_num_localities
          , locality_ns_statistics_counter }
      , {   "count/num_threads"
          , ""
          , counter_target_count
          , locality_ns_num_threads
          , locality_ns_statistics_counter }
      , {   "count/resolve_locality"
          , ""
          , counter_target_count
          , locality_ns_resolve_locality
          , locality_ns_statistics_counter }
      , {   "count/resolved_localities"
          , ""
          , counter_target_time
          , locality_ns_resolved_localities
          , locality_ns_statistics_counter }
      // counters exposing API timings
      , {   "time/free"
          , "ns"
          , counter_target_time
          , locality_ns_free
          , locality_ns_statistics_counter }
      , {   "time/localities"
          , "ns"
          , counter_target_time
          , locality_ns_localities
          , locality_ns_statistics_counter }
      , {   "time/num_localities"
          , "ns"
          , counter_target_time
          , locality_ns_num_localities
          , locality_ns_statistics_counter }
      , {   "time/num_threads"
          , "ns"
          , counter_target_time
          , locality_ns_num_threads
          , locality_ns_statistics_counter }
      , {   "time/resolve_locality"
          , "ns"
          , counter_target_time
          , locality_ns_resolve_locality
          , locality_ns_statistics_counter }
      , {   "time/resolved_localities"
          , "ns"
          , counter_target_time
          , locality_ns_resolved_localities
          , locality_ns_statistics_counter }
    };
    static std::size_t const num_locality_namespace_services =
        sizeof(locality_namespace_services)/sizeof(locality_namespace_services[0]);

    // counter description data for primary namespace components
    static counter_service_data const primary_namespace_services[] =
    {
        // counters exposing overall API invocation count and timings
        {   "primary/count"
          , ""
          , counter_target_count
          , primary_ns_statistics_counter
          , primary_ns_statistics_counter }
      , {   "primary/time"
          , "ns"
          , counter_target_time
          , primary_ns_statistics_counter
          , primary_ns_statistics_counter }
        // counters exposing API invocation counts
      , {   "count/route"
          , ""
          , counter_target_count
          , primary_ns_route
          , primary_ns_statistics_counter }
      , {   "count/bind_gid"
          , ""
          , counter_target_count
          , primary_ns_bind_gid
          , primary_ns_statistics_counter }
      , {   "count/resolve_gid"
          , ""
          , counter_target_count
          , primary_ns_resolve_gid
          , primary_ns_statistics_counter }
      , {   "count/unbind_gid"
          , ""
          , counter_target_count
          , primary_ns_unbind_gid
          , primary_ns_statistics_counter }
      , {   "count/increment_credit"
          , ""
          , counter_target_count
          , primary_ns_increment_credit
          , primary_ns_statistics_counter }
      , {   "count/decrement_credit"
          , ""
          , counter_target_count
          , primary_ns_decrement_credit
          , primary_ns_statistics_counter }
      , {   "count/allocate"
          , ""
          , counter_target_count
          , primary_ns_allocate
          , primary_ns_statistics_counter }
      // counters exposing API timings
      , {   "time/route"
          , "ns"
          , counter_target_time
          , primary_ns_route
          , primary_ns_statistics_counter }
      , {   "time/bind_gid"
          , "ns"
          , counter_target_time
          , primary_ns_bind_gid
          , primary_ns_statistics_counter }
      , {   "time/resolve_gid"
          , "ns"
          , counter_target_time
          , primary_ns_resolve_gid
          , primary_ns_statistics_counter }
      , {   "time/unbind_gid"
          , "ns"
          , counter_target_time
          , primary_ns_unbind_gid
          , primary_ns_statistics_counter }
      , {   "time/increment_credit"
          , "ns"
          , counter_target_time
          , primary_ns_increment_credit
          , primary_ns_statistics_counter }
      , {   "time/decrement_credit"
          , "ns"
          , counter_target_time
          , primary_ns_decrement_credit
          , primary_ns_statistics_counter }
      , {   "time/allocate"
          , "ns"
          , counter_target_time
          , primary_ns_allocate
          , primary_ns_statistics_counter }
    };
    static std::size_t const num_primary_namespace_services =
        sizeof(primary_namespace_services)/sizeof(primary_namespace_services[0]);

    // counter description data for symbol namespace components
    static counter_service_data const symbol_namespace_services[] =
    {
        // counters exposing overall API invocation count and timings
        {   "symbol/count"
          , ""
          , counter_target_count
          , symbol_ns_statistics_counter
          , symbol_ns_statistics_counter }
      , {   "symbol/time"
          , "ns"
          , counter_target_time
          , symbol_ns_statistics_counter
          , symbol_ns_statistics_counter }
        // counters exposing API invocation counts
      , {   "count/bind"
          , ""
          , counter_target_count
          , symbol_ns_bind
          , symbol_ns_statistics_counter }
      , {   "count/resolve"
          , ""
          , counter_target_count
          , symbol_ns_resolve
          , symbol_ns_statistics_counter }
      , {   "count/unbind"
          , ""
          , counter_target_count
          , symbol_ns_unbind
          , symbol_ns_statistics_counter }
      , {   "count/iterate_names"
          , ""
          , counter_target_count
          , symbol_ns_iterate_names
          , symbol_ns_statistics_counter }
      , {   "count/on_symbol_namespace_event"
          , ""
          , counter_target_count
          , symbol_ns_on_event
          , symbol_ns_statistics_counter }
      // counters exposing API timings
      , {   "time/bind"
          , "ns"
          , counter_target_time
          , symbol_ns_bind
          , symbol_ns_statistics_counter }
      , {   "time/resolve"
          , "ns"
          , counter_target_time
          , symbol_ns_resolve
          , symbol_ns_statistics_counter }
      , {   "time/unbind"
          , "ns"
          , counter_target_time
          , symbol_ns_unbind
          , symbol_ns_statistics_counter }
      , {   "time/iterate_names"
          , "ns"
          , counter_target_time
          , symbol_ns_iterate_names
          , symbol_ns_statistics_counter }
      , {   "time/on_symbol_namespace_event"
          , "ns"
          , counter_target_time
          , symbol_ns_on_event
          , symbol_ns_statistics_counter }
    };
    static std::size_t const num_symbol_namespace_services =
        sizeof(symbol_namespace_services)/sizeof(symbol_namespace_services[0]);

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

namespace server
{
    // forward declarations
    struct HPX_EXPORT locality_namespace;
    struct HPX_EXPORT primary_namespace;
    struct HPX_EXPORT component_namespace;
    struct HPX_EXPORT symbol_namespace;
}

}}

#endif // HPX_60B7914E_21A5_4977_AA9C_8E66C44EE0FB

