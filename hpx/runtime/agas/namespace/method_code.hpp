////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_60B7914E_21A5_4977_AA9C_8E66C44EE0FB)
#define HPX_60B7914E_21A5_4977_AA9C_8E66C44EE0FB

#include <boost/utility/binary.hpp>

namespace hpx { namespace agas
{

enum method_code
{ 
    invalid_request             = 0,
    primary_ns_bind_locality    = BOOST_BINARY_U(1000000), 
    primary_ns_bind_gid         = BOOST_BINARY_U(1000001), 
    primary_ns_page_fault       = BOOST_BINARY_U(1000010), 
    primary_ns_unbind_locality  = BOOST_BINARY_U(1000011), 
    primary_ns_unbind_gid       = BOOST_BINARY_U(1000100), 
    primary_ns_increment        = BOOST_BINARY_U(1000101), 
    primary_ns_decrement        = BOOST_BINARY_U(1000110), 
    primary_ns_localities       = BOOST_BINARY_U(1000111), 
    component_ns_bind_prefix    = BOOST_BINARY_U(0100000), 
    component_ns_bind_name      = BOOST_BINARY_U(0100001), 
    component_ns_resolve_id     = BOOST_BINARY_U(0100010), 
    component_ns_resolve_name   = BOOST_BINARY_U(0100011), 
    component_ns_unbind         = BOOST_BINARY_U(0100100), 
    symbol_ns_bind              = BOOST_BINARY_U(0010000), 
    symbol_ns_resolve           = BOOST_BINARY_U(0010001), 
    symbol_ns_unbind            = BOOST_BINARY_U(0010010), 
    symbol_ns_iterate           = BOOST_BINARY_U(0010011)  
};

}}

#endif // HPX_60B7914E_21A5_4977_AA9C_8E66C44EE0FB

