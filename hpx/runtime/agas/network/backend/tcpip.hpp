////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_3E01AB13_F49A_4A38_9B19_E18B11F4102D)
#define HPX_3E01AB13_F49A_4A38_9B19_E18B11F4102D

#include <boost/asio/ip/tcp.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>

#include <hpx/runtime/agas/network/gva.hpp>
#include <hpx/util/serialize_asio_basic_endpoint.hpp>

namespace hpx { namespace agas 
{

namespace tag { namespace network { struct tcpip; }} 

namespace traits { namespace network
{

template <>
struct endpoint_type<tag::network::tcpip>
{ typedef boost::asio::ip::tcp::endpoint type; };

template <>
struct name_hook<tag::network::tcpip>
{
    typedef char const* result_type;

    static result_type call()
    { return "tcpip"; }
};

}}}}

BOOST_CLASS_VERSION(
    hpx::agas::gva<hpx::agas::tag::network::tcpip>, 
    hpx::agas::traits::serialization_version<
        hpx::agas::tag::network::tcpip
    >::value)
BOOST_CLASS_TRACKING(
    hpx::agas::gva<hpx::agas::tag::network::tcpip>,
    boost::serialization::track_never)

#endif // HPX_3E01AB13_F49A_4A38_9B19_E18B11F4102D

