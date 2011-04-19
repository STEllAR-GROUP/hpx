////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_343471B7_B8A6_4DC1_8D83_FB9F754ED473)
#define HPX_343471B7_B8A6_4DC1_8D83_FB9F754ED473

#include <boost/asio/ip/udp.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>

#include <hpx/runtime/agas/network/gva.hpp>
#include <hpx/util/serialize_asio_basic_endpoint.hpp>

namespace hpx { namespace agas 
{

namespace tag { namespace network { struct udpip; }} 

namespace traits { namespace network
{

template <>
struct endpoint_type<tag::network::udpip>
{ typedef boost::asio::ip::udp::endpoint type; };

template <>
struct name_hook<tag::network::udpip>
{
    typedef char const* result_type;

    static result_type call()
    { return "udpip"; }
};

}}}}

BOOST_CLASS_VERSION(
    hpx::agas::full_gva<hpx::agas::tag::network::udpip>, 
    hpx::agas::traits::serialization_version<
        hpx::agas::tag::network::udpip
    >::value)
BOOST_CLASS_TRACKING(
    hpx::agas::full_gva<hpx::agas::tag::network::udpip>,
    boost::serialization::track_never)

BOOST_CLASS_VERSION(
    hpx::agas::gva<hpx::agas::tag::network::udpip>, 
    hpx::agas::traits::serialization_version<
        hpx::agas::tag::network::udpip
    >::value)
BOOST_CLASS_TRACKING(
    hpx::agas::gva<hpx::agas::tag::network::udpip>,
    boost::serialization::track_never)

#endif // HPX_343471B7_B8A6_4DC1_8D83_FB9F754ED473

