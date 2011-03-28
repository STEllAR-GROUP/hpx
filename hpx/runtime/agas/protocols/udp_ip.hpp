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

#include <hpx/runtime/agas/local_address.hpp>
#include <hpx/runtime/agas/namespaces/primary.hpp>
#include <hpx/runtime/agas/namespaces/locality.hpp>

namespace hpx { namespace agas // hpx::agas
{

namespace tag { struct udp_ip_protocol; }

namespace traits // hpx::agas::traits
{

template <>
struct protocol_name_hook<tag::udp_ip_protocol>
{
    typedef char const* result_type;

    static result_type call()
    { return "udp_ip"; }
};

template <>
struct locality_type<tag::udp_ip_protocol>
{ typedef boost::asio::ip::udp::endpoint type; };

} // hpx::agas::traits

typedef local_address<tag::udp_ip_protocol> udp_ip_local_address;

} // hpx::agas

///////////////////////////////////////////////////////////////////////////////
namespace components { namespace agas // hpx::components::agas
{

typedef primary_namespace_type<hpx::agas::tag::udp_ip_protocol>::type
    udp_ip_primary_namespace;
typedef locality_namespace_type<hpx::agas::tag::udp_ip_protocol>::type
    udp_ip_locality_namespace;

///////////////////////////////////////////////////////////////////////////////
namespace server // hpx::components::agas::server
{

typedef primary_namespace_type<hpx::agas::tag::udp_ip_protocol>::type
    udp_ip_primary_namespace;
typedef locality_namespace_type<hpx::agas::tag::udp_ip_protocol>::type
    udp_ip_locality_namespace;

} // hpx::components::agas::stubs

///////////////////////////////////////////////////////////////////////////////
namespace stubs // hpx::components::agas::stubs
{

typedef primary_namespace_type<hpx::agas::tag::udp_ip_protocol>::type
    udp_ip_primary_namespace;
typedef locality_namespace_type<hpx::agas::tag::udp_ip_protocol>::type
    udp_ip_locality_namespace;

} // hpx::components::agas::stubs

} // hpx::components::agas
} // hpx::components
} // hpx

///////////////////////////////////////////////////////////////////////////////
BOOST_CLASS_VERSION(
    hpx::agas::udp_ip_local_address,
    hpx::agas::traits::serialization_version<
        hpx::agas::tag::udp_ip_protocol
    >::value)
BOOST_CLASS_TRACKING(
    hpx::agas::udp_ip_local_address, boost::serialization::track_never)

#endif // HPX_343471B7_B8A6_4DC1_8D83_FB9F754ED473

