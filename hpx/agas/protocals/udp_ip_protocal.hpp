////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_343471B7_B8A6_4DC1_8D83_FB9F754ED473)
#define HPX_343471B7_B8A6_4DC1_8D83_FB9F754ED473

#include <boost/asio/ip/udp.hpp>

#include <hpx/agas/magic.hpp>

namespace hpx { namespace agas // hpx::agas
{

namespace tag { struct udp_ip_protocal; }

namespace magic { // hpx::agas::magic

template <>
struct protocal_name_hook<tag::udp_ip_protocal>
{
    typedef std::string result_type;

    static result_type call()
    { return "UDP/IP"; }
};

template <>
struct locality_type<tag::udp_ip_protocal>
{ typedef boost::asio::ip::udp::endpoint type; };

} // hpx::agas
} // hpx

///////////////////////////////////////////////////////////////////////////////
namespace components { namespace agas // hpx::components::agas
{

typedef primary_namespace_type<hpx::agas::tag::udp_ip_protocal>::type
    udp_ip_primary_namespace;

///////////////////////////////////////////////////////////////////////////////
namespace server // hpx::components::agas::server
{

typedef server::primary_namespace_type<hpx::agas::tag::udp_ip_protocal>::type
    udp_ip_primary_namespace;

} // hpx::components::agas::stubs

///////////////////////////////////////////////////////////////////////////////
namespace stubs // hpx::components::agas::stubs
{

typedef stubs::primary_namespace_type<hpx::agas::tag::udp_ip_protocal>::type
    udp_ip_primary_namespace;

} // hpx::components::agas::stubs

} // hpx::components::agas
} // hpx::components
} // hpx

#endif // HPX_343471B7_B8A6_4DC1_8D83_FB9F754ED473

