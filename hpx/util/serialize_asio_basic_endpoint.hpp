////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_3058286E_3A0C_4B95_943A_0492450ABEE8)
#define HPX_3058286E_3A0C_4B95_943A_0492450ABEE8

#include <boost/asio/ip/basic_endpoint.hpp>

#include <hpx/util/serialize_asio_address.hpp>

namespace boost { namespace serialization
{

template <typename Archive, typename InternetProtocol>
void save(Archive& ar, asio::ip::basic_endpoint<InternetProtocol> const& v,
          unsigned int)
{
    // serialization, annoyingly, won't take const references for save()
    boost::uint16_t port = v.port();
    asio::ip::address addr = v.address();
    ar & port & addr;
}

template <typename Archive, typename InternetProtocol>
void load(Archive& ar, asio::ip::basic_endpoint<InternetProtocol>& v,
          unsigned int)
{
    boost::uint16_t port;
    asio::ip::address addr;
    ar & port & addr;
    v.port(port);
    v.address(addr); 
}

template <typename Archive, typename InternetProtocol>
void serialize(Archive& ar, asio::ip::basic_endpoint<InternetProtocol>& v,
               unsigned int version)
{ split_free(ar, v, version); }

}}

#endif // HPX_3058286E_3A0C_4B95_943A_0492450ABEE8

