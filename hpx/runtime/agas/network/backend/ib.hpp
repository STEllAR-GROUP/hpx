////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_6A8F4842_FB5A_4B32_8D20_56368768A707)
#define HPX_6A8F4842_FB5A_4B32_8D20_56368768A707

#if defined(HPX_USE_ASIO_IB_SUPPORT)
#include <boost/asio/ip/ib.hpp> // doesn't exist (yet)
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>

#include <hpx/runtime/agas/network/gva.hpp>

namespace hpx { namespace agas 
{

namespace tag { namespace network { struct ib; }} 

namespace traits { namespace network
{

template <>
struct endpoint_type<tag::network::ib>
{ typedef boost::asio::ib::endpoint type; };

template <>
struct name_hook<tag::network::ib>
{
    typedef char const* result_type;

    static result_type call()
    { return "infiniband"; }
};

}}}}

BOOST_CLASS_VERSION(
    hpx::agas::full_gva<hpx::agas::tag::network::ib>, 
    hpx::agas::traits::serialization_version<
        hpx::agas::tag::network::ib
    >::value)
BOOST_CLASS_TRACKING(
    hpx::agas::full_gva<hpx::agas::tag::network::ib>,
    boost::serialization::track_never)

BOOST_CLASS_VERSION(
    hpx::agas::gva<hpx::agas::tag::network::ib>, 
    hpx::agas::traits::serialization_version<
        hpx::agas::tag::network::ib
    >::value)
BOOST_CLASS_TRACKING(
    hpx::agas::gva<hpx::agas::tag::network::ib>,
    boost::serialization::track_never)

#else
  #warning HPX_USE_ASIO_IB_SUPPORT is not defined, IB protocol disabled.
#endif

#endif // HPX_6A8F4842_FB5A_4B32_8D20_56368768A707

