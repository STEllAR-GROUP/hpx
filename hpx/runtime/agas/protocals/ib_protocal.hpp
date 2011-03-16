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

#include <hpx/runtime/agas/local_address.hpp>
#include <hpx/runtime/agas/namespaces/primary.hpp>
#include <hpx/runtime/agas/namespaces/locality.hpp>

namespace hpx { namespace agas // hpx::agas
{

namespace tag { struct ib_protocal; }

namespace traits // hpx::agas::traits
{

template <>
struct protocal_name_hook<tag::ib_protocal>
{
    typedef std::string result_type;

    static result_type call()
    { return "IB"; }
};

template <>
struct locality_type<tag::ib_protocal>
{ typedef boost::asio::ip::ib::endpoint type; };

} // hpx::agas::traits

typedef local_address<tag::ib_protocal> ib_local_address;

} // hpx::agas

///////////////////////////////////////////////////////////////////////////////
namespace components { namespace agas // hpx::components::agas
{

typedef primary_namespace_type<hpx::agas::tag::ib_protocal>::type
    ib_primary_namespace;
typedef locality_namespace_type<hpx::agas::tag::ib_protocal>::type
    ib_locality_namespace;

///////////////////////////////////////////////////////////////////////////////
namespace server // hpx::components::agas::server
{

typedef primary_namespace_type<hpx::agas::tag::ib_protocal>::type
    ib_primary_namespace;
typedef locality_namespace_type<hpx::agas::tag::ib_protocal>::type
    ib_locality_namespace;

} // hpx::components::agas::stubs

///////////////////////////////////////////////////////////////////////////////
namespace stubs // hpx::components::agas::stubs
{

typedef primary_namespace_type<hpx::agas::tag::ib_protocal>::type
    ib_primary_namespace;
typedef locality_namespace_type<hpx::agas::tag::ib_protocal>::type
    ib_locality_namespace;

} // hpx::components::agas::stubs

} // hpx::components::agas
} // hpx::components
} // hpx

///////////////////////////////////////////////////////////////////////////////
BOOST_CLASS_VERSION(
    hpx::agas::ib_local_address, 
    hpx::agas::traits::serialization_version<
        hpx::agas::tag::ib_protocal
    >::value)
BOOST_CLASS_TRACKING(
    hpx::agas::ib_local_address, boost::serialization::track_never)

#else
  #warning HPX_USE_ASIO_IB_SUPPORT is not defined, IB protocal disabled.
#endif

#endif // HPX_6A8F4842_FB5A_4B32_8D20_56368768A707

