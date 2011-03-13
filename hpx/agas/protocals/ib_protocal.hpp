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

#include <hpx/agas/traits.hpp>

namespace hpx { namespace agas // hpx::agas
{

namespace tag { struct ib_protocal; }

namespace traits { // hpx::agas::traits

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

} // hpx::agas
} // hpx

///////////////////////////////////////////////////////////////////////////////
namespace components { namespace agas // hpx::components::agas
{

typedef primary_namespace_type<hpx::agas::tag::ib_protocal>::type
    ib_primary_namespace;

///////////////////////////////////////////////////////////////////////////////
namespace server // hpx::components::agas::server
{

typedef server::primary_namespace_type<hpx::agas::tag::ib_protocal>::type
    ib_primary_namespace;

} // hpx::components::agas::stubs

///////////////////////////////////////////////////////////////////////////////
namespace stubs // hpx::components::agas::stubs
{

typedef stubs::primary_namespace_type<hpx::agas::tag::ib_protocal>::type
    ib_primary_namespace;

} // hpx::components::agas::stubs

} // hpx::components::agas
} // hpx::components
} // hpx

#else
  #warning HPX_USE_ASIO_IB_SUPPORT is not defined, IB protocal disabled.
#endif

#endif // HPX_6A8F4842_FB5A_4B32_8D20_56368768A707

