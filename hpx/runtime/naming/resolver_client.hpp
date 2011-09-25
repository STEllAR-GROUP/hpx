//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_CLIENT_RESOLVER_MAR_24_2008_0952AM)
#define HPX_NAMING_CLIENT_RESOLVER_MAR_24_2008_0952AM

#include <hpx/version.hpp>
#include <hpx/runtime/agas/router/legacy.hpp>

namespace hpx { namespace naming
{

class HPX_EXPORT resolver_client : public hpx::agas::legacy_router
{
public:
    typedef hpx::agas::legacy_router base_type;

    resolver_client(
        parcelset::parcelport& pp 
      , util::runtime_configuration const& ini_ 
      , runtime_mode mode
        )
      : base_type(pp, ini_, mode) {} 
};

}}  // namespace hpx::naming

#endif
