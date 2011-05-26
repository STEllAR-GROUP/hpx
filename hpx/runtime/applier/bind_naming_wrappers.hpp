////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_A21C8E6C_F75C_4F4D_AD85_8847E40E78AB)
#define HPX_A21C8E6C_F75C_4F4D_AD85_8847E40E78AB

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>

namespace hpx { namespace applier
{
    // helper functions allowing to bind and unbind a GID to a given address
    // without having to directly refer to the resolver_client
    bool HPX_EXPORT bind_gid(naming::gid_type const& gid_,
        naming::address const& addr, error_code& ec = throws);

    void HPX_EXPORT unbind_gid(naming::gid_type const& gid_,
        error_code& ec = throws);

    bool HPX_EXPORT bind_range(naming::gid_type const& gid, std::size_t count, 
        naming::address const& addr, std::size_t offset, error_code& ec = throws);

    void HPX_EXPORT unbind_range(naming::gid_type const& gid, std::size_t count, 
        error_code& ec = throws);
}}


#endif // HPX_A21C8E6C_F75C_4F4D_AD85_8847E40E78AB

