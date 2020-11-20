//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/gid_type.hpp>
#include <hpx/naming_base/naming_base.hpp>

#include <cstddef>

namespace hpx { namespace applier {
    // helper functions allowing to bind and unbind a GID to a given address
    // without having to directly refer to the resolver_client
    HPX_EXPORT bool bind_gid_local(naming::gid_type const&,
        naming::address const&, error_code& ec = throws);
    HPX_EXPORT void unbind_gid_local(
        naming::gid_type const&, error_code& ec = throws);

    HPX_EXPORT bool bind_range_local(naming::gid_type const&, std::size_t,
        naming::address const&, std::size_t, error_code& ec = throws);
    HPX_EXPORT void unbind_range_local(
        naming::gid_type const&, std::size_t, error_code& ec = throws);
}}    // namespace hpx::applier
