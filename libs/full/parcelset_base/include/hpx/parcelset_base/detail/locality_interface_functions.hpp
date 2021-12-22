//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/modules/errors.hpp>

#include <hpx/parcelset_base/parcelset_base_fwd.hpp>
#include <hpx/parcelset_base/policies/message_handler.hpp>

#include <cstddef>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parcelset::detail {

    extern HPX_EXPORT parcelset::parcel (*create_parcel)();

    extern HPX_EXPORT locality (*create_locality)(std::string const& name);

    extern HPX_EXPORT policies::message_handler* (*get_message_handler)(
        char const* action, char const* type, std::size_t num,
        std::size_t interval, locality const& loc, error_code& ec);
}    // namespace hpx::parcelset::detail

#endif
