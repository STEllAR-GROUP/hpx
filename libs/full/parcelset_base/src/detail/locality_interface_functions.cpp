//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/parcelset_base/detail/locality_interface_functions.hpp>
#include <hpx/parcelset_base/locality.hpp>

#include <cstddef>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parcelset::detail {

    parcelset::parcel (*create_parcel)() = nullptr;

    locality (*create_locality)(std::string const& name) = nullptr;

    policies::message_handler* (*get_message_handler)(char const* action,
        char const* type, std::size_t num, std::size_t interval,
        locality const& loc, error_code& ec) = nullptr;
}    // namespace hpx::parcelset::detail

#endif
