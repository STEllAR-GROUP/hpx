//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/modules/errors.hpp>

#include <hpx/parcelset_base/detail/locality_interface_functions.hpp>
#include <hpx/parcelset_base/locality.hpp>
#include <hpx/parcelset_base/locality_interface.hpp>
#include <hpx/parcelset_base/parcel_interface.hpp>
#include <hpx/parcelset_base/policies/message_handler.hpp>

#include <cstddef>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parcelset {

    parcelset::parcel create_parcel()
    {
        return detail::create_parcel();
    }

    locality create_locality(std::string const& name)
    {
        return detail::create_locality(name);
    }

    policies::message_handler* get_message_handler(char const* action,
        char const* type, std::size_t num, std::size_t interval,
        locality const& loc, error_code& ec)
    {
        return detail::get_message_handler(
            action, type, num, interval, loc, ec);
    }
}    // namespace hpx::parcelset

#endif
