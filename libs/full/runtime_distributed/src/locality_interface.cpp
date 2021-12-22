//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/assert.hpp>

#include <hpx/parcelset/parcel.hpp>
#include <hpx/parcelset/parcelhandler.hpp>
#include <hpx/parcelset_base/detail/locality_interface_functions.hpp>
#include <hpx/parcelset_base/locality.hpp>
#include <hpx/runtime_distributed.hpp>

#include <cstddef>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parcelset {

    namespace detail::impl {

        parcelset::parcel create_parcel()
        {
            return parcelset::parcel(new detail::parcel());
        }

        locality create_locality(std::string const& name)
        {
            HPX_ASSERT(get_runtime_ptr());
            return get_runtime_distributed()
                .get_parcel_handler()
                .create_locality(name);
        }

        policies::message_handler* get_message_handler(char const* action,
            char const* type, std::size_t num, std::size_t interval,
            locality const& loc, error_code& ec)
        {
            return get_runtime_distributed()
                .get_parcel_handler()
                .get_message_handler(action, type, num, interval, loc, ec);
        }
    }    // namespace detail::impl

    // initialize locality interface function pointers in naming_base module
    struct HPX_EXPORT locality_interface_functions
    {
        locality_interface_functions()
        {
            detail::create_parcel = &detail::impl::create_parcel;
            detail::create_locality = &detail::impl::create_locality;
            detail::get_message_handler = &detail::impl::get_message_handler;
        }
    };

    locality_interface_functions locality_init;
}    // namespace hpx::parcelset

#endif
