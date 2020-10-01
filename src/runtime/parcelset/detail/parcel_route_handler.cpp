//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/modules/errors.hpp>
#include <hpx/runtime/parcelset/detail/parcel_route_handler.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/runtime_distributed.hpp>

#include <system_error>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    void dijkstra_make_black();     // forward declaration only
}}

namespace hpx { namespace parcelset { namespace detail
{
    void parcel_route_handler(
        std::error_code const& ec,
        parcelset::parcel const& p)
    {
        parcelhandler& ph = hpx::get_runtime_distributed().get_parcel_handler();
        // invoke the original handler
        ph.invoke_write_handler(ec, p);

        // inform termination detection of a sent message
        if (!p.does_termination_detection())
            hpx::detail::dijkstra_make_black();
    }
}}}

#endif
