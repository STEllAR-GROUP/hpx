//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_distributed/trigger_lco.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <utility>

namespace hpx { namespace actions {

    ///////////////////////////////////////////////////////////////////////////
    struct set_lco_value_continuation
    {
        template <typename T>
        HPX_FORCEINLINE T operator()(hpx::id_type const& lco, T&& t) const
        {
            hpx::set_lco_value(lco, HPX_FORWARD(T, t));

            // Yep, 't' is a zombie, however we don't use the returned value
            // anyways. We need it for result type calculation, though.
            return HPX_FORWARD(T, t);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct set_lco_value_unmanaged_continuation
    {
        template <typename T>
        HPX_FORCEINLINE T operator()(hpx::id_type const& lco, T&& t) const
        {
            hpx::set_lco_value_unmanaged(lco, HPX_FORWARD(T, t));

            // Yep, 't' is a zombie, however we don't use the returned value
            // anyways. We need it for result type calculation, though.
            return HPX_FORWARD(T, t);
        }
    };
}}    // namespace hpx::actions
