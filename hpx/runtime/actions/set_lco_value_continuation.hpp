//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_ACTIONS_SET_LCO_VALUE_CONTINUATION_HPP
#define HPX_ACTIONS_SET_LCO_VALUE_CONTINUATION_HPP

#include <hpx/config.hpp>

#include <hpx/runtime/trigger_lco.hpp>
#include <hpx/runtime/naming/id_type.hpp>

#include <utility>

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    struct set_lco_value_continuation
    {
        template <typename T>
        HPX_FORCEINLINE T operator()(naming::id_type const& lco, T && t) const
        {
            hpx::set_lco_value(lco, std::forward<T>(t));

            // Yep, 't' is a zombie, however we don't use the returned value
            // anyways. We need it for result type calculation, though.
            return std::move(t);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct set_lco_value_simple_unmanaged_continuation
    {
        template <typename T>
        HPX_FORCEINLINE T operator()(naming::id_type const& lco, T&& t) const
        {
            hpx::set_lco_value_simple_unmanaged(lco, std::forward<T>(t));

            // Yep, 't' is a zombie, however we don't use the returned value
            // anyways. We need it for result type calculation, though.
            return std::move(t);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct set_lco_value_unmanaged_continuation
    {
        template <typename T>
        HPX_FORCEINLINE T operator()(naming::id_type const& lco, T && t) const
        {
            hpx::set_lco_value_unmanaged(lco, std::forward<T>(t));

            // Yep, 't' is a zombie, however we don't use the returned value
            // anyways. We need it for result type calculation, though.
            return std::move(t);
        }
    };
}}

#endif
