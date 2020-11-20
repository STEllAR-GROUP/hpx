////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/trigger_lco.hpp>

#include <exception>
#include <utility>

namespace hpx { namespace applier {

    template <typename Arg0>
    inline void trigger(naming::id_type const& k, Arg0&& arg0)
    {
        set_lco_value(k, std::forward<Arg0>(arg0));
    }

    inline void trigger(naming::id_type const& k)
    {
        trigger_lco_event(k);
    }

    inline void trigger_error(
        naming::id_type const& k, std::exception_ptr const& e)
    {
        set_lco_error(k, e);
    }

    inline void trigger_error(naming::id_type const& k, std::exception_ptr&& e)
    {
        set_lco_error(k, e);
    }

}}    // namespace hpx::applier
