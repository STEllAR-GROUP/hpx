////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_8B1A4443_7D95_4C0D_9970_7CEA4D049608)
#define HPX_8B1A4443_7D95_4C0D_9970_7CEA4D049608

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/performance_counters/counters.hpp>

namespace hpx { namespace performance_counters
{
    struct manage_counter
    {
        manage_counter() : counter_(naming::invalid_id) {}

        ~manage_counter()
        {
            uninstall();
        }

        // install an (existing) counter
        HPX_EXPORT counter_status install(naming::id_type const& id,
            counter_info const& info, error_code& ec = throws);

        // uninstall the counter
        HPX_EXPORT void uninstall();

    private:
        counter_info info_;
        naming::id_type counter_;
    };

    /// Install a new performance counter in a way, which will uninstall it
    /// automatically during shutdown.
    HPX_EXPORT void install_counter(naming::id_type const& id,
        counter_info const& info, error_code& ec = throws);
}}

#endif // HPX_8B1A4443_7D95_4C0D_9970_7CEA4D049608

