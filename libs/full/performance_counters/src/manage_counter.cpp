//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/actions/continuation.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/manage_counter.hpp>
#include <hpx/version.hpp>

#include <memory>

namespace hpx { namespace performance_counters {
    struct manage_counter
    {
        manage_counter()
          : counter_(naming::invalid_id)
        {
        }

        ~manage_counter()
        {
            uninstall();
        }

        // install an (existing) counter
        counter_status install(naming::id_type const& id,
            counter_info const& info, error_code& ec = throws);

        // uninstall the counter
        void uninstall();

    private:
        counter_info info_;
        naming::id_type counter_;
    };

    counter_status manage_counter::install(
        naming::id_type const& id, counter_info const& info, error_code& ec)
    {
        if (counter_ != naming::invalid_id)
        {
            HPX_THROWS_IF(ec, hpx::invalid_status, "manage_counter::install",
                "counter has been already installed");
            return status_invalid_data;
        }

        info_ = info;
        counter_ = id;

        return detail::add_counter(id, info_, ec);
    }

    void manage_counter::uninstall()
    {
        if (counter_)
        {
            error_code ec(lightweight);
            detail::remove_counter(info_, counter_, ec);
            counter_ = naming::invalid_id;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    static void counter_shutdown(std::shared_ptr<manage_counter> const& p)
    {
        HPX_ASSERT(p);
        p->uninstall();
    }

    void install_counter(
        naming::id_type const& id, counter_info const& info, error_code& ec)
    {
        std::shared_ptr<manage_counter> p = std::make_shared<manage_counter>();

        // Install the counter instance.
        p->install(id, info, ec);

        // Register the shutdown function which will clean up this counter.
        get_runtime().add_shutdown_function(
            util::bind_front(&counter_shutdown, p));
    }
}}    // namespace hpx::performance_counters
