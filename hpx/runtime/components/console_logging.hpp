//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/spinlock.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/components/server/console_logging.hpp>
#include <hpx/state.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/type_support/static.hpp>

#include <atomic>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    struct HPX_EXPORT pending_logs
    {
        typedef lcos::local::mutex prefix_mutex_type;
        typedef util::spinlock queue_mutex_type;

        enum { max_pending = 128 };

        pending_logs()
          : prefix_mtx_()
          , prefix_(naming::invalid_id)
          , queue_mtx_()
          , activated_(false)
          , is_sending_(false)
        {}

        void add(message_type const& msg);

        void cleanup();

        void activate()
        {
            activated_.store(true);
        }

    private:
        bool ensure_prefix();
        void send();
        bool is_active();

        prefix_mutex_type prefix_mtx_;
        naming::id_type prefix_;

        queue_mutex_type queue_mtx_;
        messages_type queue_;

        std::atomic<bool> activated_;
        std::atomic<bool> is_sending_;
    };

    struct pending_logs_tag {};

    // special initialization functions for console logging
    namespace detail
    {
        void init_agas_console_log(util::section const& ini);
        void init_timing_console_log(util::section const& ini);
        void init_hpx_console_log(util::section const& ini);
    }
}}

#include <hpx/config/warnings_suffix.hpp>

