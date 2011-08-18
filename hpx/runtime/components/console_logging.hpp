//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_LOGGING_DEC_16_2008_0435PM)
#define HPX_COMPONENTS_CONSOLE_LOGGING_DEC_16_2008_0435PM

#include <boost/atomic.hpp>
#include <boost/foreach.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/lcos/mutex.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/components/server/console_logging.hpp>
#include <hpx/util/static.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    struct HPX_EXPORT pending_logs
    {
        typedef lcos::mutex prefix_mutex_type;
        typedef util::spinlock queue_mutex_type;
 
        enum { max_pending = 128 };

        pending_logs()
            : prefix_(naming::invalid_id), activated_(false), queue_size_(0) {}

        void add(message_type const& msg);

        void cleanup();

        void activate()
        {
            activated_.store(true);
        }

      private:
        void send();

        prefix_mutex_type prefix_mtx_;
        naming::id_type prefix_;

        queue_mutex_type queue_mtx_;
        messages_type queue_;

        boost::atomic<bool> activated_;

        boost::atomic<std::size_t> queue_size_;
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

#endif
