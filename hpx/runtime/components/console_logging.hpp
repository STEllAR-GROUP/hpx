//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_LOGGING_DEC_16_2008_0435PM)
#define HPX_COMPONENTS_CONSOLE_LOGGING_DEC_16_2008_0435PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/components/server/console_logging.hpp>
#include <hpx/util/static.hpp>

#include <boost/foreach.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    inline void console_logging_locked(naming::id_type const& prefix, 
        server::logging_destination dest, int level, std::string const& msg)
    {
        if (NULL != applier::get_applier_ptr()) 
            applier::apply<server::console_logging_action>(prefix, dest, level, msg);
    }

    struct pending_logs
    {
        typedef boost::recursive_mutex mutex_type;    // use boost::mutex for now

        pending_logs() : has_pending_(false), sending_logs_(false) {}

        void handle_pending(naming::id_type const& prefix, 
            server::logging_destination dest, int level)
        {
            if (has_pending_) {
                // log all pending messages first
                mutex_type::scoped_lock l(mtx_, boost::defer_lock);
                if (l.try_lock()) {
                    BOOST_FOREACH(std::string const& s, pending_) 
                        console_logging_locked(prefix, dest, level, s);
                    std::vector<std::string>().swap(pending_);
                    has_pending_ = false;
                }
            }
        }

        void add_pending(std::string const& msg)
        {
            mutex_type::scoped_lock l(mtx_, boost::defer_lock);
            if (l.try_lock()) {
                pending_.push_back(msg);
                has_pending_ = true;
            }
        }

        mutex_type mtx_;
        std::vector<std::string> pending_;
        bool has_pending_;
        bool sending_logs_;
    };
    struct pending_logs_tag {};

    struct reset_on_exit
    {
        reset_on_exit(bool& flag, bool new_value)
          : flag_(flag), oldval_(flag)
        {
            flag_ = new_value;
        }
        ~reset_on_exit()
        {
            flag_ = oldval_;
        }

        bool& flag_;
        bool oldval_;
    };

    // stub function allowing to apply the console_logging action
    void console_logging(naming::id_type const& prefix, 
        server::logging_destination dest, int level, std::string const& msg)
    {
        // do logging only if applier is still valid
        util::static_<pending_logs, pending_logs_tag> logs;
        if (NULL != applier::get_applier_ptr()) {
            if (logs.get().sending_logs_) {
                logs.get().add_pending(msg);
            }
            else {
                reset_on_exit exit(logs.get().sending_logs_, true);
                logs.get().handle_pending(prefix, dest, level);
                console_logging_locked(prefix, dest, level, msg);
            }
        }
        else {
            logs.get().add_pending(msg);
        }
    }

    // special initialization functions for console logging 
    namespace detail
    {
        void init_agas_console_log(util::section const& ini);
        void init_timing_console_log(util::section const& ini);
        void init_hpx_console_log(util::section const& ini);
    }

}}

#endif
