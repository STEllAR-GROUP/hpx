//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_POLICIES_CALLBACK_NOTIFIER_JUN_18_2009_1132AM)
#define HPX_THREADMANAGER_POLICIES_CALLBACK_NOTIFIER_JUN_18_2009_1132AM

#include <hpx/config.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/util/function.hpp>

#include <cstddef>
#include <exception>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    class callback_notifier
    {
        typedef util::function_nonser<
            void(std::size_t, char const*)> on_startstop_type;
        typedef util::function_nonser<
            void(std::size_t, std::exception_ptr const&)> on_error_type;

    public:
        callback_notifier(on_startstop_type start = on_startstop_type(),
            on_startstop_type stop = on_startstop_type(),
            on_error_type on_err = on_error_type())
          : on_start_thread_(start), on_stop_thread_(stop), on_error_(on_err)
        {}

        void on_start_thread(std::size_t num_thread)
        {
            if (on_start_thread_)
                on_start_thread_(num_thread, "");
        }
        void on_stop_thread(std::size_t num_thread)
        {
            if (on_stop_thread_)
                on_stop_thread_(num_thread, "");
        }
        void on_error(std::size_t num_thread, std::exception_ptr const& e)
        {
            if (on_error_)
                on_error_(num_thread, e);
        }

        // function to call for each created thread
        on_startstop_type on_start_thread_;
        // function to call in case of unexpected stop
        on_startstop_type on_stop_thread_;
        // function to call in case of error
        on_error_type on_error_;
    };

}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
