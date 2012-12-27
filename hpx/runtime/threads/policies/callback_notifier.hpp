//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_POLICIES_CALLBACK_NOTIFIER_JUN_18_2009_1132AM)
#define HPX_THREADMANAGER_POLICIES_CALLBACK_NOTIFIER_JUN_18_2009_1132AM

#include <hpx/hpx_fwd.hpp>
#include <boost/exception_ptr.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    class callback_notifier
    {
        typedef HPX_STD_FUNCTION<void(std::size_t, char const*)> on_startstop_type;
        typedef HPX_STD_FUNCTION<void(std::size_t, boost::exception_ptr const&)>
            on_error_type;

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
        void on_error(std::size_t num_thread, boost::exception_ptr const& e)
        {
            if (on_error_)
                on_error_(num_thread, e);
        }

    private:
        on_startstop_type on_start_thread_;    ///< function to call for each created thread
        on_startstop_type on_stop_thread_;     ///< function to call in case of unexpected stop
        on_error_type on_error_;               ///< function to call in case of error
    };

}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
