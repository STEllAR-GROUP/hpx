//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_ERROR_SINK_SINGLETON_JAN_26_2009_0503PM)
#define HPX_COMPONENTS_CONSOLE_ERROR_SINK_SINGLETON_JAN_26_2009_0503PM

#include <hpx/config.hpp>
#include <hpx/util/spinlock.hpp>

#include <boost/signals2.hpp>

#include <mutex>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class console_error_dispatcher
    {
    public:
        HPX_NON_COPYABLE(console_error_dispatcher);

    public:
        typedef util::spinlock mutex_type;
        typedef void dispatcher_type(std::string const&);

        typedef boost::signals2::scoped_connection scoped_connection_type;

        console_error_dispatcher()
          : mtx_()
        {}

        template <typename F, typename Connection>
        bool register_error_sink(F sink, Connection& conn)
        {
            std::lock_guard<mutex_type> l(mtx_);
            return (conn = dispatcher_.connect(sink)).connected();
        }

        void operator()(std::string const& msg)
        {
            std::lock_guard<mutex_type> l(mtx_);
            dispatcher_(msg);
        }

    private:
        mutex_type mtx_;
        boost::signals2::signal<dispatcher_type> dispatcher_;
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT console_error_dispatcher& get_error_dispatcher();
}}}

#endif

