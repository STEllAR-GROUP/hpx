//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_ERROR_SINK_SINGLETON_JAN_26_2009_0503PM)
#define HPX_COMPONENTS_CONSOLE_ERROR_SINK_SINGLETON_JAN_26_2009_0503PM

#include <string>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/static.hpp>

#include <boost/cstdint.hpp>
#include <boost/signals.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/thread.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    class console_error_dispatcher
    {
    private:
        typedef void dispatcher_type(boost::uint32_t, boost::exception_ptr const&);
        typedef boost::mutex mutex_type;

    public:
        typedef boost::signals::scoped_connection scoped_connection_type;

        console_error_dispatcher() {}
        ~console_error_dispatcher() {}

        template <typename F, typename Connection>
        bool register_error_sink(F sink, Connection& conn)
        {
            return (conn = dispatcher_.connect(sink)).connected();
        }

        void operator()(boost::uint32_t src, boost::exception_ptr const& e)
        {
            dispatcher_(src, e);
        }

    private:
        boost::signal<dispatcher_type> dispatcher_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct error_dispatcher_tag {};

    inline console_error_dispatcher& get_error_dispatcher()
    {
        util::static_<console_error_dispatcher, error_dispatcher_tag> disp;
        return disp.get();
    }

}}}

#endif
