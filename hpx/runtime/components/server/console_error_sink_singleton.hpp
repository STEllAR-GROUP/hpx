//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_ERROR_SINK_SINGLETON_JAN_26_2009_0503PM)
#define HPX_COMPONENTS_CONSOLE_ERROR_SINK_SINGLETON_JAN_26_2009_0503PM

#include <string>

#include <hpx/hpx_fwd.hpp>

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <boost/signals.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    class console_error_dispatcher : boost::noncopyable
    {
    private:
        typedef void dispatcher_type(boost::uint32_t, std::string const&);

    public:
        typedef boost::signals::scoped_connection scoped_connection_type;

        template <typename F, typename Connection>
        bool register_error_sink(F sink, Connection& conn)
        {
            return (conn = dispatcher_.connect(sink)).connected();
        }

        void operator()(boost::uint32_t src, std::string const& msg)
        {
            dispatcher_(src, msg);
        }

    private:
        boost::signal<dispatcher_type> dispatcher_;
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT console_error_dispatcher& get_error_dispatcher();

}}}

#endif
