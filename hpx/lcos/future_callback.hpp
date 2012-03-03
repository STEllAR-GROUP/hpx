//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_CALLBACK_DEC_06_2008_1126AM)
#define HPX_LCOS_FUTURE_CALLBACK_DEC_06_2008_1126AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>

#include <boost/bind.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    class future_callback
    {
    public:
        typedef typename Future::result_type result_type;
        typedef HPX_STD_FUNCTION<void(result_type const&)> callback_type;

        future_callback(Future const& future, callback_type cb)
        {
            hpx::applier::register_work(
                boost::bind(&future_callback::wait_for_future, future, cb),
                "future_callback");
        }

    private:
        // thread function for thread waiting on future
        static threads::thread_state
            wait_for_future(Future future, callback_type cb)
        {
            cb(future.get());
            return threads::terminated;
        }
    };

}}

#endif
