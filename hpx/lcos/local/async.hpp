//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_ASYNC_MAR_10_2012_1256PM)
#define HPX_LCOS_LOCAL_ASYNC_MAR_10_2012_1256PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/future.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename F>
    future<Result> async (F const& f)
    {
        return local::packaged_task<Result>(f).get_future();
    }

    template <typename Result, typename F>
    future<Result> async (BOOST_FWD_REF(F) f)
    {
        return local::packaged_task<Result>(boost::forward<F>(f)).get_future();
    }

//    ///////////////////////////////////////////////////////////////////////////
//    template <typename Result, typename F>
//    future<Result> async_callback (
//        HPX_STD_FUNCTION<void(future<Result>)> const& data_sink, BOOST_FWD_REF(F) f)
//    {
//        return local::packaged_task<Result>(
//            data_sink, boost::forward<F>(f)).get_future();
//    }
}}}

#endif
