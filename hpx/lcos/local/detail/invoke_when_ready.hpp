//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013-2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_INVOKE_WHEN_READY_APR_28_2014_0405PM)
#define HPX_LCOS_INVOKE_WHEN_READY_APR_28_2014_0405PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/traits/serialize_as_future.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/atomic.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/end.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/not.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/utility/enable_if.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Args>
    struct when_ready //-V690
      : lcos::detail::future_data<
            typename util::invoke_fused_result_of<F&(Args)>::type
        >
    {
        typedef typename util::invoke_fused_result_of<F&(Args)>::type result_type;
        typedef lcos::detail::future_data<result_type> base_type;

    private:
        // workaround gcc regression wrongly instantiating constructors
        when_ready();
        when_ready(when_ready const&);

    public:
        typedef hpx::future<result_type> type;

        template <typename ...Ts>
        when_ready(F const& f, Ts&&... vs)
          : f_(f)
          , args_(std::forward<Ts>(vs)...)
        {}

        template <typename ...Ts>
        when_ready(F&& f, Ts&&... vs)
          : f_(std::move(f))
          , args_(std::forward<Ts>(vs)...)
        {}

        BOOST_FORCEINLINE
        void invoke(boost::mpl::false_)
        {
            try {
                base_type::set_data(
                    util::invoke_fused(f_, std::forward<Args>(args_)));
            }
            catch (...) {
                base_type::set_exception(boost::current_exception());
            }
        }

        BOOST_FORCEINLINE
        void invoke(boost::mpl::true_)
        {
            try {
                util::invoke_fused(f_, std::forward<Args>(args_));
                base_type::set_data(util::unused);
            }
            catch (...) {
                base_type::set_exception(boost::current_exception());
            }
        }

        void apply()
        {
            // wait for all futures to become ready
            traits::serialize_as_future<Args>::call(args_);

            // invoke the function
            typedef typename boost::is_void<result_type>::type is_void;
            invoke(is_void());
        }

        F f_;
        Args args_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    future<typename util::result_of<
        typename util::decay<F>::type(typename util::decay<Ts>::type...)
    >::type>
    invoke_fused_now(boost::mpl::false_, F&& f, Ts&&... vs)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type(typename util::decay<Ts>::type...)
        >::type value_type;

        try {
            return hpx::make_ready_future(
                util::invoke_r<value_type>(f, std::forward<Ts>(vs)...));
        }
        catch (...) {
            return hpx::make_exceptional_future<value_type>(boost::current_exception());
        }
    }

    template <typename F, typename ...Ts>
    future<typename util::result_of<
        typename util::decay<F>::type(typename util::decay<Ts>::type...)
    >::type>
    invoke_fused_now(boost::mpl::true_, F&& f, Ts&&... vs)
    {
        try {
            util::invoke_r<void>(f, std::forward<Ts>(vs)...);
            return hpx::make_ready_future();
        }
        catch (...) {
            return hpx::make_exceptional_future<void>(boost::current_exception());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    typename boost::disable_if<
        util::detail::any_of<
            traits::serialize_as_future<typename util::decay<Ts>::type>...>
      , future<typename util::result_of<
            typename util::decay<F>::type(typename util::decay<Ts>::type...)
         >::type>
    >::type invoke_when_ready(F&& f, Ts&&... vs)
    {
        typedef typename util::result_of<
            typename util::decay<F>::type(typename util::decay<Ts>::type...)
         >::type result_type;
        typedef typename boost::is_void<result_type>::type is_void;

        return invoke_fused_now(is_void(),
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    template <typename F, typename ...Ts>
    typename boost::enable_if<
        util::detail::any_of<
            traits::serialize_as_future<typename util::decay<Ts>::type>...>
      , future<typename util::result_of<
            typename util::decay<F>::type(typename util::decay<Ts>::type...)
         >::type>
    >::type invoke_when_ready(F&& f, Ts&&... vs)
    {
        typedef when_ready<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<Ts>::type...>
        > invoker_type;

        // launch a new thread with high priority which performs the 'waiting'
        boost::intrusive_ptr<invoker_type> p(new invoker_type(
            std::forward<F>(f), std::forward<Ts>(vs)...));
        threads::register_thread_nullary(
            util::deferred_call(&invoker_type::apply, p)
          , "hpx::lcos::local::detail::invoke_when_ready",
          threads::pending, true, threads::thread_priority_boost);

        using traits::future_access;
        return future_access<typename invoker_type::type>::create(std::move(p));
    }
}}}}

#endif
