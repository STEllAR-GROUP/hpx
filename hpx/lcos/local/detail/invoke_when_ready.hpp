//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013-2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

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
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iterate.hpp>
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

        when_ready(F const& f, Args&& args)
          : f_(f)
          , args_(std::move(args))
        {}

        when_ready(F&& f, Args&& args)
          : f_(std::move(f))
          , args_(std::move(args))
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

        BOOST_FORCEINLINE
        void invoke()
        {
            typedef typename boost::is_void<result_type>::type is_void;
            invoke(is_void());
        }

        void apply()
        {
            // wait for all futures to become ready
            traits::serialize_as_future<Args>::call(args_);

            // invoke the function
            invoke();
        }

        F f_;
        Args args_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Args>
    future<typename util::invoke_fused_result_of<
        typename util::decay<F>::type(Args)
    >::type>
    invoke_fused_now(F&& f, Args&& args, boost::mpl::false_)
    {
        typedef typename util::invoke_fused_result_of<
            typename util::decay<F>::type(Args)
        >::type value_type;

        try {
            return hpx::make_ready_future(
                util::invoke_fused(f, std::forward<Args>(args)));
        }
        catch (...) {
            return hpx::make_error_future<value_type>(boost::current_exception());
        }
    }

    template <typename F, typename Args>
    future<typename util::invoke_fused_result_of<
        typename util::decay<F>::type(Args)
    >::type>
    invoke_fused_now(F&& f, Args&& args, boost::mpl::true_)
    {
        try {
            util::invoke_fused(f, std::forward<Args>(args));
            return hpx::make_ready_future();
        }
        catch (...) {
            return hpx::make_error_future<void>(boost::current_exception());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Args>
    typename boost::disable_if<
        traits::serialize_as_future<typename util::decay<Args>::type>
      , future<typename when_ready<
            typename util::decay<F>::type
          , typename util::tuple_decay<typename util::decay<Args>::type>::type
         >::result_type>
    >::type
    invoke_fused_when_ready(F&& f, Args&& args)
    {
        typedef typename when_ready<
            typename util::decay<F>::type
          , typename util::decay<Args>::type
         >::result_type result_type;
        typedef typename boost::is_void<result_type>::type is_void;

        return invoke_fused_now(
            std::forward<F>(f), std::forward<Args>(args), is_void());
    }

    template <typename F, typename Args>
    typename boost::enable_if<
        traits::serialize_as_future<typename util::decay<Args>::type>
      , future<typename when_ready<
            typename util::decay<F>::type
          , typename util::tuple_decay<typename util::decay<Args>::type>::type
         >::result_type>
    >::type
    invoke_fused_when_ready(F&& f, Args&& args)
    {
        typedef when_ready<
            typename util::decay<F>::type
          , typename util::tuple_decay<typename util::decay<Args>::type>::type
        > invoker_type;

        // launch a new thread with high priority which performs the 'waiting'
        boost::intrusive_ptr<invoker_type> p(new invoker_type(
            std::forward<F>(f), std::forward<Args>(args)));
        threads::register_thread_nullary(
            util::deferred_call(&invoker_type::apply, p)
          , "hpx::lcos::local::detail::invoke_when_ready",
          threads::pending, true, threads::thread_priority_boost);

        using traits::future_access;
        return future_access<typename invoker_type::type>::create(std::move(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    future<typename when_ready<
        typename util::decay<F>::type
      , util::tuple<>
    >::result_type>
    invoke_when_ready(F&& f)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple());
    }
}}}}


#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/local/detail/preprocessed/invoke_when_ready.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/invoke_when_ready_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_FUNCTION_ARGUMENT_LIMIT, <hpx/lcos/local/detail/invoke_when_ready.hpp>))\
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos { namespace local { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename T)>
    future<typename when_ready<
        typename util::decay<F>::type
      , util::tuple<BOOST_PP_ENUM_PARAMS(N, T)>
    >::result_type> invoke_when_ready(F&& f, HPX_ENUM_FWD_ARGS(N, T, v))
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, T, v)));
    }
}}}}

#undef N

#endif
