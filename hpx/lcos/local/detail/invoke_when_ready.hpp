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
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
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
    template <typename Tuple>
    struct tuple_has_futures
      : boost::mpl::not_<boost::is_same<
            typename boost::mpl::find_if<
                Tuple, traits::is_future<util::decay<boost::mpl::_1> >
            >::type
          , typename boost::mpl::end<Tuple>::type
        > >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Args>
    struct when_ready;

    template <typename F, typename Args, typename Callback>
    struct set_invoke_on_completed_callback_impl
    {
        explicit set_invoke_on_completed_callback_impl(
                when_ready<F, Args>& when, Callback const& callback)
          : when_(when)
          , callback_(callback)
        {}

        template <typename Future>
        typename boost::enable_if<traits::is_future<Future> >::type
        operator()(Future& future) const
        {
            // do not touch any futures which are already ready
            if (future.valid() && !future.is_ready())
            {
                typedef
                    typename lcos::detail::shared_state_ptr_for<Future>::type
                    shared_state_ptr;

                using traits::future_access;
                shared_state_ptr const& shared_state =
                    future_access<Future>::get_shared_state(future);

                shared_state->set_on_completed(Callback(callback_));
            } else {
                ++when_.count_;
            }
        }

        template <typename Value>
        typename boost::disable_if<traits::is_future<Value> >::type
        operator()(Value& value) const
        {
            ++when_.count_;
        }

        when_ready<F, Args>& when_;
        Callback const& callback_;
    };

    template <typename F, typename Args, typename Callback>
    void set_invoke_on_completed_callback(when_ready<F, Args>& when,
        Callback const& callback)
    {
        set_invoke_on_completed_callback_impl<F, Args, Callback>
            set_invoke_on_completed_callback_helper(when, callback);
        boost::fusion::for_each(when.args_, set_invoke_on_completed_callback_helper);
    }

    template <typename F, typename Args>
    struct when_ready : boost::enable_shared_from_this<when_ready<F, Args> >
    {
    private:
        static std::size_t const needed_count_ =
            boost::fusion::result_of::size<Args>::value;

        void on_future_ready(threads::thread_id_type const& id)
        {
            if (count_.fetch_add(1) + 1 == needed_count_)
            {
                // reactivate waiting thread only if it's not us
                if (id != threads::get_self_id())
                    threads::set_thread_state(id, threads::pending);
            }
        }

    private:
        // workaround gcc regression wrongly instantiating constructors
        when_ready();
        when_ready(when_ready const&);

    public:
        typedef typename util::invoke_fused_result_of<F&(Args)>::type result_type;

        when_ready(F const& f, Args&& args)
          : f_(f)
          , args_(std::move(args))
          , count_(0)
        {}

        when_ready(F&& f, Args&& args)
          : f_(std::move(f))
          , args_(std::move(args))
          , count_(0)
        {}

        result_type operator()()
        {
            // set callback functions to executed when future is ready
            set_invoke_on_completed_callback(*this,
                util::bind(
                    &when_ready::on_future_ready, this->shared_from_this(),
                    threads::get_self_id()));

            // if all of the requested futures are already set, our
            // callback above has already been called often enough, otherwise
            // we suspend ourselves
            if (count_.load(boost::memory_order_acquire) < needed_count_)
            {
                // wait for any of the futures to return to become ready
                this_thread::suspend(threads::suspended);
            }

            // all futures should be ready
            HPX_ASSERT(count_.load(boost::memory_order_acquire) >= needed_count_);

            return util::invoke_fused_r<result_type>(f_, std::move(args_));
        }

        F f_;
        Args args_;
        boost::atomic<std::size_t> count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Args>
    future<typename util::invoke_fused_result_of<
        typename util::decay<F>::type(Args)
    >::type> invoke_fused_now(F&& f, Args&& args, boost::mpl::false_)
    {
        return hpx::make_ready_future(
            util::invoke_fused(f, std::forward<Args>(args)));
    }

    template <typename F, typename Args>
    future<typename util::invoke_fused_result_of<
        typename util::decay<F>::type(Args)
    >::type> invoke_fused_now(F&& f, Args&& args, boost::mpl::true_)
    {
        util::invoke_fused(f, std::forward<Args>(args));
        return hpx::make_ready_future();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Args>
    typename boost::disable_if<
        tuple_has_futures<typename util::decay<Args>::type>
      , future<typename when_ready<
            typename util::decay<F>::type
          , typename util::tuple_decay<typename util::decay<Args>::type>::type
         >::result_type>
    >::type invoke_fused_when_ready(F&& f, Args&& args)
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
        tuple_has_futures<typename util::decay<Args>::type>
      , future<typename when_ready<
            typename util::decay<F>::type
          , typename util::tuple_decay<typename util::decay<Args>::type>::type
         >::result_type>
    >::type invoke_fused_when_ready(F&& f, Args&& args)
    {
        typedef typename util::decay<Args>::type arguments_type;
        typedef when_ready<
            typename util::decay<F>::type
          , typename util::tuple_decay<typename util::decay<Args>::type>::type
        > invoker;

        arguments_type lazy_args(std::forward<Args>(args));
        lcos::local::futures_factory<typename invoker::result_type()> p(
            util::bind(&invoker::operator(), boost::make_shared<invoker>(
                std::forward<F>(f), std::move(lazy_args))));

        p.apply();
        return p.get_future();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    future<typename when_ready<
        typename util::decay<F>::type
      , util::tuple<>
    >::result_type> invoke_when_ready(F&& f)
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
