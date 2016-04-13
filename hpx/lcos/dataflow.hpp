//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_DATAFLOW_DEC_07_2015_1133AM)
#define HPX_DATAFLOW_DEC_07_2015_1133AM

#include <hpx/config.hpp>
#include <hpx/apply.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/dataflow.hpp>

#include <boost/fusion/include/begin.hpp>
#include <boost/fusion/include/deref.hpp>
#include <boost/fusion/include/end.hpp>
#include <boost/fusion/include/next.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/if.hpp>
#include <boost/range/functions.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/atomic.hpp>
#include <boost/utility/enable_if.hpp>

#include <iterator>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    // dispatch point used for dataflow implementations
    template <typename Func, typename Enable = void>
    struct dataflow_dispatch;

    // dispatch point used for dataflow<Action> implementations
    template <typename Action, typename Policy, typename Enable = void>
    struct dataflow_action_dispatch;

    // dispatch point used for launch_policy implementations
    template <typename Action, typename Enable = void>
    struct dataflow_launch_policy_dispatch;

    ///////////////////////////////////////////////////////////////////////////
    struct reset_dataflow_future
    {
        typedef void result_type;

        template <typename Future>
        HPX_FORCEINLINE
        typename boost::enable_if<
            hpx::traits::is_future_or_future_range<Future>
        >::type
        operator()(Future& future) const
        {
            future = Future();
        }

        template <typename Future>
        HPX_FORCEINLINE
        typename boost::enable_if<
            traits::is_future_or_future_range<Future>
        >::type
        operator()(boost::reference_wrapper<Future>& future) const
        {
            future.get() = Future();
        }

        template <typename Future>
        HPX_FORCEINLINE
        typename boost::disable_if<
            traits::is_future_or_future_range<Future>
        >::type
        operator()(Future& future) const
        {
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Args, typename Enable = void>
    struct dataflow_return;

    template <typename F, typename Args>
    struct dataflow_return<F, Args,
        typename boost::enable_if_c<
            !traits::is_action<F>::value
        >::type>
    {
        typedef typename hpx::util::detail::fused_result_of<
                F(Args &&)
            >::type type;
    };

    template <typename Action, typename Args>
    struct dataflow_return<Action, Args,
        typename boost::enable_if_c<
            traits::is_action<Action>::value
        >::type>
    {
        typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Action
                >::result_type
            >::type type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Policy, typename Func, typename Futures>
    struct dataflow_frame //-V690
      : hpx::lcos::detail::future_data<
            typename detail::dataflow_return<Func, Futures>::type>
    {
        typedef
            typename detail::dataflow_return<Func, Futures>::type
            result_type;

        typedef hpx::lcos::future<result_type> type;

        typedef
            typename boost::fusion::result_of::end<Futures>::type
            end_type;

        typedef
            typename boost::mpl::if_<
                boost::is_void<result_type>
              , void(dataflow_frame::*)(boost::mpl::true_)
              , void(dataflow_frame::*)(boost::mpl::false_)
            >::type
            execute_function_type;

    private:
        // workaround gcc regression wrongly instantiating constructors
        dataflow_frame();
        dataflow_frame(dataflow_frame const&);

    public:
        template <typename FFunc, typename FFutures>
        dataflow_frame(
            Policy policy
          , FFunc && func
          , FFutures && futures)
              : policy_(std::move(policy))
              , func_(std::forward<FFunc>(func))
              , futures_(std::forward<FFutures>(futures))
              , done_(false)
        {}

    protected:
        ///////////////////////////////////////////////////////////////////////
        HPX_FORCEINLINE
        void execute(boost::mpl::false_)
        {
            try {
                result_type res =
                    hpx::util::invoke_fused(
                        func_, std::move(futures_));

                // reset futures
                boost::fusion::for_each(futures_, reset_dataflow_future());

                this->set_data(std::move(res));
            }
            catch(...) {
                this->set_exception(boost::current_exception());
            }
        }

        HPX_FORCEINLINE
        void execute(boost::mpl::true_)
        {
            try {
                hpx::util::invoke_fused(
                    func_, std::move(futures_));

                // reset futures
                boost::fusion::for_each(futures_, reset_dataflow_future());

                this->set_data(util::unused_type());
            }
            catch(...) {
                this->set_exception(boost::current_exception());
            }
        }

        ///////////////////////////////////////////////////////////////////////
        HPX_FORCEINLINE
        void finalize(launch policy)
        {
            typedef
                boost::mpl::bool_<boost::is_void<result_type>::value>
                is_void;
            if (policy == hpx::launch::sync)
            {
                execute(is_void());
                return;
            }

            util::thread_description desc(func_, "dataflow_frame::finalize");

            // schedule the final function invocation with high priority
            execute_function_type f = &dataflow_frame::execute;
            boost::intrusive_ptr<dataflow_frame> this_(this);
            threads::register_thread_nullary(
                util::deferred_call(f, std::move(this_), is_void())
              , desc
              , threads::pending
              , true
              , threads::thread_priority_boost);
        }

        HPX_FORCEINLINE
        void finalize(threads::executor& sched)
        {
            typedef
                boost::mpl::bool_<boost::is_void<result_type>::value>
                is_void;

            execute_function_type f = &dataflow_frame::execute;
            boost::intrusive_ptr<dataflow_frame> this_(this);
            hpx::apply(sched, f, std::move(this_), is_void());
        }

        // handle executors through their executor_traits
        template <typename Executor>
        HPX_FORCEINLINE
        void finalize(Executor& exec)
        {
            typedef
                boost::mpl::bool_<boost::is_void<result_type>::value>
                is_void;

            execute_function_type f = &dataflow_frame::execute;
            boost::intrusive_ptr<dataflow_frame> this_(this);

            parallel::executor_traits<Executor>::apply_execute(exec,
                hpx::util::deferred_call(f, std::move(this_), is_void()));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter>
        HPX_FORCEINLINE
        void do_await(Iter &&, boost::mpl::true_)
        {
            done_ = true;
        }

        // Current element is a not a future or future range, e.g. a just plain
        // value.
        template <typename Iter, typename IsFuture, typename IsRange>
        HPX_FORCEINLINE
        void await_next_respawn(Iter iter, IsFuture is_future,
            IsRange is_range)
        {
            await_next(iter, is_future, is_range);

            // avoid finalizing more than once
            bool expected = true;
            if (done_.compare_exchange_strong(expected, false))
                finalize(policy_);
        }

        template <typename Iter>
        HPX_FORCEINLINE
        void await_next(Iter iter, boost::mpl::false_, boost::mpl::false_)
        {
            typedef
                typename boost::fusion::result_of::next<Iter>::type
                next_type;

            do_await(
                boost::fusion::next(iter)
              , boost::mpl::bool_<
                    boost::is_same<next_type, end_type>::value
                >()
            );
        }

        template <typename TupleIter, typename Iter>
        void await_range_respawn(TupleIter iter, Iter next, Iter end)
        {
            await_range(iter, next, end);

            // avoid finalizing more than once
            bool expected = true;
            if (done_.compare_exchange_strong(expected, false))
                finalize(policy_);
        }

        template <typename TupleIter, typename Iter>
        void await_range(TupleIter iter, Iter next, Iter end)
        {
            void (dataflow_frame::*f)(
                    TupleIter, Iter, Iter
                ) = &dataflow_frame::await_range_respawn;

            for (/**/; next != end; ++next)
            {
                typedef
                    typename std::iterator_traits<
                        Iter
                    >::value_type
                    future_type;
                typedef
                    typename traits::future_traits<
                        future_type
                    >::type
                    future_result_type;

                boost::intrusive_ptr<
                    lcos::detail::future_data<future_result_type>
                > next_future_data
                    = traits::detail::get_shared_state(*next);

                if (!next_future_data->is_ready())
                {
                    next_future_data->execute_deferred();

                    // execute_deferred might have made the future ready
                    if (!next_future_data->is_ready())
                    {
                        boost::intrusive_ptr<dataflow_frame> this_(this);
                        next_future_data->set_on_completed(
                            hpx::util::bind(
                                f
                              , std::move(this_)
                              , std::move(iter)
                              , std::move(next)
                              , std::move(end)
                            )
                        );
                        return;
                    }
                }
            }

            typedef
                typename boost::fusion::result_of::next<TupleIter>::type
                next_type;

            do_await(
                boost::fusion::next(iter)
              , boost::mpl::bool_<
                    boost::is_same<next_type, end_type>::value
                >()
            );
        }

        // Current element is a range (vector) of futures
        template <typename Iter>
        HPX_FORCEINLINE
        void await_next(Iter iter, boost::mpl::false_, boost::mpl::true_)
        {
            await_range(
                iter
              , boost::begin(boost::unwrap_ref(boost::fusion::deref(iter)))
              , boost::end(boost::unwrap_ref(boost::fusion::deref(iter)))
            );
        }

        // Current element is a simple future
        template <typename Iter>
        HPX_FORCEINLINE
        void await_next(Iter iter, boost::mpl::true_, boost::mpl::false_)
        {
            typedef
                typename boost::fusion::result_of::next<Iter>::type
                next_type;

            typedef
                typename util::decay_unwrap<
                    typename boost::fusion::result_of::deref<Iter>::type
                >::type
                future_type;
            future_type &  f_ =
                boost::fusion::deref(iter);

            typedef
                typename traits::future_traits<
                    future_type
                >::type
                future_result_type;

            boost::intrusive_ptr<
                lcos::detail::future_data<future_result_type>
            > next_future_data
                = traits::detail::get_shared_state(f_);

            if(!next_future_data->is_ready())
            {
                next_future_data->execute_deferred();

                // execute_deferred might have made the future ready
                if (!next_future_data->is_ready())
                {
                    void (dataflow_frame::*f)(
                            Iter, boost::mpl::true_, boost::mpl::false_
                        ) = &dataflow_frame::await_next_respawn;

                    boost::intrusive_ptr<dataflow_frame> this_(this);
                    next_future_data->set_on_completed(
                        hpx::util::bind(
                            f
                          , std::move(this_)
                          , std::move(iter)
                          , boost::mpl::true_()
                          , boost::mpl::false_()
                        )
                    );
                    return;
                }
            }

            do_await(
                boost::fusion::next(iter)
              , boost::mpl::bool_<
                    boost::is_same<next_type, end_type>::value
                >()
            );
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter>
        HPX_FORCEINLINE
        void do_await(Iter iter, boost::mpl::false_)
        {
            typedef
                typename util::decay_unwrap<
                    typename boost::fusion::result_of::deref<Iter>::type
                >::type
                future_type;

            typedef typename traits::is_future<
                future_type
            >::type is_future;

            typedef typename traits::is_future_range<
                future_type
            >::type is_range;

            await_next(std::move(iter), is_future(), is_range());
        }

    public:
        HPX_FORCEINLINE void do_await()
        {
            typedef
                typename boost::fusion::result_of::begin<Futures>::type
                begin_type;

            do_await(
                boost::fusion::begin(futures_)
              , boost::mpl::bool_<
                    boost::is_same<begin_type, end_type>::value
                >()
            );

            // avoid finalizing more than once
            bool expected = true;
            if (done_.compare_exchange_strong(expected, false))
                finalize(policy_);
        }

    private:
        Policy policy_;
        Func func_;
        Futures futures_;
        boost::atomic<bool> done_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // any action
    template <typename Action>
    struct dataflow_dispatch<Action,
        typename boost::enable_if_c<
            traits::is_action<Action>::value
        >::type>
    {
        template <
            typename Component, typename Signature, typename Derived,
            typename ...Ts>
        HPX_FORCEINLINE static
        typename dataflow_frame<
            launch
          , Derived
          , hpx::util::tuple<
                hpx::id_type
              , typename hpx::traits::acquire_future<Ts>::type...
            >
        >::type
        call(launch launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const& act,
            naming::id_type const& id, Ts &&... ts)
        {
            typedef
                dataflow_frame<
                    launch
                  , Derived
                  , hpx::util::tuple<
                        hpx::id_type
                      , typename hpx::traits::acquire_future<Ts>::type...
                    >
                >
                frame_type;

            boost::intrusive_ptr<frame_type> p(new frame_type(
                    launch::async
                  , Derived()
                  , hpx::util::forward_as_tuple(
                        id
                      , hpx::traits::acquire_future_disp()(
                            std::forward<Ts>(ts)
                        )...
                    )
                ));
            p->do_await();

            using traits::future_access;
            return future_access<typename frame_type::type>::create(std::move(p));
        }

        template <
            typename Component, typename Signature, typename Derived,
            typename ...Ts>
        HPX_FORCEINLINE static
        typename dataflow_frame<
            launch
          , Derived
          , hpx::util::tuple<
                hpx::id_type
              , typename hpx::traits::acquire_future<Ts>::type...
            >
        >::type
        call(hpx::actions::basic_action<Component, Signature, Derived> const& act,
            naming::id_type const& id, Ts &&... ts)
        {
            return call(launch::all, act, id, std::forward<Ts>(ts)...);
        }
    };

    // launch
    template <typename Action, typename Policy>
    struct dataflow_action_dispatch<Action, Policy,
        typename boost::enable_if_c<
            traits::is_launch_policy<Policy>::value
        >::type>
    {
        template <typename ...Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(launch launch_policy,
            naming::id_type const& id, Ts &&... ts)
        {
            return dataflow_dispatch<Action>::call(launch_policy, Action(), id,
                std::forward<Ts>(ts)...);
        }

//         template <typename DistPolicy, typename ...Ts>
//         HPX_FORCEINLINE static
//         typename boost::enable_if_c<
//             traits::is_distribution_policy<DistPolicy>::value,
//             lcos::future<
//                 typename traits::promise_local_result<
//                     typename hpx::actions::extract_action<
//                         Action
//                     >::remote_result_type
//                 >::type
//             >
//         >::type
//         call(launch launch_policy,
//             DistPolicy const& policy, Ts&&... ts)
//         {
//             return policy.template async<Action>(launch_policy,
//                 std::forward<Ts>(ts)...);
//         }
    };

    // naming::id_type
    template <typename Action>
    struct dataflow_action_dispatch<Action, naming::id_type>
    {
        template <typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(naming::id_type const& id, Ts&&... ts)
        {
            return dataflow_action_dispatch<
                    Action, launch
                >::call(launch::all, id, std::forward<Ts>(ts)...);
        }
    };

    // distribution policy
//     template <typename Action, typename Policy>
//     struct dataflow_action_dispatch<Action, Policy,
//         typename boost::enable_if_c<
//             traits::is_distribution_policy<Policy>::value
//         >::type>
//     {
//         template <typename DistPolicy, typename ...Ts>
//         HPX_FORCEINLINE static
//         lcos::future<
//             typename traits::promise_local_result<
//                 typename hpx::actions::extract_action<
//                     Action
//                 >::remote_result_type
//             >::type>
//         call(DistPolicy const& policy, Ts&&... ts)
//         {
//             return dataflow_action_dispatch<
//                     Action, launch
//                 >::call(launch::all, policy, std::forward<Ts>(ts)...);
//         }
//     };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct dataflow_launch_policy_dispatch<Action,
        typename boost::enable_if_c<
            traits::is_action<Action>::value
        >::type>
    {
        typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Action
                >::remote_result_type
            >::type result_type;

        template <typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<result_type>
        call(launch launch_policy,
            Action const&, naming::id_type const& id, Ts &&... ts)
        {
            return dataflow_action_dispatch<
                    Action, launch
                >::call(launch_policy, id, std::forward<Ts>(ts)...);
        }

//         template <typename DistPolicy, typename ...Ts>
//         HPX_FORCEINLINE static
//         typename boost::enable_if_c<
//             traits::is_distribution_policy<DistPolicy>::value,
//             lcos::future<result_type>
//         >::type
//         call(launch launch_policy,
//             Action const&, DistPolicy const& policy, Ts&&... ts)
//         {
//             return async<Action>(launch_policy, policy, std::forward<Ts>(ts)...);
//         }
    };
}}}

///////////////////////////////////////////////////////////////////////////////
// distributed dataflow: invokes action when ready
namespace hpx
{
    template <typename Action, typename F, typename ...Ts>
    HPX_FORCEINLINE
    auto dataflow(F && f, Ts &&... ts)
    ->  decltype(lcos::detail::dataflow_action_dispatch<
                    Action, typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...))
    {
        return lcos::detail::dataflow_action_dispatch<
                Action, typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}

#endif
