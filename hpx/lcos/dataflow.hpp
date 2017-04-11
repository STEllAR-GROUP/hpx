//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/ref.hpp
// hpxinspect:nodeprecatedname:boost::reference_wrapper

#ifndef HPX_LCOS_DATAFLOW_HPP
#define HPX_LCOS_DATAFLOW_HPP

#include <hpx/config.hpp>
#include <hpx/apply.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unwrap_ref.hpp>

#if HPX_HAVE_ITTNOTIFY != 0 || defined(HPX_HAVE_APEX)
#include <hpx/runtime/get_thread_name.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#if defined(HPX_HAVE_APEX)
#include <hpx/util/apex.hpp>
#else
#include <hpx/util/itt_notify.hpp>
#endif
#endif

#include <hpx/parallel/executors/executor_traits.hpp>

#include <boost/atomic.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/range/functions.hpp>
#include <boost/ref.hpp>

#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

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
        template <typename Future>
        HPX_FORCEINLINE
        typename std::enable_if<
            traits::is_future_or_future_range<Future>::value
        >::type operator()(Future& future) const
        {
            future = Future();
        }

        template <typename Future>
        HPX_FORCEINLINE
        typename std::enable_if<
            traits::is_future_or_future_range<Future>::value
        >::type operator()(boost::reference_wrapper<Future>& future) const
        {
            future.get() = Future();
        }

        template <typename Future>
        HPX_FORCEINLINE
        typename std::enable_if<
            traits::is_future_or_future_range<Future>::value
        >::type operator()(std::reference_wrapper<Future>& future) const
        {
            future.get() = Future();
        }

        template <typename Future>
        HPX_FORCEINLINE
        typename std::enable_if<
            !traits::is_future_or_future_range<Future>::value
        >::type operator()(Future& future) const
        {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Args, typename Enable = void>
    struct dataflow_return;

    template <typename F, typename Args>
    struct dataflow_return<F, Args,
        typename std::enable_if<!traits::is_action<F>::value>::type
    > : util::detail::fused_result_of<F(Args &&)>
    {};

    template <typename Action, typename Args>
    struct dataflow_return<Action, Args,
        typename std::enable_if<traits::is_action<Action>::value>::type
    >
    {
        typedef typename Action::result_type type;
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
        typedef hpx::lcos::detail::future_data<result_type> base_type;

        typedef hpx::lcos::future<result_type> type;

        typedef typename util::detail::make_index_pack<
                util::tuple_size<Futures>::value
            >::type indices_type;

        typedef std::is_void<result_type> is_void;

        template <std::size_t I>
        struct is_end
          : std::integral_constant<
                bool,
                util::tuple_size<Futures>::value == I
            >
        {};

        typedef typename std::conditional<
                is_void::value
              , void(dataflow_frame::*)(indices_type, std::true_type)
              , void(dataflow_frame::*)(indices_type, std::false_type)
            >::type execute_function_type;

    private:
        // workaround gcc regression wrongly instantiating constructors
        dataflow_frame();
        dataflow_frame(dataflow_frame const&);

    public:
        typedef typename base_type::init_no_addref init_no_addref;

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

        template <typename FFunc, typename FFutures>
        dataflow_frame(
            Policy policy
          , FFunc && func
          , FFutures && futures
          , init_no_addref no_addref)
              : base_type(no_addref)
              , policy_(std::move(policy))
              , func_(std::forward<FFunc>(func))
              , futures_(std::forward<FFutures>(futures))
              , done_(false)
        {}

    protected:
        ///////////////////////////////////////////////////////////////////////
        template <std::size_t ...Is>
        HPX_FORCEINLINE
        void execute(util::detail::pack_c<std::size_t, Is...>, std::false_type)
        {
            try {
                result_type res = util::invoke_fused(func_, std::move(futures_));

                // reset futures
                reset_dataflow_future reset;
                int const _sequencer[] = {
                    ((reset(util::get<Is>(futures_))), 0)...
                };
                (void)_sequencer;

                this->set_data(std::move(res));
            }
            catch(...) {
                this->set_exception(boost::current_exception());
            }
        }

        template <std::size_t ...Is>
        HPX_FORCEINLINE
        void execute(util::detail::pack_c<std::size_t, Is...>, std::true_type)
        {
            try {
                util::invoke_fused(func_, std::move(futures_));

                // reset futures
                reset_dataflow_future reset;
                int const _sequencer[] = {
                    ((reset(util::get<Is>(futures_))), 0)...
                };
                (void)_sequencer;

                this->set_data(util::unused_type());
            }
            catch(...) {
                this->set_exception(boost::current_exception());
            }
        }

        HPX_FORCEINLINE void done()
        {
#if HPX_HAVE_ITTNOTIFY != 0
            util::itt::string_handle const& sh =
                traits::get_function_annotation_itt<Func>::call(func_);
            util::itt::task task(hpx::get_thread_itt_domain(), sh);
#elif defined(HPX_HAVE_APEX)
            char const* name = traits::get_function_annotation<Func>::call(func_);
            if (name != nullptr)
            {
                util::apex_wrapper apex_profiler(name);
                execute(indices_type(), is_void());
            }
            else
#endif
            execute(indices_type(), is_void());
        }

        ///////////////////////////////////////////////////////////////////////
        void finalize(hpx::detail::async_policy policy)
        {
            // schedule the final function invocation with high priority
            util::thread_description desc(func_, "dataflow_frame::finalize");
            boost::intrusive_ptr<dataflow_frame> this_(this);

            // simply schedule new thread
            threads::register_thread_nullary(
                util::deferred_call(&dataflow_frame::done, std::move(this_))
              , desc
              , threads::pending
              , true
              , policy.priority()
              , std::size_t(-1)
              , threads::thread_stacksize_current);
        }

        HPX_FORCEINLINE
        void finalize(hpx::detail::sync_policy)
        {
            done();
        }

        void finalize(hpx::detail::fork_policy policy)
        {
            // schedule the final function invocation with high priority
            util::thread_description desc(func_, "dataflow_frame::finalize");
            boost::intrusive_ptr<dataflow_frame> this_(this);

            threads::thread_id_type tid = threads::register_thread_nullary(
                util::deferred_call(&dataflow_frame::done, std::move(this_))
              , desc
              , threads::pending_do_not_schedule
              , true
              , policy.priority()
              , get_worker_thread_num()
              , threads::thread_stacksize_current);

            if (tid)
            {
                // make sure this thread is executed last
                hpx::this_thread::yield_to(thread::id(std::move(tid)));
            }
        }

        void finalize(launch policy)
        {
            if (policy == launch::sync)
            {
                finalize(launch::sync);
            }
            else if (policy == launch::fork)
            {
                finalize(launch::fork);
            }
            else
            {
                finalize(launch::async);
            }
        }

        HPX_FORCEINLINE
        void finalize(threads::executor& sched)
        {
            boost::intrusive_ptr<dataflow_frame> this_(this);
            hpx::apply(sched, &dataflow_frame::done, std::move(this_));
        }

        // handle executors through their executor_traits
        template <typename Executor>
        HPX_FORCEINLINE
        typename std::enable_if<traits::is_executor<Executor>::value>::type
        finalize(Executor& exec)
        {
            boost::intrusive_ptr<dataflow_frame> this_(this);
            parallel::executor_traits<Executor>::apply_execute(exec,
                &dataflow_frame::done, std::move(this_));
        }

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t I>
        HPX_FORCEINLINE
        void do_await(std::true_type)
        {
            done_ = true;
        }

        // Current element is a not a future or future range, e.g. a just plain
        // value.
        template <std::size_t I, typename IsFuture, typename IsRange>
        HPX_FORCEINLINE
        void await_next_respawn(IsFuture is_future, IsRange is_range)
        {
            await_next<I>(is_future, is_range);

            // avoid finalizing more than once
            bool expected = true;
            if (done_.compare_exchange_strong(expected, false))
                finalize(policy_);
        }

        template <std::size_t I>
        HPX_FORCEINLINE
        void await_next(std::false_type, std::false_type)
        {
            do_await<I + 1>(is_end<I + 1>());
        }

        template <std::size_t I, typename Iter>
        void await_range_respawn(Iter next, Iter end)
        {
            await_range<I>(next, end);

            // avoid finalizing more than once
            bool expected = true;
            if (done_.compare_exchange_strong(expected, false))
                finalize(policy_);
        }

        template <std::size_t I, typename Iter>
        void await_range(Iter next, Iter end)
        {
            void (dataflow_frame::*f)(Iter, Iter) =
                &dataflow_frame::await_range_respawn<I>;

            for (/**/; next != end; ++next)
            {
                typedef
                    typename std::iterator_traits<Iter>::value_type
                    future_type;
                typedef
                    typename traits::future_traits<future_type>::type
                    future_result_type;

                typename traits::detail::shared_state_ptr<
                        future_result_type
                    >::type next_future_data =
                        traits::detail::get_shared_state(*next);

                if (next_future_data.get() != nullptr &&
                    !next_future_data->is_ready())
                {
                    next_future_data->execute_deferred();

                    // execute_deferred might have made the future ready
                    if (!next_future_data->is_ready())
                    {
                        boost::intrusive_ptr<dataflow_frame> this_(this);
                        next_future_data->set_on_completed(
                            util::deferred_call(
                                f
                              , std::move(this_)
                              , std::move(next)
                              , std::move(end)
                            )
                        );
                        return;
                    }
                }
            }

            do_await<I + 1>(is_end<I + 1>());
        }

        // Current element is a range (vector) of futures
        template <std::size_t I>
        HPX_FORCEINLINE
        void await_next(std::false_type, std::true_type)
        {
            typedef
                typename util::tuple_element<I, Futures>::type
                future_type;
            future_type & f_ = util::get<I>(futures_);

            await_range<I>(
                boost::begin(util::unwrap_ref(f_))
              , boost::end(util::unwrap_ref(f_))
            );
        }

        // Current element is a simple future
        template <std::size_t I>
        HPX_FORCEINLINE
        void await_next(std::true_type, std::false_type)
        {
            typedef
                typename util::tuple_element<I, Futures>::type
                future_type;
            future_type & f_ = util::get<I>(futures_);

            typedef
                typename traits::future_traits<future_type>::type
                future_result_type;

            typename traits::detail::shared_state_ptr<
                    future_result_type
                >::type next_future_data =
                    traits::detail::get_shared_state(f_);

            if (next_future_data.get() != nullptr &&
                !next_future_data->is_ready())
            {
                next_future_data->execute_deferred();

                // execute_deferred might have made the future ready
                if (!next_future_data->is_ready())
                {
                    void (dataflow_frame::*f)(
                            std::true_type, std::false_type
                        ) = &dataflow_frame::await_next_respawn<I>;

                    boost::intrusive_ptr<dataflow_frame> this_(this);
                    next_future_data->set_on_completed(
                        util::deferred_call(
                            f
                          , std::move(this_)
                          , std::true_type()
                          , std::false_type()
                        )
                    );
                    return;
                }
            }

            do_await<I + 1>(is_end<I + 1>());
        }

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t I>
        HPX_FORCEINLINE
        void do_await(std::false_type)
        {
            typedef
                typename util::tuple_element<I, Futures>::type
                future_type;

            typedef util::detail::any_of<
                    traits::is_future<future_type>,
                    traits::is_ref_wrapped_future<future_type>
                > is_future;

            typedef util::detail::any_of<
                    traits::is_future_range<future_type>,
                    traits::is_ref_wrapped_future_range<future_type>
                > is_range;

            await_next<I>(is_future(), is_range());
        }

    public:
        HPX_FORCEINLINE void do_await()
        {
            do_await<0>(is_end<0>());

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
        typename std::enable_if<traits::is_action<Action>::value>::type>
    {
        template <typename Policy,
            typename Component, typename Signature, typename Derived,
            typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::is_launch_policy<Policy>::value,
            typename dataflow_frame<
                Policy
              , Derived
              , util::tuple<
                    hpx::id_type
                  , typename traits::acquire_future<Ts>::type...
                >
            >::type
        >::type
        call(Policy launch_policy,
            hpx::actions::basic_action<Component, Signature, Derived> const& act,
            naming::id_type const& id, Ts &&... ts)
        {
            typedef
                dataflow_frame<
                    Policy
                  , Derived
                  , util::tuple<
                        hpx::id_type
                      , typename traits::acquire_future<Ts>::type...
                    >
                >
                frame_type;
            typedef typename frame_type::init_no_addref init_no_addref;

            boost::intrusive_ptr<frame_type> p(new frame_type(
                    launch_policy
                  , Derived()
                  , util::forward_as_tuple(
                        id
                      , traits::acquire_future_disp()(std::forward<Ts>(ts))...
                    )
                  , init_no_addref()
                ), false);
            p->do_await();

            using traits::future_access;
            return future_access<typename frame_type::type>::create(std::move(p));
        }

        template <
            typename Component, typename Signature, typename Derived,
            typename ...Ts>
        HPX_FORCEINLINE static
        typename dataflow_frame<
            hpx::detail::async_policy
          , Derived
          , util::tuple<
                hpx::id_type
              , typename traits::acquire_future<Ts>::type...
            >
        >::type
        call(hpx::actions::basic_action<Component, Signature, Derived> const& act,
            naming::id_type const& id, Ts &&... ts)
        {
            return call(launch::async, act, id, std::forward<Ts>(ts)...);
        }
    };

    // launch
    template <typename Action, typename Policy>
    struct dataflow_action_dispatch<Action, Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        template <typename ...Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<
                typename hpx::traits::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(Policy launch_policy, naming::id_type const& id, Ts &&... ts)
        {
            return dataflow_dispatch<Action>::call(launch_policy, Action(), id,
                std::forward<Ts>(ts)...);
        }

//         template <typename DistPolicy, typename ...Ts>
//         HPX_FORCEINLINE static
//         typename std::enable_if<
//             traits::is_distribution_policy<DistPolicy>::value,
//             lcos::future<
//                 typename traits::promise_local_result<
//                     typename hpx::traits::extract_action<
//                         Action
//                     >::remote_result_type
//                 >::type
//             >
//         >::type
//         call(Policy launch_policy, DistPolicy const& policy, Ts&&... ts)
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
                typename hpx::traits::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(naming::id_type const& id, Ts&&... ts)
        {
            return dataflow_action_dispatch<
                    Action, hpx::detail::async_policy
                >::call(launch::async, id, std::forward<Ts>(ts)...);
        }
    };

    // distribution policy
//     template <typename Action, typename Policy>
//     struct dataflow_action_dispatch<Action, Policy,
//         typename std::enable_if<
//             traits::is_distribution_policy<Policy>::value
//         >::type>
//     {
//         template <typename DistPolicy, typename ...Ts>
//         HPX_FORCEINLINE static
//         lcos::future<
//             typename traits::promise_local_result<
//                 typename hpx::traits::extract_action<
//                     Action
//                 >::remote_result_type
//             >::type>
//         call(DistPolicy const& policy, Ts&&... ts)
//         {
//             return dataflow_action_dispatch<
//                     Action, hpx::detail::async_policy
//                 >::call(launch::async, policy, std::forward<Ts>(ts)...);
//         }
//     };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct dataflow_launch_policy_dispatch<Action,
        typename std::enable_if<traits::is_action<Action>::value>::type>
    {
        typedef typename traits::promise_local_result<
                typename hpx::traits::extract_action<
                    Action
                >::remote_result_type
            >::type result_type;

        template <typename Policy, typename ...Ts>
        HPX_FORCEINLINE static
        lcos::future<result_type>
        call(Policy launch_policy,
            Action const&, naming::id_type const& id, Ts &&... ts)
        {
            static_assert(traits::is_launch_policy<Policy>::value,
                "Policy must be a valid launch policy");

            return dataflow_action_dispatch<
                    Action, launch
                >::call(launch_policy, id, std::forward<Ts>(ts)...);
        }

//         template <typename Policy, typename DistPolicy, typename ...Ts>
//         HPX_FORCEINLINE static
//         typename std::enable_if<
//             traits::is_distribution_policy<DistPolicy>::value,
//             lcos::future<result_type>
//         >::type
//         call(Policy launch_policy, Action const&, DistPolicy const& policy,
//             Ts&&... ts)
//         {
//             static_assert(traits::is_launch_policy<Policy>::value,
//                 "Policy must be a valid launch policy");
//
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
                    Action, typename std::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...))
    {
        return lcos::detail::dataflow_action_dispatch<
                Action, typename std::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}

#endif /*HPX_LCOS_DATAFLOW_HPP*/
