//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_LOCAL_DATAFLOW_HPP
#define HPX_LCOS_LOCAL_DATAFLOW_HPP

#include <hpx/config.hpp>
#include <hpx/apply.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/deferred_call.hpp>

#include <boost/fusion/include/begin.hpp>
#include <boost/fusion/include/deref.hpp>
#include <boost/fusion/include/end.hpp>
#include <boost/fusion/include/next.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>
#include <boost/range/functions.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/atomic.hpp>

#include <algorithm>

namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
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

        ///////////////////////////////////////////////////////////////////////
        template <typename Policy, typename Func, typename Futures>
        struct dataflow_frame //-V690
          : hpx::lcos::detail::future_data<
                typename hpx::util::detail::fused_result_of<
                    Func(Futures &&)
                >::type
            >
        {
            typedef
                typename hpx::util::detail::fused_result_of<
                    Func(Futures &&)
                >::type
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
            ///////////////////////////////////////////////////////////////////
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

            ///////////////////////////////////////////////////////////////////
            HPX_FORCEINLINE
            void finalize(BOOST_SCOPED_ENUM(launch) policy)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if (policy == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }

                // schedule the final function invocation with high priority
                execute_function_type f = &dataflow_frame::execute;
                boost::intrusive_ptr<dataflow_frame> this_(this);
                threads::register_thread_nullary(
                    util::deferred_call(f, std::move(this_), is_void())
                  , "hpx::lcos::local::dataflow::execute"
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

            ///////////////////////////////////////////////////////////////////
            template <typename Policy_, typename Iter>
            HPX_FORCEINLINE
            void do_await(Policy_ &&, Iter &&, boost::mpl::true_)
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
                    policy_
                  , boost::fusion::next(iter)
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
                    policy_
                  , boost::fusion::next(iter)
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
                    policy_
                  , boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }

            ///////////////////////////////////////////////////////////////////////
            template <typename Iter>
            HPX_FORCEINLINE
            void do_await(Policy&, Iter iter, boost::mpl::false_)
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
                    policy_
                  , boost::fusion::begin(futures_)
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

        ///////////////////////////////////////////////////////////////////////
        template <typename Func, typename Enable = void>
        struct dataflow_dispatch;

        // BOOST_SCOPED_ENUM(launch)
        template <typename Policy>
        struct dataflow_dispatch<Policy,
            typename boost::enable_if_c<
                traits::is_launch_policy<typename util::decay<Policy>::type>::value
            >::type>
        {
            template <typename F, typename ...Ts>
            HPX_FORCEINLINE static
            typename detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<F>::type
              , hpx::util::tuple<
                    typename hpx::traits::acquire_future<Ts>::type...
                >
            >::type
            call(Policy const& policy, F && f, Ts&&... ts)
            {
                typedef
                    detail::dataflow_frame<
                        BOOST_SCOPED_ENUM(launch)
                      , typename hpx::util::decay<F>::type
                      , hpx::util::tuple<
                            typename hpx::traits::acquire_future<Ts>::type...
                        >
                    >
                    frame_type;

                boost::intrusive_ptr<frame_type> p(new frame_type(
                        policy
                      , std::forward<F>(f)
                      , hpx::util::forward_as_tuple(
                            hpx::traits::acquire_future_disp()(
                                std::forward<Ts>(ts)
                            )...
                        )
                    ));
                p->do_await();

                using traits::future_access;
                return future_access<typename frame_type::type>::create(std::move(p));
            }
        };

        // plain function or function object
        template <typename Func, typename Enable>
        struct dataflow_dispatch
        {
            template <typename F, typename ...Ts>
            HPX_FORCEINLINE static
            typename detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<F>::type
              , hpx::util::tuple<
                    typename hpx::traits::acquire_future<Ts>::type...
                >
            >::type
            call(F && f, Ts&&... ts)
            {
                return dataflow_dispatch<BOOST_SCOPED_ENUM(launch)>::call(
                    launch::all, std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };

        // threads::executor
        template <typename Executor>
        struct dataflow_dispatch<Executor,
            typename boost::enable_if_c<
                traits::is_threads_executor<typename util::decay<Executor>::type>::value
            >::type>
        {
            template <typename F, typename ...Ts>
            HPX_FORCEINLINE static
            typename detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<F>::type
              , hpx::util::tuple<
                    typename hpx::traits::acquire_future<Ts>::type...
                >
            >::type
            call(Executor& sched, F && f, Ts&&... ts)
            {
                typedef
                    detail::dataflow_frame<
                        threads::executor
                      , typename hpx::util::decay<F>::type
                      , hpx::util::tuple<
                            typename hpx::traits::acquire_future<Ts>::type...
                        >
                    >
                    frame_type;

                boost::intrusive_ptr<frame_type> p(new frame_type(
                        std::forward<Executor>(sched)
                      , std::forward<F>(f)
                      , hpx::util::forward_as_tuple(
                            hpx::traits::acquire_future_disp()(
                                std::forward<Ts>(ts)
                            )...
                        )
                    ));
                p->do_await();

                using traits::future_access;
                return future_access<typename frame_type::type>::create(std::move(p));
            }
        };

        // parallel executor
        template <typename Executor>
        struct dataflow_dispatch<Executor,
            typename boost::enable_if_c<
                traits::is_executor<typename util::decay<Executor>::type>::value
            >::type>
        {
            template <typename F, typename ...Ts>
            HPX_FORCEINLINE static
            typename detail::dataflow_frame<
                typename util::decay<Executor>::type
              , typename hpx::util::decay<F>::type
              , hpx::util::tuple<
                    typename hpx::traits::acquire_future<Ts>::type...
                >
            >::type
            call(Executor& exec, F && f, Ts&&... ts)
            {
                typedef
                    detail::dataflow_frame<
                        typename hpx::util::decay<Executor>::type
                      , typename hpx::util::decay<F>::type
                      , hpx::util::tuple<
                            typename hpx::traits::acquire_future<Ts>::type...
                        >
                    >
                    frame_type;

                boost::intrusive_ptr<frame_type> p(new frame_type(
                        std::forward<Executor>(exec)
                      , std::forward<F>(f)
                      , hpx::util::forward_as_tuple(
                            hpx::traits::acquire_future_disp()(
                                std::forward<Ts>(ts)
                            )...
                        )
                    ));
                p->do_await();

                using traits::future_access;
                return future_access<typename frame_type::type>::create(std::move(p));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    HPX_FORCEINLINE
    auto dataflow(F && f, Ts&&... ts)
    ->  decltype(detail::dataflow_dispatch<F>::call(
            std::forward<F>(f), std::forward<Ts>(ts)...
        ))
    {
        return detail::dataflow_dispatch<F>::call(
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}}}

#endif
