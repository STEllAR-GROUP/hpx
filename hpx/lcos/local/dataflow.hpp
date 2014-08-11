//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_LCOS_LOCAL_DATAFLOW_HPP
#define HPX_LCOS_LOCAL_DATAFLOW_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/apply.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/traits/is_launch_policy.hpp>

#include <boost/fusion/include/begin.hpp>
#include <boost/fusion/include/deref.hpp>
#include <boost/fusion/include/end.hpp>
#include <boost/fusion/include/next.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/cat.hpp>

namespace hpx { namespace lcos { namespace local { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_future_or_future_range
      : boost::mpl::or_<traits::is_future<T>, traits::is_future_range<T> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    struct reset_dataflow_future
    {
        typedef void result_type;

        template <typename Future>
        BOOST_FORCEINLINE
        void operator()(Future& future) const
        {
            future = Future();
        }

        template <typename Future>
        BOOST_FORCEINLINE
        void operator()(boost::reference_wrapper<Future>& future) const
        {
            future.get() = Future();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Policy, typename Func, typename Futures>
    struct dataflow_frame //-V690
      : hpx::lcos::detail::future_data<
            typename hpx::util::invoke_fused_result_of<
                Func(Futures &&)
            >::type
        >
    {
        typedef
            typename hpx::util::invoke_fused_result_of<
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
          , FFutures && futures
        )
          : policy_(std::move(policy))
          , func_(std::forward<FFunc>(func))
          , futures_(std::forward<FFutures>(futures))
        {}

        BOOST_FORCEINLINE
        void execute(boost::mpl::false_)
        {
            try {
                result_type res =
                    hpx::util::invoke_fused_r<result_type>(
                        func_, std::move(futures_));

                // reset futures
                boost::fusion::for_each(futures_, reset_dataflow_future());

                this->set_data(std::move(res));
            }
            catch(...) {
                this->set_exception(boost::current_exception());
            }
        }

        BOOST_FORCEINLINE
        void execute(boost::mpl::true_)
        {
            try {
                hpx::util::invoke_fused_r<result_type>(
                    func_, std::move(futures_));

                // reset futures
                boost::fusion::for_each(futures_, reset_dataflow_future());

                this->set_data(util::unused_type());
            }
            catch(...) {
                this->set_exception(boost::current_exception());
            }
        }

        template <typename Iter>
        BOOST_FORCEINLINE
        void await(
            BOOST_SCOPED_ENUM(launch) policy, Iter && iter, boost::mpl::true_)
        {
            typedef
                boost::mpl::bool_<boost::is_void<result_type>::value>
                is_void;
            if(policy == hpx::launch::sync)
            {
                execute(is_void());
                return;
            }

            // schedule the final function invocation with high priority
            execute_function_type f = &dataflow_frame::execute;
            boost::intrusive_ptr<dataflow_frame> this_(this);
            threads::register_thread_nullary(
                util::deferred_call(f, this_, is_void())
              , "hpx::lcos::local::dataflow::execute"
              , threads::pending
              , true
              , threads::thread_priority_boost);
        }
        template <typename Iter>
        BOOST_FORCEINLINE
        void await(
            threads::executor& sched, Iter && iter, boost::mpl::true_)
        {
            typedef
                boost::mpl::bool_<boost::is_void<result_type>::value>
                is_void;
            execute_function_type f = &dataflow_frame::execute;
            boost::intrusive_ptr<dataflow_frame> this_(this);
            hpx::apply(sched, f, this_, is_void());
        }

        template <typename TupleIter, typename Iter>
        void await_range(TupleIter iter, Iter next, Iter end)
        {
            for (/**/; next != end; ++next)
            {
                if (!next->is_ready())
                {
                    void (dataflow_frame::*f)(TupleIter, Iter, Iter)
                        = &dataflow_frame::await_range;

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
                        = lcos::detail::get_shared_state(*next);

                    boost::intrusive_ptr<dataflow_frame> this_(this);
                    next_future_data->set_on_completed(
                        boost::bind(
                            f
                          , this_
                          , std::move(iter)
                          , std::move(next)
                          , std::move(end)
                        )
                    );

                    return;
                }
            }

            typedef
                typename boost::fusion::result_of::next<TupleIter>::type
                next_type;

            await(
                policy_
              , boost::fusion::next(iter)
              , boost::mpl::bool_<
                    boost::is_same<next_type, end_type>::value
                >()
            );
        }

        template <typename Iter>
        BOOST_FORCEINLINE
        void await_next(Iter iter, boost::mpl::true_)
        {
            await_range(
                iter
              , boost::begin(boost::unwrap_ref(boost::fusion::deref(iter)))
              , boost::end(boost::unwrap_ref(boost::fusion::deref(iter)))
            );
        }

        template <typename Iter>
        BOOST_FORCEINLINE
        void await_next(Iter iter, boost::mpl::false_)
        {
            typedef
                typename boost::fusion::result_of::next<Iter>::type
                next_type;

            typedef
                typename boost::unwrap_reference<
                    typename util::decay<
                        typename boost::fusion::result_of::deref<Iter>::type
                    >::type
                >::type
                future_type;
            future_type &  f_ =
                boost::fusion::deref(iter);

            if(!f_.is_ready())
            {
                void (dataflow_frame::*f)(Iter, boost::mpl::false_)
                    = &dataflow_frame::await_next;

                typedef
                    typename traits::future_traits<
                        future_type
                    >::type
                    future_result_type;

                boost::intrusive_ptr<
                    lcos::detail::future_data<future_result_type>
                > next_future_data
                    = lcos::detail::get_shared_state(f_);

                boost::intrusive_ptr<dataflow_frame> this_(this);
                next_future_data->set_on_completed(
                    hpx::util::bind(
                        f
                      , this_
                      , std::move(iter)
                      , boost::mpl::false_()
                    )
                );

                return;
            }

            await(
                policy_
              , boost::fusion::next(iter)
              , boost::mpl::bool_<
                    boost::is_same<next_type, end_type>::value
                >()
            );
        }

        template <typename Iter>
        BOOST_FORCEINLINE
        void await(Policy&, Iter iter, boost::mpl::false_)
        {
            typedef
                typename boost::unwrap_reference<
                    typename util::decay<
                        typename boost::fusion::result_of::deref<Iter>::type
                    >::type
                >::type
                future_type;

            typedef typename traits::is_future_range<
                future_type
            >::type is_range;

            await_next(std::move(iter), is_range());
        }

        BOOST_FORCEINLINE void await()
        {
            typedef
                typename boost::fusion::result_of::begin<Futures>::type
                begin_type;

            await(
                policy_
              , boost::fusion::begin(futures_)
              , boost::mpl::bool_<
                    boost::is_same<begin_type, end_type>::value
                >()
            );
        }

        Policy policy_;
        Func func_;
        Futures futures_;
    };
}}}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/local/preprocessed/dataflow.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/dataflow_" HPX_LIMIT_STR ".hpp")
#endif

#define HPX_LCOS_LOCAL_DATAFLOW_DECAY_FUTURE(z, n, d)                           \
    typename util::decay<BOOST_PP_CAT(F, n)>::type                              \
/**/

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT, <hpx/lcos/local/dataflow.hpp>))          \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_LCOS_LOCAL_DATAFLOW_DECAY_FUTURE


#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Func, BOOST_PP_ENUM_PARAMS(N, typename F)>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                BOOST_PP_ENUM(N, HPX_LCOS_LOCAL_DATAFLOW_DECAY_FUTURE, _)
            >
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , Func && func
      , HPX_ENUM_FWD_ARGS(N, F, f)
    )
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    BOOST_PP_ENUM(N, HPX_LCOS_LOCAL_DATAFLOW_DECAY_FUTURE, _)
                >
            >
            frame_type;

        boost::intrusive_ptr<frame_type> p(new frame_type(
                policy
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, F, f))
            ));
        p->await();

        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }

    template <typename Func, BOOST_PP_ENUM_PARAMS(N, typename F)>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            threads::executor
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                BOOST_PP_ENUM(N, HPX_LCOS_LOCAL_DATAFLOW_DECAY_FUTURE, _)
            >
        >
    >::type
    dataflow(
        threads::executor& sched
      , Func && func
      , HPX_ENUM_FWD_ARGS(N, F, f)
    )
    {
        typedef
            detail::dataflow_frame<
                threads::executor
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    BOOST_PP_ENUM(N, HPX_LCOS_LOCAL_DATAFLOW_DECAY_FUTURE, _)
                >
            >
            frame_type;

        boost::intrusive_ptr<frame_type> p(new frame_type(
                sched
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, F, f))
            ));
        p->await();

        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }

    template <typename Func, BOOST_PP_ENUM_PARAMS(N, typename F)>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        traits::is_launch_policy<
            typename util::decay<Func>::type
        >
      , detail::dataflow_frame<
            BOOST_SCOPED_ENUM(launch)
          , typename hpx::util::decay<Func>::type
          , hpx::util::tuple<
                BOOST_PP_ENUM(N, HPX_LCOS_LOCAL_DATAFLOW_DECAY_FUTURE, _)
            >
        >
    >::type
    dataflow(Func && func, HPX_ENUM_FWD_ARGS(N, F, f))
    {
        typedef
            detail::dataflow_frame<
                BOOST_SCOPED_ENUM(launch)
              , typename hpx::util::decay<Func>::type
              , hpx::util::tuple<
                    BOOST_PP_ENUM(N, HPX_LCOS_LOCAL_DATAFLOW_DECAY_FUTURE, _)
                >
            >
            frame_type;

        boost::intrusive_ptr<frame_type> p(new frame_type(
                launch::all
              , std::forward<Func>(func)
              , hpx::util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, F, f))
            ));
        p->await();

        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}}

#endif
