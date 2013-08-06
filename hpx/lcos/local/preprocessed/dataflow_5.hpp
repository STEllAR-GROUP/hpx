// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        template <typename Policy, typename Func, typename F0>
        struct dataflow_frame_1
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type;
            typedef
                hpx::util::tuple1<
                    f0_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_1>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_1::*)(boost::mpl::true_)
                  , void(dataflow_frame_1::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            Policy policy_;
            func_type func_;
            template <typename FFunc, typename A0>
            dataflow_frame_1(
                Policy policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0
            )
              : futures_(
                    boost::forward<A0>(f0)
                )
              , policy_(boost::move(policy))
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_1::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                threads::executor& sched, BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                execute_function_type f = &dataflow_frame_1::execute;
                hpx::apply(sched, hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                typedef
                    typename std::iterator_traits<
                        Iter
                    >::value_type
                    future_type;
                if(!next->ready())
                {
                    void (dataflow_frame_1::*f)
                        (Iter, Iter)
                        = &dataflow_frame_1::await_range;
                    typedef
                        typename lcos::future_traits<
                            future_type
                        >::value_type
                        future_result_type;
                    boost::intrusive_ptr<
                        lcos::detail::future_data_base<future_result_type>
                    > next_future_data
                        = hpx::lcos::detail::get_future_data(*next);
                    next_future_data->set_on_completed(
                        boost::move(
                            boost::bind(
                                f
                              , future_base_type(this)
                              , boost::move(next)
                              , boost::move(end)
                            )
                        )
                    );
                    return;
                }
                await_range(boost::move(++next), boost::move(end));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await_next(Iter iter, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::move(boost::begin(boost::fusion::deref(iter)))
                  , boost::move(boost::end(boost::fusion::deref(iter)))
                );
                await(
                    policy_
                  , boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
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
                    typename util::detail::remove_reference<
                        typename boost::fusion::result_of::deref<Iter>::type
                    >::type
                    future_type;
                future_type & f_ =
                    boost::fusion::deref(iter);
                if(!f_.ready())
                {
                    void (dataflow_frame_1::*f)
                        (Iter, boost::mpl::false_)
                        = &dataflow_frame_1::await_next;
                    typedef
                        typename lcos::future_traits<
                            future_type
                        >::value_type
                        future_result_type;
                    boost::intrusive_ptr<
                        lcos::detail::future_data_base<future_result_type>
                    > next_future_data
                        = hpx::lcos::detail::get_future_data(f_);
                    next_future_data->set_on_completed(
                        boost::move(
                            hpx::util::bind(
                                f
                              , future_base_type(this)
                              , boost::move(iter)
                              , boost::mpl::false_()
                            )
                        )
                    );
                    return;
                }
                await(
                    policy_
                  , boost::move(boost::fusion::next(iter))
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
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(boost::move(iter), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    policy_
                  , boost::move(boost::fusion::begin(futures_))
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                );
            }
            BOOST_FORCEINLINE
            type get_future()
            {
                await();
                return
                    lcos::detail::make_future_from_data(
                        boost::intrusive_ptr<
                            lcos::detail::future_data_base<result_type>
                        >(this)
                    );
            }
        };
    }
    
    template <typename Func, typename F0>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_1<
            BOOST_SCOPED_ENUM(launch)
          , Func
          , F0
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0
    )
    {
        typedef
            detail::dataflow_frame_1<
                BOOST_SCOPED_ENUM(launch)
              , Func
              , F0
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 )
            );
        return frame->get_future();
    }
    template <typename Func, typename F0>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_1<
            threads::executor
          , Func
          , F0
        >
    >::type
    dataflow(
        threads::executor& sched
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0
    )
    {
        typedef
            detail::dataflow_frame_1<
                threads::executor
              , Func
              , F0
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                sched
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 )
            );
        return frame->get_future();
    }
    template <typename Func, typename F0>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_1<
            BOOST_SCOPED_ENUM(launch)
          , Func
          , F0
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0)
    {
        typedef
            detail::dataflow_frame_1<
                BOOST_SCOPED_ENUM(launch)
              , Func
              , F0
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        template <typename Policy, typename Func, typename F0 , typename F1>
        struct dataflow_frame_2
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type;
            typedef
                hpx::util::tuple2<
                    f0_type , f1_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_2>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_2::*)(boost::mpl::true_)
                  , void(dataflow_frame_2::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            Policy policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1>
            dataflow_frame_2(
                Policy policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1)
                )
              , policy_(boost::move(policy))
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_2::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                threads::executor& sched, BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                execute_function_type f = &dataflow_frame_2::execute;
                hpx::apply(sched, hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                typedef
                    typename std::iterator_traits<
                        Iter
                    >::value_type
                    future_type;
                if(!next->ready())
                {
                    void (dataflow_frame_2::*f)
                        (Iter, Iter)
                        = &dataflow_frame_2::await_range;
                    typedef
                        typename lcos::future_traits<
                            future_type
                        >::value_type
                        future_result_type;
                    boost::intrusive_ptr<
                        lcos::detail::future_data_base<future_result_type>
                    > next_future_data
                        = hpx::lcos::detail::get_future_data(*next);
                    next_future_data->set_on_completed(
                        boost::move(
                            boost::bind(
                                f
                              , future_base_type(this)
                              , boost::move(next)
                              , boost::move(end)
                            )
                        )
                    );
                    return;
                }
                await_range(boost::move(++next), boost::move(end));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await_next(Iter iter, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::move(boost::begin(boost::fusion::deref(iter)))
                  , boost::move(boost::end(boost::fusion::deref(iter)))
                );
                await(
                    policy_
                  , boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
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
                    typename util::detail::remove_reference<
                        typename boost::fusion::result_of::deref<Iter>::type
                    >::type
                    future_type;
                future_type & f_ =
                    boost::fusion::deref(iter);
                if(!f_.ready())
                {
                    void (dataflow_frame_2::*f)
                        (Iter, boost::mpl::false_)
                        = &dataflow_frame_2::await_next;
                    typedef
                        typename lcos::future_traits<
                            future_type
                        >::value_type
                        future_result_type;
                    boost::intrusive_ptr<
                        lcos::detail::future_data_base<future_result_type>
                    > next_future_data
                        = hpx::lcos::detail::get_future_data(f_);
                    next_future_data->set_on_completed(
                        boost::move(
                            hpx::util::bind(
                                f
                              , future_base_type(this)
                              , boost::move(iter)
                              , boost::mpl::false_()
                            )
                        )
                    );
                    return;
                }
                await(
                    policy_
                  , boost::move(boost::fusion::next(iter))
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
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(boost::move(iter), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    policy_
                  , boost::move(boost::fusion::begin(futures_))
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                );
            }
            BOOST_FORCEINLINE
            type get_future()
            {
                await();
                return
                    lcos::detail::make_future_from_data(
                        boost::intrusive_ptr<
                            lcos::detail::future_data_base<result_type>
                        >(this)
                    );
            }
        };
    }
    
    template <typename Func, typename F0 , typename F1>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_2<
            BOOST_SCOPED_ENUM(launch)
          , Func
          , F0 , F1
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1
    )
    {
        typedef
            detail::dataflow_frame_2<
                BOOST_SCOPED_ENUM(launch)
              , Func
              , F0 , F1
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 )
            );
        return frame->get_future();
    }
    template <typename Func, typename F0 , typename F1>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_2<
            threads::executor
          , Func
          , F0 , F1
        >
    >::type
    dataflow(
        threads::executor& sched
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1
    )
    {
        typedef
            detail::dataflow_frame_2<
                threads::executor
              , Func
              , F0 , F1
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                sched
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 )
            );
        return frame->get_future();
    }
    template <typename Func, typename F0 , typename F1>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_2<
            BOOST_SCOPED_ENUM(launch)
          , Func
          , F0 , F1
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1)
    {
        typedef
            detail::dataflow_frame_2<
                BOOST_SCOPED_ENUM(launch)
              , Func
              , F0 , F1
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        template <typename Policy, typename Func, typename F0 , typename F1 , typename F2>
        struct dataflow_frame_3
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type;
            typedef
                hpx::util::tuple3<
                    f0_type , f1_type , f2_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_3>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_3::*)(boost::mpl::true_)
                  , void(dataflow_frame_3::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            Policy policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2>
            dataflow_frame_3(
                Policy policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2)
                )
              , policy_(boost::move(policy))
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_3::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                threads::executor& sched, BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                execute_function_type f = &dataflow_frame_3::execute;
                hpx::apply(sched, hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                typedef
                    typename std::iterator_traits<
                        Iter
                    >::value_type
                    future_type;
                if(!next->ready())
                {
                    void (dataflow_frame_3::*f)
                        (Iter, Iter)
                        = &dataflow_frame_3::await_range;
                    typedef
                        typename lcos::future_traits<
                            future_type
                        >::value_type
                        future_result_type;
                    boost::intrusive_ptr<
                        lcos::detail::future_data_base<future_result_type>
                    > next_future_data
                        = hpx::lcos::detail::get_future_data(*next);
                    next_future_data->set_on_completed(
                        boost::move(
                            boost::bind(
                                f
                              , future_base_type(this)
                              , boost::move(next)
                              , boost::move(end)
                            )
                        )
                    );
                    return;
                }
                await_range(boost::move(++next), boost::move(end));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await_next(Iter iter, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::move(boost::begin(boost::fusion::deref(iter)))
                  , boost::move(boost::end(boost::fusion::deref(iter)))
                );
                await(
                    policy_
                  , boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
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
                    typename util::detail::remove_reference<
                        typename boost::fusion::result_of::deref<Iter>::type
                    >::type
                    future_type;
                future_type & f_ =
                    boost::fusion::deref(iter);
                if(!f_.ready())
                {
                    void (dataflow_frame_3::*f)
                        (Iter, boost::mpl::false_)
                        = &dataflow_frame_3::await_next;
                    typedef
                        typename lcos::future_traits<
                            future_type
                        >::value_type
                        future_result_type;
                    boost::intrusive_ptr<
                        lcos::detail::future_data_base<future_result_type>
                    > next_future_data
                        = hpx::lcos::detail::get_future_data(f_);
                    next_future_data->set_on_completed(
                        boost::move(
                            hpx::util::bind(
                                f
                              , future_base_type(this)
                              , boost::move(iter)
                              , boost::mpl::false_()
                            )
                        )
                    );
                    return;
                }
                await(
                    policy_
                  , boost::move(boost::fusion::next(iter))
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
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(boost::move(iter), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    policy_
                  , boost::move(boost::fusion::begin(futures_))
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                );
            }
            BOOST_FORCEINLINE
            type get_future()
            {
                await();
                return
                    lcos::detail::make_future_from_data(
                        boost::intrusive_ptr<
                            lcos::detail::future_data_base<result_type>
                        >(this)
                    );
            }
        };
    }
    
    template <typename Func, typename F0 , typename F1 , typename F2>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_3<
            BOOST_SCOPED_ENUM(launch)
          , Func
          , F0 , F1 , F2
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2
    )
    {
        typedef
            detail::dataflow_frame_3<
                BOOST_SCOPED_ENUM(launch)
              , Func
              , F0 , F1 , F2
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 )
            );
        return frame->get_future();
    }
    template <typename Func, typename F0 , typename F1 , typename F2>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_3<
            threads::executor
          , Func
          , F0 , F1 , F2
        >
    >::type
    dataflow(
        threads::executor& sched
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2
    )
    {
        typedef
            detail::dataflow_frame_3<
                threads::executor
              , Func
              , F0 , F1 , F2
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                sched
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 )
            );
        return frame->get_future();
    }
    template <typename Func, typename F0 , typename F1 , typename F2>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_3<
            BOOST_SCOPED_ENUM(launch)
          , Func
          , F0 , F1 , F2
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2)
    {
        typedef
            detail::dataflow_frame_3<
                BOOST_SCOPED_ENUM(launch)
              , Func
              , F0 , F1 , F2
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        template <typename Policy, typename Func, typename F0 , typename F1 , typename F2 , typename F3>
        struct dataflow_frame_4
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type;
            typedef
                hpx::util::tuple4<
                    f0_type , f1_type , f2_type , f3_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_4>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_4::*)(boost::mpl::true_)
                  , void(dataflow_frame_4::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            Policy policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3>
            dataflow_frame_4(
                Policy policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3)
                )
              , policy_(boost::move(policy))
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_4::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                threads::executor& sched, BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                execute_function_type f = &dataflow_frame_4::execute;
                hpx::apply(sched, hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                typedef
                    typename std::iterator_traits<
                        Iter
                    >::value_type
                    future_type;
                if(!next->ready())
                {
                    void (dataflow_frame_4::*f)
                        (Iter, Iter)
                        = &dataflow_frame_4::await_range;
                    typedef
                        typename lcos::future_traits<
                            future_type
                        >::value_type
                        future_result_type;
                    boost::intrusive_ptr<
                        lcos::detail::future_data_base<future_result_type>
                    > next_future_data
                        = hpx::lcos::detail::get_future_data(*next);
                    next_future_data->set_on_completed(
                        boost::move(
                            boost::bind(
                                f
                              , future_base_type(this)
                              , boost::move(next)
                              , boost::move(end)
                            )
                        )
                    );
                    return;
                }
                await_range(boost::move(++next), boost::move(end));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await_next(Iter iter, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::move(boost::begin(boost::fusion::deref(iter)))
                  , boost::move(boost::end(boost::fusion::deref(iter)))
                );
                await(
                    policy_
                  , boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
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
                    typename util::detail::remove_reference<
                        typename boost::fusion::result_of::deref<Iter>::type
                    >::type
                    future_type;
                future_type & f_ =
                    boost::fusion::deref(iter);
                if(!f_.ready())
                {
                    void (dataflow_frame_4::*f)
                        (Iter, boost::mpl::false_)
                        = &dataflow_frame_4::await_next;
                    typedef
                        typename lcos::future_traits<
                            future_type
                        >::value_type
                        future_result_type;
                    boost::intrusive_ptr<
                        lcos::detail::future_data_base<future_result_type>
                    > next_future_data
                        = hpx::lcos::detail::get_future_data(f_);
                    next_future_data->set_on_completed(
                        boost::move(
                            hpx::util::bind(
                                f
                              , future_base_type(this)
                              , boost::move(iter)
                              , boost::mpl::false_()
                            )
                        )
                    );
                    return;
                }
                await(
                    policy_
                  , boost::move(boost::fusion::next(iter))
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
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(boost::move(iter), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    policy_
                  , boost::move(boost::fusion::begin(futures_))
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                );
            }
            BOOST_FORCEINLINE
            type get_future()
            {
                await();
                return
                    lcos::detail::make_future_from_data(
                        boost::intrusive_ptr<
                            lcos::detail::future_data_base<result_type>
                        >(this)
                    );
            }
        };
    }
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_4<
            BOOST_SCOPED_ENUM(launch)
          , Func
          , F0 , F1 , F2 , F3
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3
    )
    {
        typedef
            detail::dataflow_frame_4<
                BOOST_SCOPED_ENUM(launch)
              , Func
              , F0 , F1 , F2 , F3
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 )
            );
        return frame->get_future();
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_4<
            threads::executor
          , Func
          , F0 , F1 , F2 , F3
        >
    >::type
    dataflow(
        threads::executor& sched
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3
    )
    {
        typedef
            detail::dataflow_frame_4<
                threads::executor
              , Func
              , F0 , F1 , F2 , F3
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                sched
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 )
            );
        return frame->get_future();
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_4<
            BOOST_SCOPED_ENUM(launch)
          , Func
          , F0 , F1 , F2 , F3
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3)
    {
        typedef
            detail::dataflow_frame_4<
                BOOST_SCOPED_ENUM(launch)
              , Func
              , F0 , F1 , F2 , F3
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        template <typename Policy, typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
        struct dataflow_frame_5
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type;
            typedef
                hpx::util::tuple5<
                    f0_type , f1_type , f2_type , f3_type , f4_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_5>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_5::*)(boost::mpl::true_)
                  , void(dataflow_frame_5::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            Policy policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
            dataflow_frame_5(
                Policy policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4)
                )
              , policy_(boost::move(policy))
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_5::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                threads::executor& sched, BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                execute_function_type f = &dataflow_frame_5::execute;
                hpx::apply(sched, hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                typedef
                    typename std::iterator_traits<
                        Iter
                    >::value_type
                    future_type;
                if(!next->ready())
                {
                    void (dataflow_frame_5::*f)
                        (Iter, Iter)
                        = &dataflow_frame_5::await_range;
                    typedef
                        typename lcos::future_traits<
                            future_type
                        >::value_type
                        future_result_type;
                    boost::intrusive_ptr<
                        lcos::detail::future_data_base<future_result_type>
                    > next_future_data
                        = hpx::lcos::detail::get_future_data(*next);
                    next_future_data->set_on_completed(
                        boost::move(
                            boost::bind(
                                f
                              , future_base_type(this)
                              , boost::move(next)
                              , boost::move(end)
                            )
                        )
                    );
                    return;
                }
                await_range(boost::move(++next), boost::move(end));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await_next(Iter iter, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::move(boost::begin(boost::fusion::deref(iter)))
                  , boost::move(boost::end(boost::fusion::deref(iter)))
                );
                await(
                    policy_
                  , boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
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
                    typename util::detail::remove_reference<
                        typename boost::fusion::result_of::deref<Iter>::type
                    >::type
                    future_type;
                future_type & f_ =
                    boost::fusion::deref(iter);
                if(!f_.ready())
                {
                    void (dataflow_frame_5::*f)
                        (Iter, boost::mpl::false_)
                        = &dataflow_frame_5::await_next;
                    typedef
                        typename lcos::future_traits<
                            future_type
                        >::value_type
                        future_result_type;
                    boost::intrusive_ptr<
                        lcos::detail::future_data_base<future_result_type>
                    > next_future_data
                        = hpx::lcos::detail::get_future_data(f_);
                    next_future_data->set_on_completed(
                        boost::move(
                            hpx::util::bind(
                                f
                              , future_base_type(this)
                              , boost::move(iter)
                              , boost::mpl::false_()
                            )
                        )
                    );
                    return;
                }
                await(
                    policy_
                  , boost::move(boost::fusion::next(iter))
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
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(boost::move(iter), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    policy_
                  , boost::move(boost::fusion::begin(futures_))
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                );
            }
            BOOST_FORCEINLINE
            type get_future()
            {
                await();
                return
                    lcos::detail::make_future_from_data(
                        boost::intrusive_ptr<
                            lcos::detail::future_data_base<result_type>
                        >(this)
                    );
            }
        };
    }
    
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_5<
            BOOST_SCOPED_ENUM(launch)
          , Func
          , F0 , F1 , F2 , F3 , F4
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4
    )
    {
        typedef
            detail::dataflow_frame_5<
                BOOST_SCOPED_ENUM(launch)
              , Func
              , F0 , F1 , F2 , F3 , F4
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 )
            );
        return frame->get_future();
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_5<
            threads::executor
          , Func
          , F0 , F1 , F2 , F3 , F4
        >
    >::type
    dataflow(
        threads::executor& sched
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4
    )
    {
        typedef
            detail::dataflow_frame_5<
                threads::executor
              , Func
              , F0 , F1 , F2 , F3 , F4
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                sched
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 )
            );
        return frame->get_future();
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_launch_policy<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_5<
            BOOST_SCOPED_ENUM(launch)
          , Func
          , F0 , F1 , F2 , F3 , F4
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4)
    {
        typedef
            detail::dataflow_frame_5<
                BOOST_SCOPED_ENUM(launch)
              , Func
              , F0 , F1 , F2 , F3 , F4
            >
            frame_type;
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 )
            );
        return frame->get_future();
    }
}}}
