// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0>
        struct dataflow_frame_1
          : boost::enable_shared_from_this<
                dataflow_frame_1<
                    Func, F0
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type;
            typedef
                hpx::util::tuple1<
                    f0_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0>
            dataflow_frame_1(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_1::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_1::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_1::*f)
                        (Iter, Iter)
                        = &dataflow_frame_1::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_1::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_1::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
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
            Func
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
                Func
              , F0
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_1<
            Func
          , F0
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0)
    {
        typedef
            detail::dataflow_frame_1<
                Func
              , F0
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1>
        struct dataflow_frame_2
          : boost::enable_shared_from_this<
                dataflow_frame_2<
                    Func, F0 , F1
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type;
            typedef
                hpx::util::tuple2<
                    f0_type , f1_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1>
            dataflow_frame_2(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_2::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_2::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_2::*f)
                        (Iter, Iter)
                        = &dataflow_frame_2::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_2::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_2::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
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
            Func
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
                Func
              , F0 , F1
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_2<
            Func
          , F0 , F1
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1)
    {
        typedef
            detail::dataflow_frame_2<
                Func
              , F0 , F1
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2>
        struct dataflow_frame_3
          : boost::enable_shared_from_this<
                dataflow_frame_3<
                    Func, F0 , F1 , F2
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type;
            typedef
                hpx::util::tuple3<
                    f0_type , f1_type , f2_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2>
            dataflow_frame_3(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_3::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_3::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_3::*f)
                        (Iter, Iter)
                        = &dataflow_frame_3::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_3::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_3::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
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
            Func
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
                Func
              , F0 , F1 , F2
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_3<
            Func
          , F0 , F1 , F2
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2)
    {
        typedef
            detail::dataflow_frame_3<
                Func
              , F0 , F1 , F2
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3>
        struct dataflow_frame_4
          : boost::enable_shared_from_this<
                dataflow_frame_4<
                    Func, F0 , F1 , F2 , F3
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type;
            typedef
                hpx::util::tuple4<
                    f0_type , f1_type , f2_type , f3_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3>
            dataflow_frame_4(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_4::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_4::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_4::*f)
                        (Iter, Iter)
                        = &dataflow_frame_4::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_4::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_4::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
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
            Func
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
                Func
              , F0 , F1 , F2 , F3
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_4<
            Func
          , F0 , F1 , F2 , F3
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3)
    {
        typedef
            detail::dataflow_frame_4<
                Func
              , F0 , F1 , F2 , F3
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
        struct dataflow_frame_5
          : boost::enable_shared_from_this<
                dataflow_frame_5<
                    Func, F0 , F1 , F2 , F3 , F4
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type;
            typedef
                hpx::util::tuple5<
                    f0_type , f1_type , f2_type , f3_type , f4_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4>
            dataflow_frame_5(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_5::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_5::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_5::*f)
                        (Iter, Iter)
                        = &dataflow_frame_5::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_5::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_5::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
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
            Func
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
                Func
              , F0 , F1 , F2 , F3 , F4
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_5<
            Func
          , F0 , F1 , F2 , F3 , F4
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4)
    {
        typedef
            detail::dataflow_frame_5<
                Func
              , F0 , F1 , F2 , F3 , F4
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5>
        struct dataflow_frame_6
          : boost::enable_shared_from_this<
                dataflow_frame_6<
                    Func, F0 , F1 , F2 , F3 , F4 , F5
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type;
            typedef
                hpx::util::tuple6<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5>
            dataflow_frame_6(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_6::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_6::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_6::*f)
                        (Iter, Iter)
                        = &dataflow_frame_6::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_6::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_6::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_6<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5
    )
    {
        typedef
            detail::dataflow_frame_6<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_6<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5)
    {
        typedef
            detail::dataflow_frame_6<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6>
        struct dataflow_frame_7
          : boost::enable_shared_from_this<
                dataflow_frame_7<
                    Func, F0 , F1 , F2 , F3 , F4 , F5 , F6
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type;
            typedef
                hpx::util::tuple7<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6>
            dataflow_frame_7(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_7::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_7::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_7::*f)
                        (Iter, Iter)
                        = &dataflow_frame_7::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_7::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_7::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_7<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6
    )
    {
        typedef
            detail::dataflow_frame_7<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_7<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6)
    {
        typedef
            detail::dataflow_frame_7<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7>
        struct dataflow_frame_8
          : boost::enable_shared_from_this<
                dataflow_frame_8<
                    Func, F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type;
            typedef
                hpx::util::tuple8<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7>
            dataflow_frame_8(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6) , boost::forward<FF7>(f7)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_8::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_8::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_8::*f)
                        (Iter, Iter)
                        = &dataflow_frame_8::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_8::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_8::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_8<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7
    )
    {
        typedef
            detail::dataflow_frame_8<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_8<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7)
    {
        typedef
            detail::dataflow_frame_8<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8>
        struct dataflow_frame_9
          : boost::enable_shared_from_this<
                dataflow_frame_9<
                    Func, F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type;
            typedef
                hpx::util::tuple9<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8>
            dataflow_frame_9(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6) , boost::forward<FF7>(f7) , boost::forward<FF8>(f8)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_9::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_9::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_9::*f)
                        (Iter, Iter)
                        = &dataflow_frame_9::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_9::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_9::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_9<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8
    )
    {
        typedef
            detail::dataflow_frame_9<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_9<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8)
    {
        typedef
            detail::dataflow_frame_9<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9>
        struct dataflow_frame_10
          : boost::enable_shared_from_this<
                dataflow_frame_10<
                    Func, F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type;
            typedef
                hpx::util::tuple10<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9>
            dataflow_frame_10(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6) , boost::forward<FF7>(f7) , boost::forward<FF8>(f8) , boost::forward<FF9>(f9)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_10::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_10::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_10::*f)
                        (Iter, Iter)
                        = &dataflow_frame_10::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_10::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_10::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_10<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9
    )
    {
        typedef
            detail::dataflow_frame_10<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_10<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9)
    {
        typedef
            detail::dataflow_frame_10<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10>
        struct dataflow_frame_11
          : boost::enable_shared_from_this<
                dataflow_frame_11<
                    Func, F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type f10_type;
            typedef
                hpx::util::tuple11<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9 , typename FF10>
            dataflow_frame_11(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9 , BOOST_FWD_REF(FF10) f10
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6) , boost::forward<FF7>(f7) , boost::forward<FF8>(f8) , boost::forward<FF9>(f9) , boost::forward<FF10>(f10)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_11::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_11::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_11::*f)
                        (Iter, Iter)
                        = &dataflow_frame_11::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_11::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_11::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_11<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10
    )
    {
        typedef
            detail::dataflow_frame_11<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_11<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10)
    {
        typedef
            detail::dataflow_frame_11<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11>
        struct dataflow_frame_12
          : boost::enable_shared_from_this<
                dataflow_frame_12<
                    Func, F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type f10_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type f11_type;
            typedef
                hpx::util::tuple12<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9 , typename FF10 , typename FF11>
            dataflow_frame_12(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9 , BOOST_FWD_REF(FF10) f10 , BOOST_FWD_REF(FF11) f11
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6) , boost::forward<FF7>(f7) , boost::forward<FF8>(f8) , boost::forward<FF9>(f9) , boost::forward<FF10>(f10) , boost::forward<FF11>(f11)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_12::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_12::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_12::*f)
                        (Iter, Iter)
                        = &dataflow_frame_12::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_12::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_12::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_12<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11
    )
    {
        typedef
            detail::dataflow_frame_12<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_12<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11)
    {
        typedef
            detail::dataflow_frame_12<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12>
        struct dataflow_frame_13
          : boost::enable_shared_from_this<
                dataflow_frame_13<
                    Func, F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type f10_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type f11_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type f12_type;
            typedef
                hpx::util::tuple13<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type , f12_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type , f12_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9 , typename FF10 , typename FF11 , typename FF12>
            dataflow_frame_13(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9 , BOOST_FWD_REF(FF10) f10 , BOOST_FWD_REF(FF11) f11 , BOOST_FWD_REF(FF12) f12
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6) , boost::forward<FF7>(f7) , boost::forward<FF8>(f8) , boost::forward<FF9>(f9) , boost::forward<FF10>(f10) , boost::forward<FF11>(f11) , boost::forward<FF12>(f12)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_13::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_13::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_13::*f)
                        (Iter, Iter)
                        = &dataflow_frame_13::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_13::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_13::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_13<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12
    )
    {
        typedef
            detail::dataflow_frame_13<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_13<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12)
    {
        typedef
            detail::dataflow_frame_13<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13>
        struct dataflow_frame_14
          : boost::enable_shared_from_this<
                dataflow_frame_14<
                    Func, F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type f10_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type f11_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type f12_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F13 >::type >::type f13_type;
            typedef
                hpx::util::tuple14<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type , f12_type , f13_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type , f12_type , f13_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9 , typename FF10 , typename FF11 , typename FF12 , typename FF13>
            dataflow_frame_14(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9 , BOOST_FWD_REF(FF10) f10 , BOOST_FWD_REF(FF11) f11 , BOOST_FWD_REF(FF12) f12 , BOOST_FWD_REF(FF13) f13
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6) , boost::forward<FF7>(f7) , boost::forward<FF8>(f8) , boost::forward<FF9>(f9) , boost::forward<FF10>(f10) , boost::forward<FF11>(f11) , boost::forward<FF12>(f12) , boost::forward<FF13>(f13)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_14::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_14::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_14::*f)
                        (Iter, Iter)
                        = &dataflow_frame_14::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_14::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_14::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_14<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12 , BOOST_FWD_REF(F13) f13
    )
    {
        typedef
            detail::dataflow_frame_14<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_14<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12 , BOOST_FWD_REF(F13) f13)
    {
        typedef
            detail::dataflow_frame_14<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 )
            );
        frame->await();
        return frame->result_;
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13 , typename F14>
        struct dataflow_frame_15
          : boost::enable_shared_from_this<
                dataflow_frame_15<
                    Func, F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13 , F14
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            func_type func_;
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type f10_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type f11_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type f12_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F13 >::type >::type f13_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F14 >::type >::type f14_type;
            typedef
                hpx::util::tuple15<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type , f12_type , f13_type , f14_type
                >
                futures_type;
            futures_type futures_;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type , f12_type , f13_type , f14_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9 , typename FF10 , typename FF11 , typename FF12 , typename FF13 , typename FF14>
            dataflow_frame_15(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9 , BOOST_FWD_REF(FF10) f10 , BOOST_FWD_REF(FF11) f11 , BOOST_FWD_REF(FF12) f12 , BOOST_FWD_REF(FF13) f13 , BOOST_FWD_REF(FF14) f14
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6) , boost::forward<FF7>(f7) , boost::forward<FF8>(f8) , boost::forward<FF9>(f9) , boost::forward<FF10>(f10) , boost::forward<FF11>(f11) , boost::forward<FF12>(f12) , boost::forward<FF13>(f13) , boost::forward<FF14>(f14)
                )
              , policy_(policy)
            {}
            result_type execute() const
            {
                return boost::fusion::invoke(func_, futures_);
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_15::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                boost::fusion::invoke(func_, futures_);
                result_promise_.set_value();
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(reset_completion_handler_)
                {
                    reset_completion_handler_();
                    reset_completion_handler_.reset();
                }
                if(!result_.valid())
                {
                    result_ = hpx::async(
                        policy_
                      , hpx::util::bind(
                            &dataflow_frame_15::execute
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                result_promise_.set_value(
                    boost::fusion::invoke(func_, futures_)
                );
            }
            template <typename Iter>
            void await_range(Iter next, Iter end)
            {
                if(next == end) return;
                if(!next->ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_15::*f)
                        (Iter, Iter)
                        = &dataflow_frame_15::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            typename std::iterator_traits<
                                Iter
                            >::value_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(*next)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , boost::bind(
                                        f
                                      , this->shared_from_this()
                                      , next
                                      , end
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::bind(
                                f
                              , this->shared_from_this()
                              , next
                              , end
                            )
                        );
                    }
                    return;
                }
                await_range(++next, end);
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::true_)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                await_range(
                    boost::begin(boost::fusion::deref(iter))
                  , boost::end(boost::fusion::deref(iter))
                );
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await_next(Iter const& iter, IsVoid, boost::mpl::false_)
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
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_15::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &dataflow_frame_15::await_next;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
                        >::type
                        completed_callback_type;
                    completed_callback_type cb 
                        = boost::move(
                            hpx::lcos::detail::get_future_data(f_)
                            ->set_on_completed(
                                completed_callback_type()
                            )
                        );
                    
                    if(cb)
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                            compose_cb(
                                boost::move(cb)
                              , hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)->set_on_completed(
                            boost::move(
                              hpx::util::bind(
                                    f
                                  , this->shared_from_this()
                                  , iter
                                  , IsVoid()
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                  , IsVoid()
                );
            }
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const& iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::deref<Iter>::type
                    future_type;
                typedef typename traits::is_future_range<
                    future_type
                >::type is_range;
                await_next(iter, IsVoid(), is_range());
            }
            BOOST_FORCEINLINE void await()
            {
                typedef
                    typename boost::fusion::result_of::begin<futures_type>::type
                    begin_type;
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::bool_<
                        boost::is_same<begin_type, end_type>::value
                    >()
                  , boost::mpl::bool_<boost::is_same<void, result_type>::value>()
                );
            }
            BOOST_SCOPED_ENUM(launch) policy_;
            type result_;
            promise_result_type result_promise_;
            HPX_STD_FUNCTION<void()> reset_completion_handler_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13 , typename F14>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , detail::dataflow_frame_15<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13 , F14
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12 , BOOST_FWD_REF(F13) f13 , BOOST_FWD_REF(F14) f14
    )
    {
        typedef
            detail::dataflow_frame_15<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13 , F14
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 ) , boost::forward<F14>( f14 )
            );
        frame->await();
        return frame->result_;
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13 , typename F14>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_same<
            hpx::launch
          , typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Func
                >::type
            >::type
        >
      , detail::dataflow_frame_15<
            Func
          , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13 , F14
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12 , BOOST_FWD_REF(F13) f13 , BOOST_FWD_REF(F14) f14)
    {
        typedef
            detail::dataflow_frame_15<
                Func
              , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13 , F14
            >
            frame_type;
        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 ) , boost::forward<F14>( f14 )
            );
        frame->await();
        return frame->result_;
    }
}}}
