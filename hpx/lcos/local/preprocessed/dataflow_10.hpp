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
                typename boost::result_of<
                    func_type(
                        f0_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0>
            dataflow_frame_1(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0)
                )
              , state_(0)
            {}
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                boost::fusion::invoke(func_, futures_);
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future();
                }
                else
                {
                    result_promise_.set_value();
                }
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future(
                            boost::fusion::invoke(func_, futures_)
                        );
                }
                else
                {
                    result_promise_.set_value(
                        boost::fusion::invoke(func_, futures_)
                    );
                }
            }
            
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const & iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                typedef
                    typename boost::fusion::result_of::end<futures_type>::type
                    end_type;
                if(!boost::fusion::deref(iter).ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_1::*f)
                        (Iter const &, boost::mpl::false_, IsVoid)
                        = &dataflow_frame_1::await;
                    boost::fusion::deref(iter).then(
                        boost::bind(
                            f
                          , this->shared_from_this()
                          , iter
                          , boost::mpl::false_()
                          , IsVoid()
                        )
                    );
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , typename boost::is_same<next_type, end_type>::type()
                  , IsVoid()
                );
            }
            BOOST_FORCEINLINE void await()
            {
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::false_()
                  , typename boost::is_same<void, result_type>::type()
                );
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_1<
        Func
      , F0
    >::future_result_type
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
                boost::forward<Func>(func)
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
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1>
            dataflow_frame_2(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1)
                )
              , state_(0)
            {}
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                boost::fusion::invoke(func_, futures_);
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future();
                }
                else
                {
                    result_promise_.set_value();
                }
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future(
                            boost::fusion::invoke(func_, futures_)
                        );
                }
                else
                {
                    result_promise_.set_value(
                        boost::fusion::invoke(func_, futures_)
                    );
                }
            }
            
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const & iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                typedef
                    typename boost::fusion::result_of::end<futures_type>::type
                    end_type;
                if(!boost::fusion::deref(iter).ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_2::*f)
                        (Iter const &, boost::mpl::false_, IsVoid)
                        = &dataflow_frame_2::await;
                    boost::fusion::deref(iter).then(
                        boost::bind(
                            f
                          , this->shared_from_this()
                          , iter
                          , boost::mpl::false_()
                          , IsVoid()
                        )
                    );
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , typename boost::is_same<next_type, end_type>::type()
                  , IsVoid()
                );
            }
            BOOST_FORCEINLINE void await()
            {
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::false_()
                  , typename boost::is_same<void, result_type>::type()
                );
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_2<
        Func
      , F0 , F1
    >::future_result_type
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
                boost::forward<Func>(func)
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
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2>
            dataflow_frame_3(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2)
                )
              , state_(0)
            {}
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                boost::fusion::invoke(func_, futures_);
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future();
                }
                else
                {
                    result_promise_.set_value();
                }
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future(
                            boost::fusion::invoke(func_, futures_)
                        );
                }
                else
                {
                    result_promise_.set_value(
                        boost::fusion::invoke(func_, futures_)
                    );
                }
            }
            
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const & iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                typedef
                    typename boost::fusion::result_of::end<futures_type>::type
                    end_type;
                if(!boost::fusion::deref(iter).ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_3::*f)
                        (Iter const &, boost::mpl::false_, IsVoid)
                        = &dataflow_frame_3::await;
                    boost::fusion::deref(iter).then(
                        boost::bind(
                            f
                          , this->shared_from_this()
                          , iter
                          , boost::mpl::false_()
                          , IsVoid()
                        )
                    );
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , typename boost::is_same<next_type, end_type>::type()
                  , IsVoid()
                );
            }
            BOOST_FORCEINLINE void await()
            {
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::false_()
                  , typename boost::is_same<void, result_type>::type()
                );
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_3<
        Func
      , F0 , F1 , F2
    >::future_result_type
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
                boost::forward<Func>(func)
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
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3>
            dataflow_frame_4(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3)
                )
              , state_(0)
            {}
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                boost::fusion::invoke(func_, futures_);
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future();
                }
                else
                {
                    result_promise_.set_value();
                }
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future(
                            boost::fusion::invoke(func_, futures_)
                        );
                }
                else
                {
                    result_promise_.set_value(
                        boost::fusion::invoke(func_, futures_)
                    );
                }
            }
            
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const & iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                typedef
                    typename boost::fusion::result_of::end<futures_type>::type
                    end_type;
                if(!boost::fusion::deref(iter).ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_4::*f)
                        (Iter const &, boost::mpl::false_, IsVoid)
                        = &dataflow_frame_4::await;
                    boost::fusion::deref(iter).then(
                        boost::bind(
                            f
                          , this->shared_from_this()
                          , iter
                          , boost::mpl::false_()
                          , IsVoid()
                        )
                    );
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , typename boost::is_same<next_type, end_type>::type()
                  , IsVoid()
                );
            }
            BOOST_FORCEINLINE void await()
            {
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::false_()
                  , typename boost::is_same<void, result_type>::type()
                );
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_4<
        Func
      , F0 , F1 , F2 , F3
    >::future_result_type
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
                boost::forward<Func>(func)
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
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4>
            dataflow_frame_5(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4)
                )
              , state_(0)
            {}
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                boost::fusion::invoke(func_, futures_);
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future();
                }
                else
                {
                    result_promise_.set_value();
                }
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future(
                            boost::fusion::invoke(func_, futures_)
                        );
                }
                else
                {
                    result_promise_.set_value(
                        boost::fusion::invoke(func_, futures_)
                    );
                }
            }
            
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const & iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                typedef
                    typename boost::fusion::result_of::end<futures_type>::type
                    end_type;
                if(!boost::fusion::deref(iter).ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_5::*f)
                        (Iter const &, boost::mpl::false_, IsVoid)
                        = &dataflow_frame_5::await;
                    boost::fusion::deref(iter).then(
                        boost::bind(
                            f
                          , this->shared_from_this()
                          , iter
                          , boost::mpl::false_()
                          , IsVoid()
                        )
                    );
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , typename boost::is_same<next_type, end_type>::type()
                  , IsVoid()
                );
            }
            BOOST_FORCEINLINE void await()
            {
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::false_()
                  , typename boost::is_same<void, result_type>::type()
                );
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_5<
        Func
      , F0 , F1 , F2 , F3 , F4
    >::future_result_type
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
                boost::forward<Func>(func)
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
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5>
            dataflow_frame_6(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5)
                )
              , state_(0)
            {}
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                boost::fusion::invoke(func_, futures_);
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future();
                }
                else
                {
                    result_promise_.set_value();
                }
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future(
                            boost::fusion::invoke(func_, futures_)
                        );
                }
                else
                {
                    result_promise_.set_value(
                        boost::fusion::invoke(func_, futures_)
                    );
                }
            }
            
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const & iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                typedef
                    typename boost::fusion::result_of::end<futures_type>::type
                    end_type;
                if(!boost::fusion::deref(iter).ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_6::*f)
                        (Iter const &, boost::mpl::false_, IsVoid)
                        = &dataflow_frame_6::await;
                    boost::fusion::deref(iter).then(
                        boost::bind(
                            f
                          , this->shared_from_this()
                          , iter
                          , boost::mpl::false_()
                          , IsVoid()
                        )
                    );
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , typename boost::is_same<next_type, end_type>::type()
                  , IsVoid()
                );
            }
            BOOST_FORCEINLINE void await()
            {
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::false_()
                  , typename boost::is_same<void, result_type>::type()
                );
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_6<
        Func
      , F0 , F1 , F2 , F3 , F4 , F5
    >::future_result_type
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
                boost::forward<Func>(func)
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
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6>
            dataflow_frame_7(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6)
                )
              , state_(0)
            {}
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                boost::fusion::invoke(func_, futures_);
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future();
                }
                else
                {
                    result_promise_.set_value();
                }
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future(
                            boost::fusion::invoke(func_, futures_)
                        );
                }
                else
                {
                    result_promise_.set_value(
                        boost::fusion::invoke(func_, futures_)
                    );
                }
            }
            
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const & iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                typedef
                    typename boost::fusion::result_of::end<futures_type>::type
                    end_type;
                if(!boost::fusion::deref(iter).ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_7::*f)
                        (Iter const &, boost::mpl::false_, IsVoid)
                        = &dataflow_frame_7::await;
                    boost::fusion::deref(iter).then(
                        boost::bind(
                            f
                          , this->shared_from_this()
                          , iter
                          , boost::mpl::false_()
                          , IsVoid()
                        )
                    );
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , typename boost::is_same<next_type, end_type>::type()
                  , IsVoid()
                );
            }
            BOOST_FORCEINLINE void await()
            {
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::false_()
                  , typename boost::is_same<void, result_type>::type()
                );
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_7<
        Func
      , F0 , F1 , F2 , F3 , F4 , F5 , F6
    >::future_result_type
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
                boost::forward<Func>(func)
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
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7>
            dataflow_frame_8(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6) , boost::forward<FF7>(f7)
                )
              , state_(0)
            {}
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                boost::fusion::invoke(func_, futures_);
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future();
                }
                else
                {
                    result_promise_.set_value();
                }
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future(
                            boost::fusion::invoke(func_, futures_)
                        );
                }
                else
                {
                    result_promise_.set_value(
                        boost::fusion::invoke(func_, futures_)
                    );
                }
            }
            
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const & iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                typedef
                    typename boost::fusion::result_of::end<futures_type>::type
                    end_type;
                if(!boost::fusion::deref(iter).ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_8::*f)
                        (Iter const &, boost::mpl::false_, IsVoid)
                        = &dataflow_frame_8::await;
                    boost::fusion::deref(iter).then(
                        boost::bind(
                            f
                          , this->shared_from_this()
                          , iter
                          , boost::mpl::false_()
                          , IsVoid()
                        )
                    );
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , typename boost::is_same<next_type, end_type>::type()
                  , IsVoid()
                );
            }
            BOOST_FORCEINLINE void await()
            {
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::false_()
                  , typename boost::is_same<void, result_type>::type()
                );
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_8<
        Func
      , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7
    >::future_result_type
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
                boost::forward<Func>(func)
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
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8>
            dataflow_frame_9(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6) , boost::forward<FF7>(f7) , boost::forward<FF8>(f8)
                )
              , state_(0)
            {}
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                boost::fusion::invoke(func_, futures_);
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future();
                }
                else
                {
                    result_promise_.set_value();
                }
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future(
                            boost::fusion::invoke(func_, futures_)
                        );
                }
                else
                {
                    result_promise_.set_value(
                        boost::fusion::invoke(func_, futures_)
                    );
                }
            }
            
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const & iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                typedef
                    typename boost::fusion::result_of::end<futures_type>::type
                    end_type;
                if(!boost::fusion::deref(iter).ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_9::*f)
                        (Iter const &, boost::mpl::false_, IsVoid)
                        = &dataflow_frame_9::await;
                    boost::fusion::deref(iter).then(
                        boost::bind(
                            f
                          , this->shared_from_this()
                          , iter
                          , boost::mpl::false_()
                          , IsVoid()
                        )
                    );
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , typename boost::is_same<next_type, end_type>::type()
                  , IsVoid()
                );
            }
            BOOST_FORCEINLINE void await()
            {
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::false_()
                  , typename boost::is_same<void, result_type>::type()
                );
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_9<
        Func
      , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8
    >::future_result_type
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
                boost::forward<Func>(func)
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
                typename boost::result_of<
                    func_type(
                        f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9>
            dataflow_frame_10(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    boost::forward<FF0>(f0) , boost::forward<FF1>(f1) , boost::forward<FF2>(f2) , boost::forward<FF3>(f3) , boost::forward<FF4>(f4) , boost::forward<FF5>(f5) , boost::forward<FF6>(f6) , boost::forward<FF7>(f7) , boost::forward<FF8>(f8) , boost::forward<FF9>(f9)
                )
              , state_(0)
            {}
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::true_
            )
            {
                boost::fusion::invoke(func_, futures_);
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future();
                }
                else
                {
                    result_promise_.set_value();
                }
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_, boost::mpl::false_
            )
            {
                if(!result_.valid())
                {
                    result_
                        = hpx::make_ready_future(
                            boost::fusion::invoke(func_, futures_)
                        );
                }
                else
                {
                    result_promise_.set_value(
                        boost::fusion::invoke(func_, futures_)
                    );
                }
            }
            
            template <typename Iter, typename IsVoid>
            BOOST_FORCEINLINE
            void await(Iter const & iter, boost::mpl::false_, IsVoid)
            {
                typedef
                    typename boost::fusion::result_of::next<Iter>::type
                    next_type;
                typedef
                    typename boost::fusion::result_of::end<futures_type>::type
                    end_type;
                if(!boost::fusion::deref(iter).ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    void (dataflow_frame_10::*f)
                        (Iter const &, boost::mpl::false_, IsVoid)
                        = &dataflow_frame_10::await;
                    boost::fusion::deref(iter).then(
                        boost::bind(
                            f
                          , this->shared_from_this()
                          , iter
                          , boost::mpl::false_()
                          , IsVoid()
                        )
                    );
                    return;
                }
                await(
                    boost::fusion::next(iter)
                  , typename boost::is_same<next_type, end_type>::type()
                  , IsVoid()
                );
            }
            BOOST_FORCEINLINE void await()
            {
                await(
                    boost::fusion::begin(futures_)
                  , boost::mpl::false_()
                  , typename boost::is_same<void, result_type>::type()
                );
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_10<
        Func
      , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9
    >::future_result_type
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
                boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 )
            );
        frame->await();
        return frame->result_;
    }
}}}
