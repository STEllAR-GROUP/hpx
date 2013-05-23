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
            typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F0>::type >::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type
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
              , f0_( boost::forward<FF0>(f0) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0;
                }
                if(!f0_.ready())
                {
                    state_ = 1;
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    f0_.then(
                        boost::bind(
                            &dataflow_frame_1::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                
L0:
                f0_result_
                    = f0_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_
                        )
                    );
                }
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
            typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F0>::type >::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F1>::type >::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type
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
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1;
                }
                if(!f0_.ready())
                {
                    state_ = 1;
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    f0_.then(
                        boost::bind(
                            &dataflow_frame_2::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_2::await , this->shared_from_this() ) ); return; }
L1:
                f1_result_
                    = f1_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_
                        )
                    );
                }
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
            typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F0>::type >::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F1>::type >::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F2>::type >::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type
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
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2;
                }
                if(!f0_.ready())
                {
                    state_ = 1;
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    f0_.then(
                        boost::bind(
                            &dataflow_frame_3::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_3::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_3::await , this->shared_from_this() ) ); return; }
L2:
                f2_result_
                    = f2_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_
                        )
                    );
                }
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
            typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F0>::type >::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F1>::type >::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F2>::type >::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F3>::type >::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type
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
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3;
                }
                if(!f0_.ready())
                {
                    state_ = 1;
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    f0_.then(
                        boost::bind(
                            &dataflow_frame_4::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_4::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_4::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_4::await , this->shared_from_this() ) ); return; }
L3:
                f3_result_
                    = f3_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_
                        )
                    );
                }
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
            typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F0>::type >::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F1>::type >::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F2>::type >::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F3>::type >::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename boost::remove_const<typename hpx::util::detail::remove_reference<F4>::type >::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type , f4_result_type
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
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) ) , f4_( boost::forward<FF4>(f4) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3; case 5 : goto L4;
                }
                if(!f0_.ready())
                {
                    state_ = 1;
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }
                    f0_.then(
                        boost::bind(
                            &dataflow_frame_5::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_5::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_5::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_5::await , this->shared_from_this() ) ); return; } L3: f3_result_ = f3_.get(); if(!f4_.ready()) { state_ = 5; if(!result_.valid()) { result_ = result_promise_.get_future(); } f4_. then( boost::bind( &dataflow_frame_5::await , this->shared_from_this() ) ); return; }
L4:
                f4_result_
                    = f4_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_
                        )
                    );
                }
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
