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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_;
            
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_;
            
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_;
            
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_;
            
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename hpx::util::detail::remove_reference<F4>::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_;
            
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename hpx::util::detail::remove_reference<F4>::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_; typedef typename hpx::util::detail::remove_reference<F5>::type f5_type; typedef typename future_traits< f5_type >::value_type f5_result_type; f5_type f5_; f5_result_type f5_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type , f4_result_type , f5_result_type
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
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) ) , f4_( boost::forward<FF4>(f4) ) , f5_( boost::forward<FF5>(f5) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3; case 5 : goto L4; case 6 : goto L5;
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
                            &dataflow_frame_6::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_6::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_6::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_6::await , this->shared_from_this() ) ); return; } L3: f3_result_ = f3_.get(); if(!f4_.ready()) { state_ = 5; if(!result_.valid()) { result_ = result_promise_.get_future(); } f4_. then( boost::bind( &dataflow_frame_6::await , this->shared_from_this() ) ); return; } L4: f4_result_ = f4_.get(); if(!f5_.ready()) { state_ = 6; if(!result_.valid()) { result_ = result_promise_.get_future(); } f5_. then( boost::bind( &dataflow_frame_6::await , this->shared_from_this() ) ); return; }
L5:
                f5_result_
                    = f5_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_
                        )
                    );
                }
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename hpx::util::detail::remove_reference<F4>::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_; typedef typename hpx::util::detail::remove_reference<F5>::type f5_type; typedef typename future_traits< f5_type >::value_type f5_result_type; f5_type f5_; f5_result_type f5_result_; typedef typename hpx::util::detail::remove_reference<F6>::type f6_type; typedef typename future_traits< f6_type >::value_type f6_result_type; f6_type f6_; f6_result_type f6_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type , f4_result_type , f5_result_type , f6_result_type
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
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) ) , f4_( boost::forward<FF4>(f4) ) , f5_( boost::forward<FF5>(f5) ) , f6_( boost::forward<FF6>(f6) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3; case 5 : goto L4; case 6 : goto L5; case 7 : goto L6;
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
                            &dataflow_frame_7::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_7::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_7::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_7::await , this->shared_from_this() ) ); return; } L3: f3_result_ = f3_.get(); if(!f4_.ready()) { state_ = 5; if(!result_.valid()) { result_ = result_promise_.get_future(); } f4_. then( boost::bind( &dataflow_frame_7::await , this->shared_from_this() ) ); return; } L4: f4_result_ = f4_.get(); if(!f5_.ready()) { state_ = 6; if(!result_.valid()) { result_ = result_promise_.get_future(); } f5_. then( boost::bind( &dataflow_frame_7::await , this->shared_from_this() ) ); return; } L5: f5_result_ = f5_.get(); if(!f6_.ready()) { state_ = 7; if(!result_.valid()) { result_ = result_promise_.get_future(); } f6_. then( boost::bind( &dataflow_frame_7::await , this->shared_from_this() ) ); return; }
L6:
                f6_result_
                    = f6_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_
                        )
                    );
                }
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename hpx::util::detail::remove_reference<F4>::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_; typedef typename hpx::util::detail::remove_reference<F5>::type f5_type; typedef typename future_traits< f5_type >::value_type f5_result_type; f5_type f5_; f5_result_type f5_result_; typedef typename hpx::util::detail::remove_reference<F6>::type f6_type; typedef typename future_traits< f6_type >::value_type f6_result_type; f6_type f6_; f6_result_type f6_result_; typedef typename hpx::util::detail::remove_reference<F7>::type f7_type; typedef typename future_traits< f7_type >::value_type f7_result_type; f7_type f7_; f7_result_type f7_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type , f4_result_type , f5_result_type , f6_result_type , f7_result_type
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
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) ) , f4_( boost::forward<FF4>(f4) ) , f5_( boost::forward<FF5>(f5) ) , f6_( boost::forward<FF6>(f6) ) , f7_( boost::forward<FF7>(f7) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3; case 5 : goto L4; case 6 : goto L5; case 7 : goto L6; case 8 : goto L7;
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
                            &dataflow_frame_8::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_8::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_8::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_8::await , this->shared_from_this() ) ); return; } L3: f3_result_ = f3_.get(); if(!f4_.ready()) { state_ = 5; if(!result_.valid()) { result_ = result_promise_.get_future(); } f4_. then( boost::bind( &dataflow_frame_8::await , this->shared_from_this() ) ); return; } L4: f4_result_ = f4_.get(); if(!f5_.ready()) { state_ = 6; if(!result_.valid()) { result_ = result_promise_.get_future(); } f5_. then( boost::bind( &dataflow_frame_8::await , this->shared_from_this() ) ); return; } L5: f5_result_ = f5_.get(); if(!f6_.ready()) { state_ = 7; if(!result_.valid()) { result_ = result_promise_.get_future(); } f6_. then( boost::bind( &dataflow_frame_8::await , this->shared_from_this() ) ); return; } L6: f6_result_ = f6_.get(); if(!f7_.ready()) { state_ = 8; if(!result_.valid()) { result_ = result_promise_.get_future(); } f7_. then( boost::bind( &dataflow_frame_8::await , this->shared_from_this() ) ); return; }
L7:
                f7_result_
                    = f7_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_
                        )
                    );
                }
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename hpx::util::detail::remove_reference<F4>::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_; typedef typename hpx::util::detail::remove_reference<F5>::type f5_type; typedef typename future_traits< f5_type >::value_type f5_result_type; f5_type f5_; f5_result_type f5_result_; typedef typename hpx::util::detail::remove_reference<F6>::type f6_type; typedef typename future_traits< f6_type >::value_type f6_result_type; f6_type f6_; f6_result_type f6_result_; typedef typename hpx::util::detail::remove_reference<F7>::type f7_type; typedef typename future_traits< f7_type >::value_type f7_result_type; f7_type f7_; f7_result_type f7_result_; typedef typename hpx::util::detail::remove_reference<F8>::type f8_type; typedef typename future_traits< f8_type >::value_type f8_result_type; f8_type f8_; f8_result_type f8_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type , f4_result_type , f5_result_type , f6_result_type , f7_result_type , f8_result_type
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
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) ) , f4_( boost::forward<FF4>(f4) ) , f5_( boost::forward<FF5>(f5) ) , f6_( boost::forward<FF6>(f6) ) , f7_( boost::forward<FF7>(f7) ) , f8_( boost::forward<FF8>(f8) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3; case 5 : goto L4; case 6 : goto L5; case 7 : goto L6; case 8 : goto L7; case 9 : goto L8;
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
                            &dataflow_frame_9::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_9::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_9::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_9::await , this->shared_from_this() ) ); return; } L3: f3_result_ = f3_.get(); if(!f4_.ready()) { state_ = 5; if(!result_.valid()) { result_ = result_promise_.get_future(); } f4_. then( boost::bind( &dataflow_frame_9::await , this->shared_from_this() ) ); return; } L4: f4_result_ = f4_.get(); if(!f5_.ready()) { state_ = 6; if(!result_.valid()) { result_ = result_promise_.get_future(); } f5_. then( boost::bind( &dataflow_frame_9::await , this->shared_from_this() ) ); return; } L5: f5_result_ = f5_.get(); if(!f6_.ready()) { state_ = 7; if(!result_.valid()) { result_ = result_promise_.get_future(); } f6_. then( boost::bind( &dataflow_frame_9::await , this->shared_from_this() ) ); return; } L6: f6_result_ = f6_.get(); if(!f7_.ready()) { state_ = 8; if(!result_.valid()) { result_ = result_promise_.get_future(); } f7_. then( boost::bind( &dataflow_frame_9::await , this->shared_from_this() ) ); return; } L7: f7_result_ = f7_.get(); if(!f8_.ready()) { state_ = 9; if(!result_.valid()) { result_ = result_promise_.get_future(); } f8_. then( boost::bind( &dataflow_frame_9::await , this->shared_from_this() ) ); return; }
L8:
                f8_result_
                    = f8_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_
                        )
                    );
                }
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename hpx::util::detail::remove_reference<F4>::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_; typedef typename hpx::util::detail::remove_reference<F5>::type f5_type; typedef typename future_traits< f5_type >::value_type f5_result_type; f5_type f5_; f5_result_type f5_result_; typedef typename hpx::util::detail::remove_reference<F6>::type f6_type; typedef typename future_traits< f6_type >::value_type f6_result_type; f6_type f6_; f6_result_type f6_result_; typedef typename hpx::util::detail::remove_reference<F7>::type f7_type; typedef typename future_traits< f7_type >::value_type f7_result_type; f7_type f7_; f7_result_type f7_result_; typedef typename hpx::util::detail::remove_reference<F8>::type f8_type; typedef typename future_traits< f8_type >::value_type f8_result_type; f8_type f8_; f8_result_type f8_result_; typedef typename hpx::util::detail::remove_reference<F9>::type f9_type; typedef typename future_traits< f9_type >::value_type f9_result_type; f9_type f9_; f9_result_type f9_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type , f4_result_type , f5_result_type , f6_result_type , f7_result_type , f8_result_type , f9_result_type
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
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) ) , f4_( boost::forward<FF4>(f4) ) , f5_( boost::forward<FF5>(f5) ) , f6_( boost::forward<FF6>(f6) ) , f7_( boost::forward<FF7>(f7) ) , f8_( boost::forward<FF8>(f8) ) , f9_( boost::forward<FF9>(f9) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3; case 5 : goto L4; case 6 : goto L5; case 7 : goto L6; case 8 : goto L7; case 9 : goto L8; case 10 : goto L9;
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
                            &dataflow_frame_10::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_10::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_10::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_10::await , this->shared_from_this() ) ); return; } L3: f3_result_ = f3_.get(); if(!f4_.ready()) { state_ = 5; if(!result_.valid()) { result_ = result_promise_.get_future(); } f4_. then( boost::bind( &dataflow_frame_10::await , this->shared_from_this() ) ); return; } L4: f4_result_ = f4_.get(); if(!f5_.ready()) { state_ = 6; if(!result_.valid()) { result_ = result_promise_.get_future(); } f5_. then( boost::bind( &dataflow_frame_10::await , this->shared_from_this() ) ); return; } L5: f5_result_ = f5_.get(); if(!f6_.ready()) { state_ = 7; if(!result_.valid()) { result_ = result_promise_.get_future(); } f6_. then( boost::bind( &dataflow_frame_10::await , this->shared_from_this() ) ); return; } L6: f6_result_ = f6_.get(); if(!f7_.ready()) { state_ = 8; if(!result_.valid()) { result_ = result_promise_.get_future(); } f7_. then( boost::bind( &dataflow_frame_10::await , this->shared_from_this() ) ); return; } L7: f7_result_ = f7_.get(); if(!f8_.ready()) { state_ = 9; if(!result_.valid()) { result_ = result_promise_.get_future(); } f8_. then( boost::bind( &dataflow_frame_10::await , this->shared_from_this() ) ); return; } L8: f8_result_ = f8_.get(); if(!f9_.ready()) { state_ = 10; if(!result_.valid()) { result_ = result_promise_.get_future(); } f9_. then( boost::bind( &dataflow_frame_10::await , this->shared_from_this() ) ); return; }
L9:
                f9_result_
                    = f9_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_
                        )
                    );
                }
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename hpx::util::detail::remove_reference<F4>::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_; typedef typename hpx::util::detail::remove_reference<F5>::type f5_type; typedef typename future_traits< f5_type >::value_type f5_result_type; f5_type f5_; f5_result_type f5_result_; typedef typename hpx::util::detail::remove_reference<F6>::type f6_type; typedef typename future_traits< f6_type >::value_type f6_result_type; f6_type f6_; f6_result_type f6_result_; typedef typename hpx::util::detail::remove_reference<F7>::type f7_type; typedef typename future_traits< f7_type >::value_type f7_result_type; f7_type f7_; f7_result_type f7_result_; typedef typename hpx::util::detail::remove_reference<F8>::type f8_type; typedef typename future_traits< f8_type >::value_type f8_result_type; f8_type f8_; f8_result_type f8_result_; typedef typename hpx::util::detail::remove_reference<F9>::type f9_type; typedef typename future_traits< f9_type >::value_type f9_result_type; f9_type f9_; f9_result_type f9_result_; typedef typename hpx::util::detail::remove_reference<F10>::type f10_type; typedef typename future_traits< f10_type >::value_type f10_result_type; f10_type f10_; f10_result_type f10_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type , f4_result_type , f5_result_type , f6_result_type , f7_result_type , f8_result_type , f9_result_type , f10_result_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9 , typename FF10>
            dataflow_frame_11(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9 , BOOST_FWD_REF(FF10) f10
            )
              : func_(boost::forward<FFunc>(func))
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) ) , f4_( boost::forward<FF4>(f4) ) , f5_( boost::forward<FF5>(f5) ) , f6_( boost::forward<FF6>(f6) ) , f7_( boost::forward<FF7>(f7) ) , f8_( boost::forward<FF8>(f8) ) , f9_( boost::forward<FF9>(f9) ) , f10_( boost::forward<FF10>(f10) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3; case 5 : goto L4; case 6 : goto L5; case 7 : goto L6; case 8 : goto L7; case 9 : goto L8; case 10 : goto L9; case 11 : goto L10;
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
                            &dataflow_frame_11::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_11::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_11::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_11::await , this->shared_from_this() ) ); return; } L3: f3_result_ = f3_.get(); if(!f4_.ready()) { state_ = 5; if(!result_.valid()) { result_ = result_promise_.get_future(); } f4_. then( boost::bind( &dataflow_frame_11::await , this->shared_from_this() ) ); return; } L4: f4_result_ = f4_.get(); if(!f5_.ready()) { state_ = 6; if(!result_.valid()) { result_ = result_promise_.get_future(); } f5_. then( boost::bind( &dataflow_frame_11::await , this->shared_from_this() ) ); return; } L5: f5_result_ = f5_.get(); if(!f6_.ready()) { state_ = 7; if(!result_.valid()) { result_ = result_promise_.get_future(); } f6_. then( boost::bind( &dataflow_frame_11::await , this->shared_from_this() ) ); return; } L6: f6_result_ = f6_.get(); if(!f7_.ready()) { state_ = 8; if(!result_.valid()) { result_ = result_promise_.get_future(); } f7_. then( boost::bind( &dataflow_frame_11::await , this->shared_from_this() ) ); return; } L7: f7_result_ = f7_.get(); if(!f8_.ready()) { state_ = 9; if(!result_.valid()) { result_ = result_promise_.get_future(); } f8_. then( boost::bind( &dataflow_frame_11::await , this->shared_from_this() ) ); return; } L8: f8_result_ = f8_.get(); if(!f9_.ready()) { state_ = 10; if(!result_.valid()) { result_ = result_promise_.get_future(); } f9_. then( boost::bind( &dataflow_frame_11::await , this->shared_from_this() ) ); return; } L9: f9_result_ = f9_.get(); if(!f10_.ready()) { state_ = 11; if(!result_.valid()) { result_ = result_promise_.get_future(); } f10_. then( boost::bind( &dataflow_frame_11::await , this->shared_from_this() ) ); return; }
L10:
                f10_result_
                    = f10_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_ , f10_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_ , f10_result_
                        )
                    );
                }
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_11<
        Func
      , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10
    >::future_result_type
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
                boost::forward<Func>(func)
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename hpx::util::detail::remove_reference<F4>::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_; typedef typename hpx::util::detail::remove_reference<F5>::type f5_type; typedef typename future_traits< f5_type >::value_type f5_result_type; f5_type f5_; f5_result_type f5_result_; typedef typename hpx::util::detail::remove_reference<F6>::type f6_type; typedef typename future_traits< f6_type >::value_type f6_result_type; f6_type f6_; f6_result_type f6_result_; typedef typename hpx::util::detail::remove_reference<F7>::type f7_type; typedef typename future_traits< f7_type >::value_type f7_result_type; f7_type f7_; f7_result_type f7_result_; typedef typename hpx::util::detail::remove_reference<F8>::type f8_type; typedef typename future_traits< f8_type >::value_type f8_result_type; f8_type f8_; f8_result_type f8_result_; typedef typename hpx::util::detail::remove_reference<F9>::type f9_type; typedef typename future_traits< f9_type >::value_type f9_result_type; f9_type f9_; f9_result_type f9_result_; typedef typename hpx::util::detail::remove_reference<F10>::type f10_type; typedef typename future_traits< f10_type >::value_type f10_result_type; f10_type f10_; f10_result_type f10_result_; typedef typename hpx::util::detail::remove_reference<F11>::type f11_type; typedef typename future_traits< f11_type >::value_type f11_result_type; f11_type f11_; f11_result_type f11_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type , f4_result_type , f5_result_type , f6_result_type , f7_result_type , f8_result_type , f9_result_type , f10_result_type , f11_result_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9 , typename FF10 , typename FF11>
            dataflow_frame_12(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9 , BOOST_FWD_REF(FF10) f10 , BOOST_FWD_REF(FF11) f11
            )
              : func_(boost::forward<FFunc>(func))
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) ) , f4_( boost::forward<FF4>(f4) ) , f5_( boost::forward<FF5>(f5) ) , f6_( boost::forward<FF6>(f6) ) , f7_( boost::forward<FF7>(f7) ) , f8_( boost::forward<FF8>(f8) ) , f9_( boost::forward<FF9>(f9) ) , f10_( boost::forward<FF10>(f10) ) , f11_( boost::forward<FF11>(f11) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3; case 5 : goto L4; case 6 : goto L5; case 7 : goto L6; case 8 : goto L7; case 9 : goto L8; case 10 : goto L9; case 11 : goto L10; case 12 : goto L11;
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
                            &dataflow_frame_12::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_12::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_12::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_12::await , this->shared_from_this() ) ); return; } L3: f3_result_ = f3_.get(); if(!f4_.ready()) { state_ = 5; if(!result_.valid()) { result_ = result_promise_.get_future(); } f4_. then( boost::bind( &dataflow_frame_12::await , this->shared_from_this() ) ); return; } L4: f4_result_ = f4_.get(); if(!f5_.ready()) { state_ = 6; if(!result_.valid()) { result_ = result_promise_.get_future(); } f5_. then( boost::bind( &dataflow_frame_12::await , this->shared_from_this() ) ); return; } L5: f5_result_ = f5_.get(); if(!f6_.ready()) { state_ = 7; if(!result_.valid()) { result_ = result_promise_.get_future(); } f6_. then( boost::bind( &dataflow_frame_12::await , this->shared_from_this() ) ); return; } L6: f6_result_ = f6_.get(); if(!f7_.ready()) { state_ = 8; if(!result_.valid()) { result_ = result_promise_.get_future(); } f7_. then( boost::bind( &dataflow_frame_12::await , this->shared_from_this() ) ); return; } L7: f7_result_ = f7_.get(); if(!f8_.ready()) { state_ = 9; if(!result_.valid()) { result_ = result_promise_.get_future(); } f8_. then( boost::bind( &dataflow_frame_12::await , this->shared_from_this() ) ); return; } L8: f8_result_ = f8_.get(); if(!f9_.ready()) { state_ = 10; if(!result_.valid()) { result_ = result_promise_.get_future(); } f9_. then( boost::bind( &dataflow_frame_12::await , this->shared_from_this() ) ); return; } L9: f9_result_ = f9_.get(); if(!f10_.ready()) { state_ = 11; if(!result_.valid()) { result_ = result_promise_.get_future(); } f10_. then( boost::bind( &dataflow_frame_12::await , this->shared_from_this() ) ); return; } L10: f10_result_ = f10_.get(); if(!f11_.ready()) { state_ = 12; if(!result_.valid()) { result_ = result_promise_.get_future(); } f11_. then( boost::bind( &dataflow_frame_12::await , this->shared_from_this() ) ); return; }
L11:
                f11_result_
                    = f11_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_ , f10_result_ , f11_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_ , f10_result_ , f11_result_
                        )
                    );
                }
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_12<
        Func
      , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11
    >::future_result_type
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
                boost::forward<Func>(func)
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename hpx::util::detail::remove_reference<F4>::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_; typedef typename hpx::util::detail::remove_reference<F5>::type f5_type; typedef typename future_traits< f5_type >::value_type f5_result_type; f5_type f5_; f5_result_type f5_result_; typedef typename hpx::util::detail::remove_reference<F6>::type f6_type; typedef typename future_traits< f6_type >::value_type f6_result_type; f6_type f6_; f6_result_type f6_result_; typedef typename hpx::util::detail::remove_reference<F7>::type f7_type; typedef typename future_traits< f7_type >::value_type f7_result_type; f7_type f7_; f7_result_type f7_result_; typedef typename hpx::util::detail::remove_reference<F8>::type f8_type; typedef typename future_traits< f8_type >::value_type f8_result_type; f8_type f8_; f8_result_type f8_result_; typedef typename hpx::util::detail::remove_reference<F9>::type f9_type; typedef typename future_traits< f9_type >::value_type f9_result_type; f9_type f9_; f9_result_type f9_result_; typedef typename hpx::util::detail::remove_reference<F10>::type f10_type; typedef typename future_traits< f10_type >::value_type f10_result_type; f10_type f10_; f10_result_type f10_result_; typedef typename hpx::util::detail::remove_reference<F11>::type f11_type; typedef typename future_traits< f11_type >::value_type f11_result_type; f11_type f11_; f11_result_type f11_result_; typedef typename hpx::util::detail::remove_reference<F12>::type f12_type; typedef typename future_traits< f12_type >::value_type f12_result_type; f12_type f12_; f12_result_type f12_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type , f4_result_type , f5_result_type , f6_result_type , f7_result_type , f8_result_type , f9_result_type , f10_result_type , f11_result_type , f12_result_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9 , typename FF10 , typename FF11 , typename FF12>
            dataflow_frame_13(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9 , BOOST_FWD_REF(FF10) f10 , BOOST_FWD_REF(FF11) f11 , BOOST_FWD_REF(FF12) f12
            )
              : func_(boost::forward<FFunc>(func))
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) ) , f4_( boost::forward<FF4>(f4) ) , f5_( boost::forward<FF5>(f5) ) , f6_( boost::forward<FF6>(f6) ) , f7_( boost::forward<FF7>(f7) ) , f8_( boost::forward<FF8>(f8) ) , f9_( boost::forward<FF9>(f9) ) , f10_( boost::forward<FF10>(f10) ) , f11_( boost::forward<FF11>(f11) ) , f12_( boost::forward<FF12>(f12) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3; case 5 : goto L4; case 6 : goto L5; case 7 : goto L6; case 8 : goto L7; case 9 : goto L8; case 10 : goto L9; case 11 : goto L10; case 12 : goto L11; case 13 : goto L12;
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
                            &dataflow_frame_13::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; } L3: f3_result_ = f3_.get(); if(!f4_.ready()) { state_ = 5; if(!result_.valid()) { result_ = result_promise_.get_future(); } f4_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; } L4: f4_result_ = f4_.get(); if(!f5_.ready()) { state_ = 6; if(!result_.valid()) { result_ = result_promise_.get_future(); } f5_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; } L5: f5_result_ = f5_.get(); if(!f6_.ready()) { state_ = 7; if(!result_.valid()) { result_ = result_promise_.get_future(); } f6_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; } L6: f6_result_ = f6_.get(); if(!f7_.ready()) { state_ = 8; if(!result_.valid()) { result_ = result_promise_.get_future(); } f7_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; } L7: f7_result_ = f7_.get(); if(!f8_.ready()) { state_ = 9; if(!result_.valid()) { result_ = result_promise_.get_future(); } f8_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; } L8: f8_result_ = f8_.get(); if(!f9_.ready()) { state_ = 10; if(!result_.valid()) { result_ = result_promise_.get_future(); } f9_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; } L9: f9_result_ = f9_.get(); if(!f10_.ready()) { state_ = 11; if(!result_.valid()) { result_ = result_promise_.get_future(); } f10_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; } L10: f10_result_ = f10_.get(); if(!f11_.ready()) { state_ = 12; if(!result_.valid()) { result_ = result_promise_.get_future(); } f11_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; } L11: f11_result_ = f11_.get(); if(!f12_.ready()) { state_ = 13; if(!result_.valid()) { result_ = result_promise_.get_future(); } f12_. then( boost::bind( &dataflow_frame_13::await , this->shared_from_this() ) ); return; }
L12:
                f12_result_
                    = f12_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_ , f10_result_ , f11_result_ , f12_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_ , f10_result_ , f11_result_ , f12_result_
                        )
                    );
                }
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_13<
        Func
      , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12
    >::future_result_type
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
                boost::forward<Func>(func)
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename hpx::util::detail::remove_reference<F4>::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_; typedef typename hpx::util::detail::remove_reference<F5>::type f5_type; typedef typename future_traits< f5_type >::value_type f5_result_type; f5_type f5_; f5_result_type f5_result_; typedef typename hpx::util::detail::remove_reference<F6>::type f6_type; typedef typename future_traits< f6_type >::value_type f6_result_type; f6_type f6_; f6_result_type f6_result_; typedef typename hpx::util::detail::remove_reference<F7>::type f7_type; typedef typename future_traits< f7_type >::value_type f7_result_type; f7_type f7_; f7_result_type f7_result_; typedef typename hpx::util::detail::remove_reference<F8>::type f8_type; typedef typename future_traits< f8_type >::value_type f8_result_type; f8_type f8_; f8_result_type f8_result_; typedef typename hpx::util::detail::remove_reference<F9>::type f9_type; typedef typename future_traits< f9_type >::value_type f9_result_type; f9_type f9_; f9_result_type f9_result_; typedef typename hpx::util::detail::remove_reference<F10>::type f10_type; typedef typename future_traits< f10_type >::value_type f10_result_type; f10_type f10_; f10_result_type f10_result_; typedef typename hpx::util::detail::remove_reference<F11>::type f11_type; typedef typename future_traits< f11_type >::value_type f11_result_type; f11_type f11_; f11_result_type f11_result_; typedef typename hpx::util::detail::remove_reference<F12>::type f12_type; typedef typename future_traits< f12_type >::value_type f12_result_type; f12_type f12_; f12_result_type f12_result_; typedef typename hpx::util::detail::remove_reference<F13>::type f13_type; typedef typename future_traits< f13_type >::value_type f13_result_type; f13_type f13_; f13_result_type f13_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type , f4_result_type , f5_result_type , f6_result_type , f7_result_type , f8_result_type , f9_result_type , f10_result_type , f11_result_type , f12_result_type , f13_result_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9 , typename FF10 , typename FF11 , typename FF12 , typename FF13>
            dataflow_frame_14(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9 , BOOST_FWD_REF(FF10) f10 , BOOST_FWD_REF(FF11) f11 , BOOST_FWD_REF(FF12) f12 , BOOST_FWD_REF(FF13) f13
            )
              : func_(boost::forward<FFunc>(func))
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) ) , f4_( boost::forward<FF4>(f4) ) , f5_( boost::forward<FF5>(f5) ) , f6_( boost::forward<FF6>(f6) ) , f7_( boost::forward<FF7>(f7) ) , f8_( boost::forward<FF8>(f8) ) , f9_( boost::forward<FF9>(f9) ) , f10_( boost::forward<FF10>(f10) ) , f11_( boost::forward<FF11>(f11) ) , f12_( boost::forward<FF12>(f12) ) , f13_( boost::forward<FF13>(f13) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3; case 5 : goto L4; case 6 : goto L5; case 7 : goto L6; case 8 : goto L7; case 9 : goto L8; case 10 : goto L9; case 11 : goto L10; case 12 : goto L11; case 13 : goto L12; case 14 : goto L13;
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
                            &dataflow_frame_14::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L3: f3_result_ = f3_.get(); if(!f4_.ready()) { state_ = 5; if(!result_.valid()) { result_ = result_promise_.get_future(); } f4_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L4: f4_result_ = f4_.get(); if(!f5_.ready()) { state_ = 6; if(!result_.valid()) { result_ = result_promise_.get_future(); } f5_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L5: f5_result_ = f5_.get(); if(!f6_.ready()) { state_ = 7; if(!result_.valid()) { result_ = result_promise_.get_future(); } f6_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L6: f6_result_ = f6_.get(); if(!f7_.ready()) { state_ = 8; if(!result_.valid()) { result_ = result_promise_.get_future(); } f7_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L7: f7_result_ = f7_.get(); if(!f8_.ready()) { state_ = 9; if(!result_.valid()) { result_ = result_promise_.get_future(); } f8_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L8: f8_result_ = f8_.get(); if(!f9_.ready()) { state_ = 10; if(!result_.valid()) { result_ = result_promise_.get_future(); } f9_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L9: f9_result_ = f9_.get(); if(!f10_.ready()) { state_ = 11; if(!result_.valid()) { result_ = result_promise_.get_future(); } f10_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L10: f10_result_ = f10_.get(); if(!f11_.ready()) { state_ = 12; if(!result_.valid()) { result_ = result_promise_.get_future(); } f11_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L11: f11_result_ = f11_.get(); if(!f12_.ready()) { state_ = 13; if(!result_.valid()) { result_ = result_promise_.get_future(); } f12_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; } L12: f12_result_ = f12_.get(); if(!f13_.ready()) { state_ = 14; if(!result_.valid()) { result_ = result_promise_.get_future(); } f13_. then( boost::bind( &dataflow_frame_14::await , this->shared_from_this() ) ); return; }
L13:
                f13_result_
                    = f13_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_ , f10_result_ , f11_result_ , f12_result_ , f13_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_ , f10_result_ , f11_result_ , f12_result_ , f13_result_
                        )
                    );
                }
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_14<
        Func
      , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13
    >::future_result_type
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
                boost::forward<Func>(func)
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
            typedef typename hpx::util::detail::remove_reference<F0>::type f0_type; typedef typename future_traits< f0_type >::value_type f0_result_type; f0_type f0_; f0_result_type f0_result_; typedef typename hpx::util::detail::remove_reference<F1>::type f1_type; typedef typename future_traits< f1_type >::value_type f1_result_type; f1_type f1_; f1_result_type f1_result_; typedef typename hpx::util::detail::remove_reference<F2>::type f2_type; typedef typename future_traits< f2_type >::value_type f2_result_type; f2_type f2_; f2_result_type f2_result_; typedef typename hpx::util::detail::remove_reference<F3>::type f3_type; typedef typename future_traits< f3_type >::value_type f3_result_type; f3_type f3_; f3_result_type f3_result_; typedef typename hpx::util::detail::remove_reference<F4>::type f4_type; typedef typename future_traits< f4_type >::value_type f4_result_type; f4_type f4_; f4_result_type f4_result_; typedef typename hpx::util::detail::remove_reference<F5>::type f5_type; typedef typename future_traits< f5_type >::value_type f5_result_type; f5_type f5_; f5_result_type f5_result_; typedef typename hpx::util::detail::remove_reference<F6>::type f6_type; typedef typename future_traits< f6_type >::value_type f6_result_type; f6_type f6_; f6_result_type f6_result_; typedef typename hpx::util::detail::remove_reference<F7>::type f7_type; typedef typename future_traits< f7_type >::value_type f7_result_type; f7_type f7_; f7_result_type f7_result_; typedef typename hpx::util::detail::remove_reference<F8>::type f8_type; typedef typename future_traits< f8_type >::value_type f8_result_type; f8_type f8_; f8_result_type f8_result_; typedef typename hpx::util::detail::remove_reference<F9>::type f9_type; typedef typename future_traits< f9_type >::value_type f9_result_type; f9_type f9_; f9_result_type f9_result_; typedef typename hpx::util::detail::remove_reference<F10>::type f10_type; typedef typename future_traits< f10_type >::value_type f10_result_type; f10_type f10_; f10_result_type f10_result_; typedef typename hpx::util::detail::remove_reference<F11>::type f11_type; typedef typename future_traits< f11_type >::value_type f11_result_type; f11_type f11_; f11_result_type f11_result_; typedef typename hpx::util::detail::remove_reference<F12>::type f12_type; typedef typename future_traits< f12_type >::value_type f12_result_type; f12_type f12_; f12_result_type f12_result_; typedef typename hpx::util::detail::remove_reference<F13>::type f13_type; typedef typename future_traits< f13_type >::value_type f13_result_type; f13_type f13_; f13_result_type f13_result_; typedef typename hpx::util::detail::remove_reference<F14>::type f14_type; typedef typename future_traits< f14_type >::value_type f14_result_type; f14_type f14_; f14_result_type f14_result_;
            
            typedef
                typename boost::result_of<
                    func_type(
                        f0_result_type , f1_result_type , f2_result_type , f3_result_type , f4_result_type , f5_result_type , f6_result_type , f7_result_type , f8_result_type , f9_result_type , f10_result_type , f11_result_type , f12_result_type , f13_result_type , f14_result_type
                    )
                >::type
                result_type;
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;
            template <typename FFunc, typename FF0 , typename FF1 , typename FF2 , typename FF3 , typename FF4 , typename FF5 , typename FF6 , typename FF7 , typename FF8 , typename FF9 , typename FF10 , typename FF11 , typename FF12 , typename FF13 , typename FF14>
            dataflow_frame_15(
                BOOST_FWD_REF(FFunc) func, BOOST_FWD_REF(FF0) f0 , BOOST_FWD_REF(FF1) f1 , BOOST_FWD_REF(FF2) f2 , BOOST_FWD_REF(FF3) f3 , BOOST_FWD_REF(FF4) f4 , BOOST_FWD_REF(FF5) f5 , BOOST_FWD_REF(FF6) f6 , BOOST_FWD_REF(FF7) f7 , BOOST_FWD_REF(FF8) f8 , BOOST_FWD_REF(FF9) f9 , BOOST_FWD_REF(FF10) f10 , BOOST_FWD_REF(FF11) f11 , BOOST_FWD_REF(FF12) f12 , BOOST_FWD_REF(FF13) f13 , BOOST_FWD_REF(FF14) f14
            )
              : func_(boost::forward<FFunc>(func))
              , f0_( boost::forward<FF0>(f0) ) , f1_( boost::forward<FF1>(f1) ) , f2_( boost::forward<FF2>(f2) ) , f3_( boost::forward<FF3>(f3) ) , f4_( boost::forward<FF4>(f4) ) , f5_( boost::forward<FF5>(f5) ) , f6_( boost::forward<FF6>(f6) ) , f7_( boost::forward<FF7>(f7) ) , f8_( boost::forward<FF8>(f8) ) , f9_( boost::forward<FF9>(f9) ) , f10_( boost::forward<FF10>(f10) ) , f11_( boost::forward<FF11>(f11) ) , f12_( boost::forward<FF12>(f12) ) , f13_( boost::forward<FF13>(f13) ) , f14_( boost::forward<FF14>(f14) )
              , state_(0)
            {}
            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    case 1 : goto L0; case 2 : goto L1; case 3 : goto L2; case 4 : goto L3; case 5 : goto L4; case 6 : goto L5; case 7 : goto L6; case 8 : goto L7; case 9 : goto L8; case 10 : goto L9; case 11 : goto L10; case 12 : goto L11; case 13 : goto L12; case 14 : goto L13; case 15 : goto L14;
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
                            &dataflow_frame_15::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }
                L0: f0_result_ = f0_.get(); if(!f1_.ready()) { state_ = 2; if(!result_.valid()) { result_ = result_promise_.get_future(); } f1_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L1: f1_result_ = f1_.get(); if(!f2_.ready()) { state_ = 3; if(!result_.valid()) { result_ = result_promise_.get_future(); } f2_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L2: f2_result_ = f2_.get(); if(!f3_.ready()) { state_ = 4; if(!result_.valid()) { result_ = result_promise_.get_future(); } f3_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L3: f3_result_ = f3_.get(); if(!f4_.ready()) { state_ = 5; if(!result_.valid()) { result_ = result_promise_.get_future(); } f4_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L4: f4_result_ = f4_.get(); if(!f5_.ready()) { state_ = 6; if(!result_.valid()) { result_ = result_promise_.get_future(); } f5_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L5: f5_result_ = f5_.get(); if(!f6_.ready()) { state_ = 7; if(!result_.valid()) { result_ = result_promise_.get_future(); } f6_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L6: f6_result_ = f6_.get(); if(!f7_.ready()) { state_ = 8; if(!result_.valid()) { result_ = result_promise_.get_future(); } f7_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L7: f7_result_ = f7_.get(); if(!f8_.ready()) { state_ = 9; if(!result_.valid()) { result_ = result_promise_.get_future(); } f8_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L8: f8_result_ = f8_.get(); if(!f9_.ready()) { state_ = 10; if(!result_.valid()) { result_ = result_promise_.get_future(); } f9_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L9: f9_result_ = f9_.get(); if(!f10_.ready()) { state_ = 11; if(!result_.valid()) { result_ = result_promise_.get_future(); } f10_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L10: f10_result_ = f10_.get(); if(!f11_.ready()) { state_ = 12; if(!result_.valid()) { result_ = result_promise_.get_future(); } f11_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L11: f11_result_ = f11_.get(); if(!f12_.ready()) { state_ = 13; if(!result_.valid()) { result_ = result_promise_.get_future(); } f12_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L12: f12_result_ = f12_.get(); if(!f13_.ready()) { state_ = 14; if(!result_.valid()) { result_ = result_promise_.get_future(); } f13_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; } L13: f13_result_ = f13_.get(); if(!f14_.ready()) { state_ = 15; if(!result_.valid()) { result_ = result_promise_.get_future(); } f14_. then( boost::bind( &dataflow_frame_15::await , this->shared_from_this() ) ); return; }
L14:
                f14_result_
                    = f14_.get();
                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_ , f10_result_ , f11_result_ , f12_result_ , f13_result_ , f14_result_
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            f0_result_ , f1_result_ , f2_result_ , f3_result_ , f4_result_ , f5_result_ , f6_result_ , f7_result_ , f8_result_ , f9_result_ , f10_result_ , f11_result_ , f12_result_ , f13_result_ , f14_result_
                        )
                    );
                }
            }
            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }
    template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13 , typename F14>
    BOOST_FORCEINLINE
    typename detail::dataflow_frame_15<
        Func
      , F0 , F1 , F2 , F3 , F4 , F5 , F6 , F7 , F8 , F9 , F10 , F11 , F12 , F13 , F14
    >::future_result_type
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
                boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 ) , boost::forward<F14>( f14 )
            );
        frame->await();
        return frame->result_;
    }
}}}
