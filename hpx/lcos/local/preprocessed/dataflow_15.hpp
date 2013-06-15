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
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0>
            dataflow_frame_1(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0
            )
              : futures_(
                    boost::forward<A0>(f0)
                )
              , policy_(policy)
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
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_1::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1>
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
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1>
            dataflow_frame_2(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1)
                )
              , policy_(policy)
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
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_2::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2>
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
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2>
            dataflow_frame_3(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2)
                )
              , policy_(policy)
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
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_3::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3>
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
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3>
            dataflow_frame_4(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3)
                )
              , policy_(policy)
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
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_4::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
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
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
            dataflow_frame_5(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4)
                )
              , policy_(policy)
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
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_5::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5>
        struct dataflow_frame_6
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type;
            typedef
                hpx::util::tuple6<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_6>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_6::*)(boost::mpl::true_)
                  , void(dataflow_frame_6::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
            dataflow_frame_6(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4 , BOOST_FWD_REF(A5) f5
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4) , boost::forward<A5>(f5)
                )
              , policy_(policy)
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_6::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                    void (dataflow_frame_6::*f)
                        (Iter, Iter)
                        = &dataflow_frame_6::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                    void (dataflow_frame_6::*f)
                        (Iter, boost::mpl::false_)
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 )
            );
        return frame->get_future();
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6>
        struct dataflow_frame_7
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type;
            typedef
                hpx::util::tuple7<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_7>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_7::*)(boost::mpl::true_)
                  , void(dataflow_frame_7::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
            dataflow_frame_7(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4 , BOOST_FWD_REF(A5) f5 , BOOST_FWD_REF(A6) f6
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4) , boost::forward<A5>(f5) , boost::forward<A6>(f6)
                )
              , policy_(policy)
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_7::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                    void (dataflow_frame_7::*f)
                        (Iter, Iter)
                        = &dataflow_frame_7::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                    void (dataflow_frame_7::*f)
                        (Iter, boost::mpl::false_)
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 )
            );
        return frame->get_future();
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7>
        struct dataflow_frame_8
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type;
            typedef
                hpx::util::tuple8<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_8>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_8::*)(boost::mpl::true_)
                  , void(dataflow_frame_8::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
            dataflow_frame_8(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4 , BOOST_FWD_REF(A5) f5 , BOOST_FWD_REF(A6) f6 , BOOST_FWD_REF(A7) f7
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4) , boost::forward<A5>(f5) , boost::forward<A6>(f6) , boost::forward<A7>(f7)
                )
              , policy_(policy)
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_8::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                    void (dataflow_frame_8::*f)
                        (Iter, Iter)
                        = &dataflow_frame_8::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                    void (dataflow_frame_8::*f)
                        (Iter, boost::mpl::false_)
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 )
            );
        return frame->get_future();
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8>
        struct dataflow_frame_9
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type;
            typedef
                hpx::util::tuple9<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_9>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_9::*)(boost::mpl::true_)
                  , void(dataflow_frame_9::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
            dataflow_frame_9(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4 , BOOST_FWD_REF(A5) f5 , BOOST_FWD_REF(A6) f6 , BOOST_FWD_REF(A7) f7 , BOOST_FWD_REF(A8) f8
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4) , boost::forward<A5>(f5) , boost::forward<A6>(f6) , boost::forward<A7>(f7) , boost::forward<A8>(f8)
                )
              , policy_(policy)
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_9::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                    void (dataflow_frame_9::*f)
                        (Iter, Iter)
                        = &dataflow_frame_9::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                    void (dataflow_frame_9::*f)
                        (Iter, boost::mpl::false_)
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 )
            );
        return frame->get_future();
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9>
        struct dataflow_frame_10
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type;
            typedef
                hpx::util::tuple10<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_10>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_10::*)(boost::mpl::true_)
                  , void(dataflow_frame_10::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
            dataflow_frame_10(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4 , BOOST_FWD_REF(A5) f5 , BOOST_FWD_REF(A6) f6 , BOOST_FWD_REF(A7) f7 , BOOST_FWD_REF(A8) f8 , BOOST_FWD_REF(A9) f9
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4) , boost::forward<A5>(f5) , boost::forward<A6>(f6) , boost::forward<A7>(f7) , boost::forward<A8>(f8) , boost::forward<A9>(f9)
                )
              , policy_(policy)
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_10::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                    void (dataflow_frame_10::*f)
                        (Iter, Iter)
                        = &dataflow_frame_10::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                    void (dataflow_frame_10::*f)
                        (Iter, boost::mpl::false_)
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 )
            );
        return frame->get_future();
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10>
        struct dataflow_frame_11
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type f10_type;
            typedef
                hpx::util::tuple11<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_11>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_11::*)(boost::mpl::true_)
                  , void(dataflow_frame_11::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
            dataflow_frame_11(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4 , BOOST_FWD_REF(A5) f5 , BOOST_FWD_REF(A6) f6 , BOOST_FWD_REF(A7) f7 , BOOST_FWD_REF(A8) f8 , BOOST_FWD_REF(A9) f9 , BOOST_FWD_REF(A10) f10
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4) , boost::forward<A5>(f5) , boost::forward<A6>(f6) , boost::forward<A7>(f7) , boost::forward<A8>(f8) , boost::forward<A9>(f9) , boost::forward<A10>(f10)
                )
              , policy_(policy)
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type(); boost::fusion::at_c< 10 >(futures_) = f10_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type(); boost::fusion::at_c< 10 >(futures_) = f10_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_11::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                    void (dataflow_frame_11::*f)
                        (Iter, Iter)
                        = &dataflow_frame_11::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                    void (dataflow_frame_11::*f)
                        (Iter, boost::mpl::false_)
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 )
            );
        return frame->get_future();
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11>
        struct dataflow_frame_12
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type f10_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type f11_type;
            typedef
                hpx::util::tuple12<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_12>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_12::*)(boost::mpl::true_)
                  , void(dataflow_frame_12::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
            dataflow_frame_12(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4 , BOOST_FWD_REF(A5) f5 , BOOST_FWD_REF(A6) f6 , BOOST_FWD_REF(A7) f7 , BOOST_FWD_REF(A8) f8 , BOOST_FWD_REF(A9) f9 , BOOST_FWD_REF(A10) f10 , BOOST_FWD_REF(A11) f11
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4) , boost::forward<A5>(f5) , boost::forward<A6>(f6) , boost::forward<A7>(f7) , boost::forward<A8>(f8) , boost::forward<A9>(f9) , boost::forward<A10>(f10) , boost::forward<A11>(f11)
                )
              , policy_(policy)
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type(); boost::fusion::at_c< 10 >(futures_) = f10_type(); boost::fusion::at_c< 11 >(futures_) = f11_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type(); boost::fusion::at_c< 10 >(futures_) = f10_type(); boost::fusion::at_c< 11 >(futures_) = f11_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_12::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                    void (dataflow_frame_12::*f)
                        (Iter, Iter)
                        = &dataflow_frame_12::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                    void (dataflow_frame_12::*f)
                        (Iter, boost::mpl::false_)
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 )
            );
        return frame->get_future();
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12>
        struct dataflow_frame_13
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type f10_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type f11_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type f12_type;
            typedef
                hpx::util::tuple13<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type , f12_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_13>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_13::*)(boost::mpl::true_)
                  , void(dataflow_frame_13::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
            dataflow_frame_13(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4 , BOOST_FWD_REF(A5) f5 , BOOST_FWD_REF(A6) f6 , BOOST_FWD_REF(A7) f7 , BOOST_FWD_REF(A8) f8 , BOOST_FWD_REF(A9) f9 , BOOST_FWD_REF(A10) f10 , BOOST_FWD_REF(A11) f11 , BOOST_FWD_REF(A12) f12
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4) , boost::forward<A5>(f5) , boost::forward<A6>(f6) , boost::forward<A7>(f7) , boost::forward<A8>(f8) , boost::forward<A9>(f9) , boost::forward<A10>(f10) , boost::forward<A11>(f11) , boost::forward<A12>(f12)
                )
              , policy_(policy)
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type(); boost::fusion::at_c< 10 >(futures_) = f10_type(); boost::fusion::at_c< 11 >(futures_) = f11_type(); boost::fusion::at_c< 12 >(futures_) = f12_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type(); boost::fusion::at_c< 10 >(futures_) = f10_type(); boost::fusion::at_c< 11 >(futures_) = f11_type(); boost::fusion::at_c< 12 >(futures_) = f12_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_13::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                    void (dataflow_frame_13::*f)
                        (Iter, Iter)
                        = &dataflow_frame_13::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                    void (dataflow_frame_13::*f)
                        (Iter, boost::mpl::false_)
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 )
            );
        return frame->get_future();
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13>
        struct dataflow_frame_14
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F13 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type f10_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type f11_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type f12_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F13 >::type >::type f13_type;
            typedef
                hpx::util::tuple14<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type , f12_type , f13_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F13 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_14>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_14::*)(boost::mpl::true_)
                  , void(dataflow_frame_14::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
            dataflow_frame_14(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4 , BOOST_FWD_REF(A5) f5 , BOOST_FWD_REF(A6) f6 , BOOST_FWD_REF(A7) f7 , BOOST_FWD_REF(A8) f8 , BOOST_FWD_REF(A9) f9 , BOOST_FWD_REF(A10) f10 , BOOST_FWD_REF(A11) f11 , BOOST_FWD_REF(A12) f12 , BOOST_FWD_REF(A13) f13
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4) , boost::forward<A5>(f5) , boost::forward<A6>(f6) , boost::forward<A7>(f7) , boost::forward<A8>(f8) , boost::forward<A9>(f9) , boost::forward<A10>(f10) , boost::forward<A11>(f11) , boost::forward<A12>(f12) , boost::forward<A13>(f13)
                )
              , policy_(policy)
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type(); boost::fusion::at_c< 10 >(futures_) = f10_type(); boost::fusion::at_c< 11 >(futures_) = f11_type(); boost::fusion::at_c< 12 >(futures_) = f12_type(); boost::fusion::at_c< 13 >(futures_) = f13_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type(); boost::fusion::at_c< 10 >(futures_) = f10_type(); boost::fusion::at_c< 11 >(futures_) = f11_type(); boost::fusion::at_c< 12 >(futures_) = f12_type(); boost::fusion::at_c< 13 >(futures_) = f13_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_14::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                    void (dataflow_frame_14::*f)
                        (Iter, Iter)
                        = &dataflow_frame_14::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                    void (dataflow_frame_14::*f)
                        (Iter, boost::mpl::false_)
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 )
            );
        return frame->get_future();
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 )
            );
        return frame->get_future();
    }
}}}
namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12 , typename F13 , typename F14>
        struct dataflow_frame_15
          : hpx::lcos::detail::future_data<
                typename boost::result_of<
                    typename hpx::util::detail::remove_reference<Func>::type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F13 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F14 >::type >::type
                    )
                >::type
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;
            
            typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type f0_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type f1_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type f2_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type f3_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type f4_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type f5_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type f6_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type f7_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type f8_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type f9_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type f10_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type f11_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type f12_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F13 >::type >::type f13_type; typedef typename boost::remove_const< typename hpx::util::detail::remove_reference< F14 >::type >::type f14_type;
            typedef
                hpx::util::tuple15<
                    f0_type , f1_type , f2_type , f3_type , f4_type , f5_type , f6_type , f7_type , f8_type , f9_type , f10_type , f11_type , f12_type , f13_type , f14_type
                >
                futures_type;
            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;
            typedef
                typename boost::result_of<
                    func_type(
                        typename boost::remove_const< typename hpx::util::detail::remove_reference< F0 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F1 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F2 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F3 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F4 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F5 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F6 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F7 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F8 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F9 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F10 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F11 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F12 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F13 >::type >::type , typename boost::remove_const< typename hpx::util::detail::remove_reference< F14 >::type >::type
                    )
                >::type
                result_type;
            typedef
                boost::intrusive_ptr<dataflow_frame_15>
                future_base_type;
            typedef hpx::lcos::future<result_type> type;
            typedef
                typename boost::mpl::if_<
                    boost::is_void<result_type>
                  , void(dataflow_frame_15::*)(boost::mpl::true_)
                  , void(dataflow_frame_15::*)(boost::mpl::false_)
                >::type
                execute_function_type;
            futures_type futures_;
            BOOST_SCOPED_ENUM(launch) policy_;
            func_type func_;
            template <typename FFunc, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
            dataflow_frame_15(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , BOOST_FWD_REF(A0) f0 , BOOST_FWD_REF(A1) f1 , BOOST_FWD_REF(A2) f2 , BOOST_FWD_REF(A3) f3 , BOOST_FWD_REF(A4) f4 , BOOST_FWD_REF(A5) f5 , BOOST_FWD_REF(A6) f6 , BOOST_FWD_REF(A7) f7 , BOOST_FWD_REF(A8) f8 , BOOST_FWD_REF(A9) f9 , BOOST_FWD_REF(A10) f10 , BOOST_FWD_REF(A11) f11 , BOOST_FWD_REF(A12) f12 , BOOST_FWD_REF(A13) f13 , BOOST_FWD_REF(A14) f14
            )
              : futures_(
                    boost::forward<A0>(f0) , boost::forward<A1>(f1) , boost::forward<A2>(f2) , boost::forward<A3>(f3) , boost::forward<A4>(f4) , boost::forward<A5>(f5) , boost::forward<A6>(f6) , boost::forward<A7>(f7) , boost::forward<A8>(f8) , boost::forward<A9>(f9) , boost::forward<A10>(f10) , boost::forward<A11>(f11) , boost::forward<A12>(f12) , boost::forward<A13>(f13) , boost::forward<A14>(f14)
                )
              , policy_(policy)
              , func_(boost::forward<FFunc>(func))
            {}
            BOOST_FORCEINLINE
            void execute(boost::mpl::false_)
            {
                result_type res(
                    boost::move(boost::fusion::invoke(func_, futures_))
                );
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type(); boost::fusion::at_c< 10 >(futures_) = f10_type(); boost::fusion::at_c< 11 >(futures_) = f11_type(); boost::fusion::at_c< 12 >(futures_) = f12_type(); boost::fusion::at_c< 13 >(futures_) = f13_type(); boost::fusion::at_c< 14 >(futures_) = f14_type();
                this->set_data(boost::move(res));
            }
            BOOST_FORCEINLINE
            void execute(boost::mpl::true_)
            {
                boost::fusion::invoke(func_, futures_);
                boost::fusion::at_c< 0 >(futures_) = f0_type(); boost::fusion::at_c< 1 >(futures_) = f1_type(); boost::fusion::at_c< 2 >(futures_) = f2_type(); boost::fusion::at_c< 3 >(futures_) = f3_type(); boost::fusion::at_c< 4 >(futures_) = f4_type(); boost::fusion::at_c< 5 >(futures_) = f5_type(); boost::fusion::at_c< 6 >(futures_) = f6_type(); boost::fusion::at_c< 7 >(futures_) = f7_type(); boost::fusion::at_c< 8 >(futures_) = f8_type(); boost::fusion::at_c< 9 >(futures_) = f9_type(); boost::fusion::at_c< 10 >(futures_) = f10_type(); boost::fusion::at_c< 11 >(futures_) = f11_type(); boost::fusion::at_c< 12 >(futures_) = f12_type(); boost::fusion::at_c< 13 >(futures_) = f13_type(); boost::fusion::at_c< 14 >(futures_) = f14_type();
                this->set_data(util::unused_type());
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(
                BOOST_FWD_REF(Iter) iter, boost::mpl::true_)
            {
                typedef
                    boost::mpl::bool_<boost::is_void<result_type>::value>
                    is_void;
                if(policy_ == hpx::launch::sync)
                {
                    execute(is_void());
                    return;
                }
                execute_function_type f = &dataflow_frame_15::execute;
                hpx::apply(hpx::util::bind(f, future_base_type(this), is_void()));
            }
            template <typename Iter>
            BOOST_FORCEINLINE
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
                    void (dataflow_frame_15::*f)
                        (Iter, Iter)
                        = &dataflow_frame_15::await_range;
                    typedef
                        typename extract_completed_callback_type<
                            future_type
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
                                      , future_base_type(this)
                                      , boost::move(next)
                                      , boost::move(end)
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(*next)
                        ->set_on_completed(
                            boost::move(
                                boost::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(next)
                                  , boost::move(end)
                                )
                            )
                        );
                    }
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
                    boost::move(boost::fusion::next(iter))
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
                    void (dataflow_frame_15::*f)
                        (Iter, boost::mpl::false_)
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
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                compose_cb(
                                    boost::move(cb)
                                  , hpx::util::bind(
                                        f
                                      , future_base_type(this)
                                      , boost::move(iter)
                                      , boost::mpl::false_()
                                    )
                                )
                            )
                        );
                    }
                    else
                    {
                        hpx::lcos::detail::get_future_data(f_)
                        ->set_on_completed(
                            boost::move(
                                hpx::util::bind(
                                    f
                                  , future_base_type(this)
                                  , boost::move(iter)
                                  , boost::mpl::false_()
                                )
                            )
                        );
                    }
                    return;
                }
                await(
                    boost::move(boost::fusion::next(iter))
                  , boost::mpl::bool_<
                        boost::is_same<next_type, end_type>::value
                    >()
                );
            }
            template <typename Iter>
            BOOST_FORCEINLINE
            void await(Iter iter, boost::mpl::false_)
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
                    boost::move(boost::fusion::begin(futures_))
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                policy
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 ) , boost::forward<F14>( f14 )
            );
        return frame->get_future();
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
        boost::intrusive_ptr<frame_type> frame =
            new frame_type(
                launch::all
              , boost::forward<Func>(func)
              , boost::forward<F0>( f0 ) , boost::forward<F1>( f1 ) , boost::forward<F2>( f2 ) , boost::forward<F3>( f3 ) , boost::forward<F4>( f4 ) , boost::forward<F5>( f5 ) , boost::forward<F6>( f6 ) , boost::forward<F7>( f7 ) , boost::forward<F8>( f8 ) , boost::forward<F9>( f9 ) , boost::forward<F10>( f10 ) , boost::forward<F11>( f11 ) , boost::forward<F12>( f12 ) , boost::forward<F13>( f13 ) , boost::forward<F14>( f14 )
            );
        return frame->get_future();
    }
}}}
