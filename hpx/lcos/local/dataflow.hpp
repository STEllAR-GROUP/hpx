//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_LCOS_LOCAL_DATAFLOW_HPP
#define HPX_LCOS_LOCAL_DATAFLOW_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/detail/remove_reference.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>

#include <boost/fusion/include/deref.hpp>
#include <boost/fusion/include/begin.hpp>
#include <boost/fusion/include/end.hpp>
#include <boost/fusion/include/next.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/cat.hpp>

namespace hpx { namespace lcos { namespace local { namespace detail
{
    template <typename T>
    struct is_future_or_future_range
      : boost::mpl::or_<traits::is_future<T>, traits::is_future_range<T> >
    {};

    template <
        typename Future
      , typename IsFutureRange = typename traits::is_future_range<Future>::type
    >
    struct extract_completed_callback_type;

    template <typename Future>
    struct extract_completed_callback_type<Future, boost::mpl::true_>
    {
        typedef
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Future
                >::type
            >::type::value_type::future_data_type
            future_data_type;

        typedef
            typename future_data_type::completed_callback_type
            type;
    };

    template <typename Future>
    struct extract_completed_callback_type<Future, boost::mpl::false_>
    {
        typedef
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<
                    Future
                >::type
            >::type::future_data_type
            future_data_type;

        typedef
            typename future_data_type::completed_callback_type
            type;
    };

    template <typename F1, typename F2>
    struct compose_cb_impl
    {
        typename util::detail::remove_reference<F1>::type f1_;
        typename util::detail::remove_reference<F2>::type f2_;

        template <typename FF1, typename FF2>
        compose_cb_impl(BOOST_FWD_REF(FF1) f1, BOOST_FWD_REF(FF2) f2)
          : f1_(boost::forward<FF1>(f1))
          , f2_(boost::forward<FF2>(f2))
        {}

        template <typename Future>
        void operator()(Future & f)
        {
            f1_(f);
            f2_(f);
        }
    };

    template <typename F1, typename F2>
    compose_cb_impl<F1, F2>
    compose_cb(BOOST_FWD_REF(F1) f1, BOOST_FWD_REF(F2) f2)
    {
        return
            boost::move(
                compose_cb_impl<F1, F2>(
                    boost::forward<F1>(f1)
                  , boost::forward<F2>(f2)
                )
            );
    }
}}}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/local/preprocessed/dataflow.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/dataflow_" HPX_LIMIT_STR ".hpp")
#endif

#define HPX_LCOS_LOCAL_DATAFLOW_FRAME_RESULT_OF(z, n, d)                        \
    BOOST_PP_CAT(BOOST_PP_CAT(f, n), _type)                                     \
/**/

#define HPX_LCOS_LOCAL_DATAFLOW_FRAME_CTOR_LIST(z, n, d)                        \
    boost::forward<BOOST_PP_CAT(FF, n)>(BOOST_PP_CAT(f, n))                     \
/**/

#define HPX_LCOS_LOCAL_DATAFLOW_FRAME_MEMBER(z, n, d)                           \
    BOOST_PP_CAT(BOOST_PP_CAT(f, n), _type)                                     \
/**/
#define HPX_LCOS_LOCAL_DATAFLOW_FRAME_MEMBER_TYPES(z, n, d)                     \
    typedef                                                                     \
        typename boost::remove_const<                                           \
            typename hpx::util::detail::remove_reference<                       \
                BOOST_PP_CAT(F, n)                                              \
            >::type                                                             \
        >::type                                                                 \
        BOOST_PP_CAT(BOOST_PP_CAT(f, n), _type);                                \
/**/

#define HPX_LCOS_LOCAL_DATAFLOW_FRAME_INVOKE(z, n, d)                           \
    BOOST_PP_CAT(BOOST_PP_CAT(f, n), _result_)                                  \
/**/

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT, <hpx/lcos/local/dataflow.hpp>))          \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_LCOS_LOCAL_DATAFLOW_FRAME_CTOR_LIST
#undef HPX_LCOS_LOCAL_DATAFLOW_FRAME_MEMBER
#undef HPX_LCOS_LOCAL_DATAFLOW_FRAME_RESULT_OF
#undef HPX_LCOS_LOCAL_DATAFLOW_FRAME_INVOKE

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos { namespace local {
    namespace detail {
        template <typename Func, BOOST_PP_ENUM_PARAMS(N, typename F)>
        struct BOOST_PP_CAT(dataflow_frame_, N)
          : boost::enable_shared_from_this<
                BOOST_PP_CAT(dataflow_frame_, N)<
                    Func, BOOST_PP_ENUM_PARAMS(N, F)
                >
            >
        {
            typedef
                typename hpx::util::detail::remove_reference<Func>::type
                func_type;

            func_type func_;
            BOOST_PP_REPEAT(N, HPX_LCOS_LOCAL_DATAFLOW_FRAME_MEMBER_TYPES, _)
            typedef
                BOOST_PP_CAT(hpx::util::tuple, N)<
                    BOOST_PP_ENUM(N, HPX_LCOS_LOCAL_DATAFLOW_FRAME_MEMBER, _)
                >
                futures_type;
            futures_type futures_;

            typedef
                typename boost::fusion::result_of::end<futures_type>::type
                end_type;

            typedef
                typename boost::result_of<
                    func_type(
                        BOOST_PP_ENUM(
                            N
                          , HPX_LCOS_LOCAL_DATAFLOW_FRAME_RESULT_OF
                          , _
                        )
                    )
                >::type
                result_type;

            typedef hpx::future<result_type> type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;

            template <typename FFunc, BOOST_PP_ENUM_PARAMS(N, typename FF)>
            BOOST_PP_CAT(dataflow_frame_, N)(
                BOOST_SCOPED_ENUM(launch) policy
              , BOOST_FWD_REF(FFunc) func
              , HPX_ENUM_FWD_ARGS(N, FF, f)
            )
              : func_(boost::forward<FFunc>(func))
              , futures_(
                    BOOST_PP_ENUM(N, HPX_LCOS_LOCAL_DATAFLOW_FRAME_CTOR_LIST, _)
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
                            &BOOST_PP_CAT(dataflow_frame_, N)::execute
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
                            &BOOST_PP_CAT(dataflow_frame_, N)::execute
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

                    void (BOOST_PP_CAT(dataflow_frame_, N)::*f)
                        (Iter, Iter)
                        = &BOOST_PP_CAT(dataflow_frame_, N)::await_range;

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
                future_type &  f_ =
                    boost::fusion::deref(iter);

                if(!f_.ready())
                {
                    if(!result_.valid())
                    {
                        result_ = result_promise_.get_future();
                    }

                    void (BOOST_PP_CAT(dataflow_frame_, N)::*f)
                        (Iter const &, IsVoid, boost::mpl::false_)
                        = &BOOST_PP_CAT(dataflow_frame_, N)::await_next;

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

    template <typename Func, BOOST_PP_ENUM_PARAMS(N, typename F)>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        detail::is_future_or_future_range<
            typename boost::remove_const<
                typename hpx::util::detail::remove_reference<Func>::type
            >::type
        >
      , BOOST_PP_CAT(detail::dataflow_frame_, N)<
            Func
          , BOOST_PP_ENUM_PARAMS(N, F)
        >
    >::type
    dataflow(
        BOOST_SCOPED_ENUM(launch) policy
      , BOOST_FWD_REF(Func) func
      , HPX_ENUM_FWD_ARGS(N, F, f)
    )
    {
        typedef
            BOOST_PP_CAT(detail::dataflow_frame_, N)<
                Func
              , BOOST_PP_ENUM_PARAMS(N, F)
            >
            frame_type;

        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                policy
              , boost::forward<Func>(func)
              , HPX_ENUM_FORWARD_ARGS(N, F, f)
            );

        frame->await();

        return frame->result_;
    }

    template <typename Func, BOOST_PP_ENUM_PARAMS(N, typename F)>
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
      , BOOST_PP_CAT(detail::dataflow_frame_, N)<
            Func
          , BOOST_PP_ENUM_PARAMS(N, F)
        >
    >::type
    dataflow(BOOST_FWD_REF(Func) func, HPX_ENUM_FWD_ARGS(N, F, f))
    {
        typedef
            BOOST_PP_CAT(detail::dataflow_frame_, N)<
                Func
              , BOOST_PP_ENUM_PARAMS(N, F)
            >
            frame_type;

        boost::shared_ptr<frame_type> frame =
            boost::make_shared<frame_type>(
                launch::all
              , boost::forward<Func>(func)
              , HPX_ENUM_FORWARD_ARGS(N, F, f)
            );

        frame->await();

        return frame->result_;
    }

}}}

#endif
