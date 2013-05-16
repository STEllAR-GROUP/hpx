//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_LCOS_LOCAL_DATAFLOW_HPP
#define HPX_LCOS_LOCAL_DATAFLOW_HPP

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/local/preprocessed/dataflow.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/dataflow_" HPX_LIMIT_STR ".hpp")
#endif

#define HPX_LCOS_LOCAL_DATAFLOW_FRAME_RESULT_OF(z, n, d)                        \
    BOOST_PP_CAT(BOOST_PP_CAT(f, n), _result_type)                              \
/**/

#define HPX_LCOS_LOCAL_DATAFLOW_FRAME_CTOR_LIST(z, n, d)                        \
    BOOST_PP_CAT(BOOST_PP_CAT(f, n), _)(                                        \
        boost::forward<BOOST_PP_CAT(FF, n)>(BOOST_PP_CAT(f, n))                 \
    )                                                                           \
/**/

#define HPX_LCOS_LOCAL_DATAFLOW_FRAME_MEMBER(z, n, d)                           \
    typedef                                                                     \
        typename hpx::util::detail::remove_reference<BOOST_PP_CAT(F, n)>::type  \
        BOOST_PP_CAT(BOOST_PP_CAT(f, n), _type);                                \
                                                                                \
    typedef                                                                     \
        typename future_traits<                                                 \
            BOOST_PP_CAT(BOOST_PP_CAT(f, n), _type)                             \
        >::value_type                                                           \
        BOOST_PP_CAT(BOOST_PP_CAT(f, n), _result_type);                         \
                                                                                \
    BOOST_PP_CAT(BOOST_PP_CAT(f, n), _type)                                     \
        BOOST_PP_CAT(BOOST_PP_CAT(f, n), _);                                    \
    BOOST_PP_CAT(BOOST_PP_CAT(f, n), _result_type)                              \
        BOOST_PP_CAT(BOOST_PP_CAT(f, n), _result_);                             \
/**/
                    
#define HPX_LCOS_LOCAL_DATAFLOW_FRAME_STATE_SWITCH(z, n, d)                     \
    case BOOST_PP_INC(n) : goto BOOST_PP_CAT(L, n);                             \
/**/

#define HPX_LCOS_LOCAL_DATAFLOW_FRAME_LABEL(z, n, d)                            \
BOOST_PP_CAT(L, n):                                                             \
    BOOST_PP_CAT(BOOST_PP_CAT(f, n), _result_)                                  \
        = BOOST_PP_CAT(BOOST_PP_CAT(f, n), _).get();                            \
                                                                                \
    if(!BOOST_PP_CAT(BOOST_PP_CAT(f, BOOST_PP_INC(n)), _).ready())              \
    {                                                                           \
        state_ = BOOST_PP_INC(BOOST_PP_INC(n));                                 \
        if(!result_.valid())                                                    \
        {                                                                       \
            result_ = result_promise_.get_future();                             \
        }                                                                       \
        BOOST_PP_CAT(BOOST_PP_CAT(f, BOOST_PP_INC(n)), _).                      \
            then(                                                               \
                boost::bind(                                                    \
                    &BOOST_PP_CAT(dataflow_frame_, N)::await                    \
                  , this->shared_from_this()                                    \
                )                                                               \
            );                                                                  \
        return;                                                                 \
    }                                                                           \
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
#undef HPX_LCOS_LOCAL_DATAFLOW_FRAME_STATE_SWITCH
#undef HPX_LCOS_LOCAL_DATAFLOW_FRAME_LABEL
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
            BOOST_PP_REPEAT(N, HPX_LCOS_LOCAL_DATAFLOW_FRAME_MEMBER, _)
            
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
            typedef hpx::future<result_type> future_result_type;
            typedef hpx::lcos::local::promise<result_type> promise_result_type;

            template <typename FFunc, BOOST_PP_ENUM_PARAMS(N, typename FF)>
            BOOST_PP_CAT(dataflow_frame_, N)(
                BOOST_FWD_REF(FFunc) func, HPX_ENUM_FWD_ARGS(N, FF, f)
            )
              : func_(boost::forward<FFunc>(func))
              , BOOST_PP_ENUM(N, HPX_LCOS_LOCAL_DATAFLOW_FRAME_CTOR_LIST, _)
              , state_(0)
            {}

            BOOST_FORCEINLINE void await()
            {
                switch (state_)
                {
                    BOOST_PP_REPEAT(
                        N
                      , HPX_LCOS_LOCAL_DATAFLOW_FRAME_STATE_SWITCH
                      , _
                    )
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
                            &BOOST_PP_CAT(dataflow_frame_, N)::await
                          , this->shared_from_this()
                        )
                    );
                    return;
                }

                BOOST_PP_REPEAT(
                    BOOST_PP_DEC(N)
                  , HPX_LCOS_LOCAL_DATAFLOW_FRAME_LABEL
                  , _
                )

BOOST_PP_CAT(L, BOOST_PP_DEC(N)):
                BOOST_PP_CAT(BOOST_PP_CAT(f, BOOST_PP_DEC(N)), _result_)
                    = BOOST_PP_CAT(BOOST_PP_CAT(f, BOOST_PP_DEC(N)), _).get();

                if(state_ == 0)
                {
                    result_
                        = hpx::make_ready_future(
                            func_(
                                BOOST_PP_ENUM(
                                    N
                                  , HPX_LCOS_LOCAL_DATAFLOW_FRAME_INVOKE
                                  , _
                                )
                            )
                        );
                }
                else
                {
                    result_promise_.set_value(
                        func_(
                            BOOST_PP_ENUM(
                                N
                              , HPX_LCOS_LOCAL_DATAFLOW_FRAME_INVOKE
                              , _
                            )
                        )
                    );
                }
            }

            future_result_type result_;
            promise_result_type result_promise_;
            int state_;
        };
    }

    template <typename Func, BOOST_PP_ENUM_PARAMS(N, typename F)>
    BOOST_FORCEINLINE
    typename BOOST_PP_CAT(detail::dataflow_frame_, N)<
        Func
      , BOOST_PP_ENUM_PARAMS(N, F)
    >::future_result_type
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
                boost::forward<Func>(func)
              , HPX_ENUM_FORWARD_ARGS(N, F, f)
            );

        frame->await();

        return frame->result_;
    }

}}}

#endif
