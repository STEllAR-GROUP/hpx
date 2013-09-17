//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_BIND_HPP
#define HPX_UTIL_BIND_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_bind_expression.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/remove_reference.hpp>
#include <hpx/util/invoke.hpp>

#include <boost/get_pointer.hpp>

#include <boost/fusion/include/invoke_function_object.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/vector.hpp>

#include <boost/detail/workaround.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/result_of.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <boost/type_traits/is_same.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    namespace detail
    {
        struct not_enough_arguments {};

        struct not_callable {};

        namespace result_of
        {
            template <typename Env, typename T>
            struct eval
            {
                typedef T& type;
            };

            template <typename Env, typename T>
            struct eval<Env, boost::reference_wrapper<T const> >
            {
                typedef T const& type;
            };

            template <typename Env, typename T>
            struct eval<Env, boost::reference_wrapper<T> >
            {
                typedef T& type;
            };
        }

        template <typename Env, typename T>
        T& eval(Env&, T& t)
        {
            return t;
        }

        template <typename Env, typename T>
        T const& eval(Env&, T const& t)
        {
            return t;
        }

        template <typename Env, typename T>
        T& eval(Env&, boost::reference_wrapper<T> const& r)
        {
            return r.get();
        }

        template <typename Env, typename T>
        T const& eval(Env&, boost::reference_wrapper<T const> const& r)
        {
            return r.get();
        }

        template <typename Env, typename T>
        T& eval(Env&, boost::reference_wrapper<T>& r)
        {
            return r.get();
        }

        template <typename Env, typename T>
        T const& eval(Env&, boost::reference_wrapper<T const>& r)
        {
            return r.get();
        }

        ///////////////////////////////////////////////////////////////////////
        using boost::get_pointer;

        template <typename T>
        T* get_pointer(boost::shared_ptr<T>& p)
        {
            return p.get();
        }

        template <typename T>
        T* get_pointer(boost::intrusive_ptr<T>& p)
        {
            return p.get();
        }

        template <typename T>
        T * get_pointer(T& t)
        {
            return &t;
        }

        template <typename T>
        T const * get_pointer(T const& t)
        {
            return &t;
        }
    }

    namespace placeholders
    {
        template <int N>
        struct arg {};

#define HPX_UTIL_BIND_PLACEHOLDERS(Z, N, D)                                     \
        typedef arg<N> BOOST_PP_CAT(BOOST_PP_CAT(arg_, BOOST_PP_INC(N)), _type);\
        BOOST_PP_CAT(BOOST_PP_CAT(arg_, BOOST_PP_INC(N)), _type) const          \
            BOOST_PP_CAT(arg_, BOOST_PP_INC(N)) = {};                           \
        BOOST_PP_CAT(BOOST_PP_CAT(arg_, BOOST_PP_INC(N)), _type) const          \
            BOOST_PP_CAT(_, BOOST_PP_INC(N)) = {};                              \
/**/

        BOOST_PP_REPEAT(
            HPX_FUNCTION_ARGUMENT_LIMIT
          , HPX_UTIL_BIND_PLACEHOLDERS, _
        )

#undef HPX_UTIL_BIND_PLACEHOLDERS

        template <typename>
        struct is_placeholder
          : boost::mpl::false_
        {};

        template <typename T>
        struct is_placeholder<T const>
          : is_placeholder<T>
        {};

        template <int N>
        struct is_placeholder<arg<N> >
          : boost::mpl::true_
        {};
    }

    namespace detail
    {
        namespace result_of
        {
            template <typename Env, int N>
            struct eval<Env, util::placeholders::arg<N> >
            {
                typedef
                    typename boost::mpl::eval_if_c<
                        N < util::tuple_size<Env>::value
                      , util::tuple_element<N, Env>
                      , boost::mpl::identity<not_enough_arguments>
                    >::type
                    type;
            };

            template <typename Env, int N>
            struct eval<Env, util::placeholders::arg<N> const>
            {
                typedef
                    typename boost::mpl::eval_if_c<
                        N < util::tuple_size<Env>::value
                      , util::tuple_element<N, Env>
                      , boost::mpl::identity<not_enough_arguments>
                    >::type
                    type;
            };
        }

        template <typename Env, int N>
        typename util::tuple_element<N, Env>::type
        eval(Env& env, util::placeholders::arg<N>)
        {
            return util::get<N>(env);
        }
    }

#define HPX_UTIL_BIND_EVAL_TYPE(Z, N, D)                                        \
    typename result_of::eval<hpx::util::tuple<>, BOOST_PP_CAT(Arg, N)>::type    \
/**/

#define HPX_UTIL_BIND_CONST_EVAL_TYPE(Z, N, D)                                  \
    typename result_of::eval<hpx::util::tuple<>, BOOST_PP_CAT(Arg, N) const>::type\
/**/

#define HPX_UTIL_BIND_EVAL(Z, N, D)                                             \
    ::hpx::util::detail::eval(env, BOOST_PP_CAT(arg, N))                        \
/**/

#define HPX_UTIL_BIND_REMOVE_REFERENCE(Z, N, D)                                 \
        typename detail::remove_reference<BOOST_PP_CAT(D, N)>::type             \
/**/

#define HPX_UTIL_BIND_REFERENCE(Z, N, D)                                        \
    typename detail::env_value_type<BOOST_PP_CAT(D, N)>::type                   \
/**/

#define HPX_UTIL_BIND_FUNCTION_OPERATOR(Z, NN, D)                               \
    template <typename This BOOST_PP_ENUM_TRAILING_PARAMS(NN, typename A)>      \
    struct result<This const(BOOST_PP_ENUM_PARAMS(NN, A))>                      \
    {                                                                           \
        typedef                                                                 \
        typename result_of::BOOST_PP_CAT(bound, N)<                             \
            F const(BOOST_PP_ENUM_PARAMS(NN, A)) BOOST_PP_ENUM_TRAILING_PARAMS(N, Arg)\
                >::type type;                                                   \
    };                                                                          \
                                                                                \
    template <typename This BOOST_PP_ENUM_TRAILING_PARAMS(NN, typename A)>      \
    struct result<This(BOOST_PP_ENUM_PARAMS(NN, A))>                            \
    {                                                                           \
        typedef                                                                 \
        typename result_of::BOOST_PP_CAT(bound, N)<                             \
            F(BOOST_PP_ENUM_PARAMS(NN, A)) BOOST_PP_ENUM_TRAILING_PARAMS(N, Arg)\
                >::type type;                                                   \
    };                                                                          \
                                                                                \
    template <BOOST_PP_ENUM_PARAMS(NN, typename A)>                             \
    BOOST_FORCEINLINE                                                           \
    typename result_of::BOOST_PP_CAT(bound, N)<                                 \
        F const(BOOST_PP_ENUM_PARAMS(NN, A)) BOOST_PP_ENUM_TRAILING_PARAMS(N, Arg)\
            >::type                                                             \
    operator()(HPX_ENUM_FWD_ARGS(NN, A, a)) const                               \
    {                                                                           \
        typedef                                                                 \
            hpx::util::tuple<                                                   \
                BOOST_PP_ENUM(NN, HPX_UTIL_BIND_REFERENCE, A)                   \
            >                                                                   \
            env_type;                                                           \
        env_type env(HPX_ENUM_FORWARD_ARGS(NN, A, a));                          \
        return util::invoke(f, HPX_UTIL_STRIP(D));                              \
    }                                                                           \
                                                                                \
    template <BOOST_PP_ENUM_PARAMS(NN, typename A)>                             \
    BOOST_FORCEINLINE                                                           \
    typename result_of::BOOST_PP_CAT(bound, N)<                                 \
        F(BOOST_PP_ENUM_PARAMS(NN, A)), BOOST_PP_ENUM_PARAMS(N, Arg)            \
            >::type                                                             \
    operator()(HPX_ENUM_FWD_ARGS(NN, A, a))                                     \
    {                                                                           \
        typedef                                                                 \
            hpx::util::tuple<                                                   \
                BOOST_PP_ENUM(NN, HPX_UTIL_BIND_REFERENCE, A)                   \
            >                                                                   \
            env_type;                                                           \
        env_type env(HPX_ENUM_FORWARD_ARGS(NN, A, a));                          \
        return util::invoke(f, HPX_UTIL_STRIP(D));                              \
    }                                                                           \
/**/

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        namespace result_of
        {
            template <typename F, typename Enable = void>
            struct bound0;

#define HPX_UTIL_BIND_RESULT_OF_BOUND0(Z, NN, D)                                \
            template <                                                          \
                typename F                                                      \
              , BOOST_PP_ENUM_PARAMS(NN, typename A)                            \
            >                                                                   \
            struct bound0<                                                      \
                F(BOOST_PP_ENUM_PARAMS(NN, A))                                  \
            >                                                                   \
            {                                                                   \
                typedef                                                         \
                    typename util::invoke_result_of<F()>::type                  \
                    type;                                                       \
            };                                                                  \
/**/

            BOOST_PP_REPEAT_FROM_TO(
                1
              , HPX_FUNCTION_ARGUMENT_LIMIT
              , HPX_UTIL_BIND_RESULT_OF_BOUND0
              , _
            )

#undef HPX_UTIL_BIND_RESULT_OF_BOUND0

            template <typename F>
            struct bound0<F()
              , typename boost::enable_if<traits::is_callable<F>>::type>
            {
                typedef
                    typename util::invoke_result_of<F()>::type
                    type;
            };
            
            template <typename F>
            struct bound0<F()
              , typename boost::disable_if<traits::is_callable<F>>::type>
            {
                typedef not_callable type;
            };
        }

        template <typename F>
        struct bound0
        {
            typedef typename util::decay<F>::type functor_type;

            functor_type f;

            template <typename Sig>
            struct result;

            template <typename This>
            struct result<This const()>
            {
                typedef
                    typename result_of::bound0<F const()>::type
                    type;
            };

            template <typename This>
            struct result<This()>
            {
                typedef
                    typename result_of::bound0<F()>::type
                    type;
            };

            // default constructor is needed for serialization
            bound0()
            {}

            bound0(bound0 const& other)
                : f(other.f)
            {}

            bound0(BOOST_RV_REF(bound0) other)
                : f(boost::move(other.f))
            {}

            bound0(BOOST_RV_REF(functor_type) f_)
                : f(boost::move(f_))
            {}

            bound0(functor_type const& f_)
                : f(f_)
            {}

            bound0& operator=(BOOST_COPY_ASSIGN_REF(bound0) other)
            {
                f = other.f;
                return *this;
            }

            bound0& operator=(BOOST_RV_REF(bound0) other)
            {
                f = boost::move(other.f);
                return *this;
            }

            BOOST_FORCEINLINE
            typename result_of::bound0<F()>::type
            operator()()
            {
                return util::invoke(f);
            }

            BOOST_FORCEINLINE
            typename result_of::bound0<F const()>::type
            operator()() const
            {
                return util::invoke(f);
            }

#define HPX_UTIL_BIND_FUNCTION_OPERATOR0(Z, NN, D)                              \
    template <typename This BOOST_PP_ENUM_TRAILING_PARAMS(NN, typename A)>      \
    struct result<This const(BOOST_PP_ENUM_PARAMS(NN, A))>                      \
    {                                                                           \
        typedef                                                                 \
        typename result_of::bound0<                                             \
            F const(BOOST_PP_ENUM_PARAMS(NN, A))>::type type;                   \
    };                                                                          \
                                                                                \
    template <typename This BOOST_PP_ENUM_TRAILING_PARAMS(NN, typename A)>      \
    struct result<This(BOOST_PP_ENUM_PARAMS(NN, A))>                            \
    {                                                                           \
        typedef                                                                 \
        typename result_of::bound0<                                             \
            F(BOOST_PP_ENUM_PARAMS(NN, A))>::type type;                         \
    };                                                                          \
                                                                                \
    template <BOOST_PP_ENUM_PARAMS(NN, typename A)>                             \
    BOOST_FORCEINLINE                                                           \
    typename result_of::bound0<                                                 \
        F const(BOOST_PP_ENUM_PARAMS(NN, A))>::type                             \
    operator()(HPX_ENUM_FWD_ARGS(NN, A, a)) const                               \
    {                                                                           \
        typedef                                                                 \
            hpx::util::tuple<                                                   \
                BOOST_PP_ENUM(NN, HPX_UTIL_BIND_REFERENCE, A)                   \
            >                                                                   \
            env_type;                                                           \
        env_type env(HPX_ENUM_FORWARD_ARGS(NN, A, a));                          \
        return util::invoke(f);                                                 \
    }                                                                           \
                                                                                \
    template <BOOST_PP_ENUM_PARAMS(NN, typename A)>                             \
    BOOST_FORCEINLINE                                                           \
    typename result_of::bound0<                                                 \
        F(BOOST_PP_ENUM_PARAMS(NN, A))>::type                                   \
    operator()(HPX_ENUM_FWD_ARGS(NN, A, a))                                     \
    {                                                                           \
        typedef                                                                 \
            hpx::util::tuple<                                                   \
                BOOST_PP_ENUM(NN, HPX_UTIL_BIND_REFERENCE, A)                   \
            >                                                                   \
            env_type;                                                           \
        env_type env(HPX_ENUM_FORWARD_ARGS(NN, A, a));                          \
        return util::invoke(f);                                                 \
    }                                                                           \
/**/

            BOOST_PP_REPEAT_FROM_TO(
                1
              , HPX_FUNCTION_ARGUMENT_LIMIT
              , HPX_UTIL_BIND_FUNCTION_OPERATOR0, _
            )

#undef HPX_UTIL_BIND_FUNCTION_OPERATOR0

        private:
            BOOST_COPYABLE_AND_MOVABLE(bound0)
        };

        namespace result_of
        {
            template <typename Env, typename F>
            struct eval<Env, util::detail::bound0<F> const>
            {
                typedef typename util::invoke_result_of<F const()>::type type;
            };

            template <typename Env, typename F>
            struct eval<Env, util::detail::bound0<F> >
            {
                typedef typename util::invoke_result_of<F()>::type type;
            };
        }

        template <typename Env, typename F>
        typename result_of::eval<Env, util::detail::bound0<F>>::type
        eval(Env& env, util::detail::bound0<F>& f)
        {
            return f();
        }

        template <typename Env, typename F>
        typename result_of::eval<Env, util::detail::bound0<F> const>::type
        eval(Env& env, util::detail::bound0<F> const& f)
        {
            return f();
        }
    }

    template <typename F>
    typename boost::disable_if<
        hpx::traits::is_action<typename detail::remove_reference<F>::type>,
        detail::bound0<typename util::decay<F>::type>
    >::type
    bind(BOOST_FWD_REF(F) f)
    {
        return
            detail::bound0<typename util::decay<F>::type>
                (boost::forward<F>(f));
    }
    
    template <typename R>
    detail::bound0<R(*)()>
    bind(R(*f)())
    {
        return detail::bound0<R(*)()>(f);
    }
}}

namespace hpx { namespace traits
{
    template <
        typename F
    >
    struct is_bind_expression<
        hpx::util::detail::bound0<F>
    > : boost::mpl::true_
    {};
}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    // serialization of the bound object, just serialize the function object
    template <typename F>
    void serialize(hpx::util::portable_binary_iarchive& ar
      , hpx::util::detail::bound0<F>& bound
      , unsigned int const)
    {
        ar& bound.f;
    }

    template <typename F>
    void serialize(hpx::util::portable_binary_oarchive& ar
      , hpx::util::detail::bound0<F>& bound
      , unsigned int const)
    {
        ar& bound.f;
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/preprocessed/bind.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/bind_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/bind.hpp>                                                 \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#undef HPX_UTIL_BIND_EVAL_TYPE
#undef HPX_UTIL_BIND_CONST_EVAL_TYPE
#undef HPX_UTIL_BIND_EVAL
#undef HPX_UTIL_BIND_REMOVE_REFERENCE
#undef HPX_UTIL_BIND_REFERENCE
#undef HPX_UTIL_BIND_FUNCTION_OPERATOR

#endif

#else  // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_FRAME_ITERATION(1)
#define NN BOOST_PP_FRAME_ITERATION(1)

#define HPX_UTIL_BIND_INIT_MEMBER(Z, N, D)                                      \
    BOOST_PP_CAT(arg, N)(boost::forward<BOOST_PP_CAT(A, N)>(BOOST_PP_CAT(a, N)))\
/**/
#define HPX_UTIL_BIND_MEMBER_TYPE(Z, NN, D)                                     \
    BOOST_PP_CAT(Arg, NN)                                                       \
/**/
#define HPX_UTIL_BIND_MEMBER(Z, N, D)                                           \
    typename decay<BOOST_PP_CAT(Arg, N)>::type BOOST_PP_CAT(arg, N);            \
/**/

#define HPX_UTIL_BIND_INIT_COPY_MEMBER(Z, N, D)                                 \
    BOOST_PP_CAT(arg, N)(other.BOOST_PP_CAT(arg, N))                            \
/**/

#define HPX_UTIL_BIND_INIT_MOVE_MEMBER(Z, N, D)                                 \
    BOOST_PP_CAT(arg, N)(boost::move(other.BOOST_PP_CAT(arg, N)))               \
/**/

#define HPX_UTIL_BIND_ASSIGN_MEMBER(Z, N, D)                                    \
    BOOST_PP_CAT(arg, N) = other.BOOST_PP_CAT(arg, N);                          \
/**/

#define HPX_UTIL_BIND_MOVE_MEMBER(Z, N, D)                                      \
    BOOST_PP_CAT(arg, N) = boost::move(other.BOOST_PP_CAT(arg, N));             \
/**/

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    namespace detail
    {
#define HPX_UTIL_BIND_RESULT_OF_BOUND_ARGS(Z, NN, D)                            \
    typename detail::result_of::eval<                                           \
        HPX_UTIL_STRIP(D)                                                       \
      , HPX_UTIL_BIND_MEMBER_TYPE(Z, NN, D)                                     \
    >::type                                                                     \

        namespace result_of
        {
            template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
            struct BOOST_PP_CAT(bound, N);

#define HPX_UTIL_BIND_RESULT_OF_BOUND(Z, NN, D)                                 \
            template <                                                          \
                typename F                                                      \
              BOOST_PP_ENUM_TRAILING_PARAMS(NN, typename A)                     \
              , BOOST_PP_ENUM_PARAMS(N, typename Arg)                           \
            >                                                                   \
            struct BOOST_PP_CAT(bound, N)<                                      \
                F(BOOST_PP_ENUM_PARAMS(NN, A))                                  \
              , BOOST_PP_ENUM_PARAMS(N, Arg)                                    \
            >                                                                   \
            {                                                                   \
                typedef                                                         \
                    hpx::util::tuple<                                           \
                        BOOST_PP_ENUM(NN, HPX_UTIL_BIND_REFERENCE, A)           \
                    >                                                           \
                    env_type;                                                   \
                typedef                                                         \
                    typename boost::mpl::fold<                                  \
                        boost::mpl::vector<                                     \
                            BOOST_PP_ENUM(                                      \
                                N                                               \
                              , HPX_UTIL_BIND_RESULT_OF_BOUND_ARGS              \
                              , env_type                                        \
                            )                                                   \
                        >                                                       \
                      , boost::mpl::false_                                      \
                      , boost::mpl::or_<                                        \
                            boost::mpl::_1                                      \
                          , boost::is_same<boost::mpl::_2, not_enough_arguments>\
                        >                                                       \
                    >::type                                                     \
                    insufficient_arguments;                                     \
                typedef                                                         \
                    typename boost::mpl::eval_if<                               \
                        insufficient_arguments                                  \
                      , boost::mpl::identity<not_enough_arguments>              \
                      , util::invoke_result_of<                                 \
                            F(                                                  \
                                BOOST_PP_ENUM(                                  \
                                    N                                           \
                                  , HPX_UTIL_BIND_RESULT_OF_BOUND_ARGS          \
                                  , env_type                                    \
                                )                                               \
                            )                                                   \
                        >                                                       \
                    >::type                                                     \
                    type;                                                       \
            };                                                                  \
/**/

            BOOST_PP_REPEAT_FROM_TO(
                1
              , HPX_FUNCTION_ARGUMENT_LIMIT
              , HPX_UTIL_BIND_RESULT_OF_BOUND
              , _
            )

            template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
            struct BOOST_PP_CAT(bound, N)<F(), BOOST_PP_ENUM_PARAMS(N, Arg)>
            {
                typedef
                    typename boost::mpl::eval_if<
                        traits::is_callable<F, BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL_TYPE, _)>
                      , util::invoke_result_of<
                            F(BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL_TYPE, _))
                        >
                      , boost::mpl::identity<not_callable>
                    >::type
                    type;
            };

            template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
            struct BOOST_PP_CAT(bound, N)<F const(), BOOST_PP_ENUM_PARAMS(N, Arg)>
            {
                typedef
                    typename boost::mpl::eval_if<
                        traits::is_callable<F const, BOOST_PP_ENUM(N, HPX_UTIL_BIND_CONST_EVAL_TYPE, _)>
                      , util::invoke_result_of<
                            F const(BOOST_PP_ENUM(N, HPX_UTIL_BIND_CONST_EVAL_TYPE, _))
                        >
                      , boost::mpl::identity<not_callable>
                    >::type
                    type;
            };
        }

        template <
            typename F
          , BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        struct BOOST_PP_CAT(bound, N)
        {
            typedef typename util::decay<F>::type functor_type;

            functor_type f;

            template <typename Sig>
            struct result;

            template <typename This>
            struct result<This const()>
            {
                typedef
                    typename result_of::BOOST_PP_CAT(bound, N)<
                        F const()
                      , BOOST_PP_ENUM_PARAMS(N, Arg)
                    >::type
                    type;
            };

            template <typename This>
            struct result<This()>
            {
                typedef
                    typename result_of::BOOST_PP_CAT(bound, N)<
                        F()
                      , BOOST_PP_ENUM_PARAMS(N, Arg)
                    >::type
                    type;
            };

            // default constructor is needed for serialization
            BOOST_PP_CAT(bound, N)()
            {}

            template <BOOST_PP_ENUM_PARAMS(N, typename A)>
            BOOST_PP_CAT(bound, N)(
                BOOST_RV_REF(functor_type) f_
              , HPX_ENUM_FWD_ARGS(N, A, a)
            )
                : f(boost::move(f_))
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MEMBER, _)
            {}

            template <BOOST_PP_ENUM_PARAMS(N, typename A)>
            BOOST_PP_CAT(bound, N)(
                functor_type const& f_
              , HPX_ENUM_FWD_ARGS(N, A, a)
            )
                : f(f_)
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MEMBER, _)
            {}

            BOOST_PP_CAT(bound, N)(
                    BOOST_PP_CAT(bound, N) const& other)
                : f(other.f)
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_COPY_MEMBER, _)
            {}

            BOOST_PP_CAT(bound, N)(
                    BOOST_RV_REF(BOOST_PP_CAT(bound, N)) other)
                : f(boost::move(other.f))
                , BOOST_PP_ENUM(N, HPX_UTIL_BIND_INIT_MOVE_MEMBER, _)
            {}

            BOOST_PP_CAT(bound, N)& operator=(
                BOOST_COPY_ASSIGN_REF(BOOST_PP_CAT(bound, N)) other)
            {
                f = other.f;
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_ASSIGN_MEMBER, _)
                return *this;
            }

            BOOST_PP_CAT(bound, N)& operator=(
                BOOST_RV_REF(BOOST_PP_CAT(bound, N)) other)
            {
                f = boost::move(other.f);
                BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MOVE_MEMBER, _)
                return *this;
            }

            BOOST_FORCEINLINE
            typename result_of::BOOST_PP_CAT(bound, N)<
                F()
              , BOOST_PP_ENUM_PARAMS(N, Arg)
            >::type
            operator()()
            {
                typedef hpx::util::tuple<> env_type;
                env_type env;
                return util::invoke(f, BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL, _));
            }

            BOOST_FORCEINLINE
            typename result_of::BOOST_PP_CAT(bound, N)<
                F const()
              , BOOST_PP_ENUM_PARAMS(N, Arg)
            >::type
            operator()() const
            {
                typedef hpx::util::tuple<> env_type;
                env_type env;
                return util::invoke(f, BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL, _));
            }

            BOOST_PP_REPEAT_FROM_TO(
                1
              , HPX_FUNCTION_ARGUMENT_LIMIT
              , HPX_UTIL_BIND_FUNCTION_OPERATOR
              , (BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL, _))
            )

            BOOST_PP_REPEAT(N, HPX_UTIL_BIND_MEMBER, _)
        };

        namespace result_of
        {
            template <typename Env
              , typename F
              , BOOST_PP_ENUM_PARAMS(N, typename Arg)
            >
            struct eval<
                Env
              , BOOST_PP_CAT(detail::bound, N)<
                    F
                  , BOOST_PP_ENUM_PARAMS(N, Arg)
                > const
            >
            {
                typedef
                    typename boost::fusion::result_of::invoke_function_object<
                        BOOST_PP_CAT(detail::bound, N)<
                            F
                          , BOOST_PP_ENUM_PARAMS(N, Arg)
                        > const&
                      , Env
                    >::type
                    type;
            };

            template <typename Env
              , typename F
              , BOOST_PP_ENUM_PARAMS(N, typename Arg)
            >
            struct eval<
                Env
              , BOOST_PP_CAT(detail::bound, N)<
                    F
                  , BOOST_PP_ENUM_PARAMS(N, Arg)
                >
            >
            {
                typedef
                    typename boost::fusion::result_of::invoke_function_object<
                        BOOST_PP_CAT(detail::bound, N)<
                            F
                          , BOOST_PP_ENUM_PARAMS(N, Arg)
                        >&
                      , Env
                    >::type
                    type;
            };
        }

        template <
            typename Env
          , typename F
          , BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        typename result_of::eval<
            Env,
            BOOST_PP_CAT(detail::bound, N)<
                F
              , BOOST_PP_ENUM_PARAMS(N, Arg)
            > const
        >::type
        eval(
            Env& env
          , BOOST_PP_CAT(detail::bound, N)<
                F
              , BOOST_PP_ENUM_PARAMS(N, Arg)
            > const& f
        )
        {
            return
                boost::fusion::invoke_function_object<
                    BOOST_PP_CAT(detail::bound, N)<
                        F
                      , BOOST_PP_ENUM_PARAMS(N, Arg)
                    > const&
                >(f, env);
        }

        template <
            typename Env
          , typename F
          , BOOST_PP_ENUM_PARAMS(N, typename Arg)
        >
        typename result_of::eval<
            Env,
            BOOST_PP_CAT(detail::bound, N)<
                F
              , BOOST_PP_ENUM_PARAMS(N, Arg)
            >
        >::type
        eval(
            Env& env
          , BOOST_PP_CAT(detail::bound, N)<
                F
              , BOOST_PP_ENUM_PARAMS(N, Arg)
            >& f
        )
        {
            return
                boost::fusion::invoke_function_object<
                    BOOST_PP_CAT(detail::bound, N)<
                        F
                      , BOOST_PP_ENUM_PARAMS(N, Arg)
                    >&
                >(f, env);
        }
    }

    template <typename F
      , BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    typename boost::disable_if<
        hpx::traits::is_action<typename detail::remove_reference<F>::type>
      , BOOST_PP_CAT(detail::bound, N)<
            typename util::decay<F>::type
          , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
        >
    >::type
    bind(
        BOOST_FWD_REF(F) f
      , HPX_ENUM_FWD_ARGS(N, A, a)
    )
    {
        return
            BOOST_PP_CAT(detail::bound, N)<
                typename util::decay<F>::type
              , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
            >(
                boost::forward<F>(f)
              , HPX_ENUM_FORWARD_ARGS(N, A, a)
            );
    }

    template <
        typename R
      , BOOST_PP_ENUM_PARAMS(N, typename T)
      , BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    BOOST_PP_CAT(detail::bound, N)<
        R(*)(BOOST_PP_ENUM_PARAMS(N, T))
      , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
    >
    bind(
        R(*f)(BOOST_PP_ENUM_PARAMS(N, T))
      , HPX_ENUM_FWD_ARGS(N, A, a)
    )
    {
        return
            BOOST_PP_CAT(detail::bound, N)<
                R(*)(BOOST_PP_ENUM_PARAMS(N, T))
              , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
            >
            (f, HPX_ENUM_FORWARD_ARGS(N, A, a));
    }
    
    template <
        typename R
      , typename C
      , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename T)
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(N)) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    BOOST_PP_CAT(detail::bound, N)<
        R(C::*)(BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T))
      , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
    >
    bind(
        R(C::*f)(BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T))
      , HPX_ENUM_FWD_ARGS(N, A, a)
    )
    {
        return
            BOOST_PP_CAT(detail::bound, N)<
                R(C::*)(BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T))
              , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
            >
            (f, HPX_ENUM_FORWARD_ARGS(N, A, a));
    }
    
    template <
        typename R
      , typename C
      , BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename T)
      BOOST_PP_COMMA_IF(BOOST_PP_DEC(N)) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    BOOST_PP_CAT(detail::bound, N)<
        R(C::*)(BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)) const
      , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
    >
    bind(
        R(C::*f)(BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)) const
      , HPX_ENUM_FWD_ARGS(N, A, a)
    )
    {
        return
            BOOST_PP_CAT(detail::bound, N)<
                R(C::*)(BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)) const
              , BOOST_PP_ENUM(N, HPX_UTIL_BIND_REMOVE_REFERENCE, A)
            >
            (f, HPX_ENUM_FORWARD_ARGS(N, A, a));
    }
}}

namespace hpx { namespace traits
{
    template <
        typename F
      , BOOST_PP_ENUM_PARAMS(N, typename Arg)
    >
    struct is_bind_expression<
        hpx::util::detail::BOOST_PP_CAT(bound, N)<
            F
          , BOOST_PP_ENUM_PARAMS(N, Arg)
        >
    > : boost::mpl::true_
    {};
}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
#define HPX_UTIL_BIND_SERIALIZE_MEMBER(Z, NNN, _) ar& BOOST_PP_CAT(bound.arg, NNN);

    // serialization of the bound object, just serialize function object and
    // members
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void serialize(hpx::util::portable_binary_iarchive& ar
      , BOOST_PP_CAT(hpx::util::detail::bound, N)<
            F, BOOST_PP_ENUM_PARAMS(N, Arg)
        >& bound
      , unsigned int const)
    {
        ar& bound.f;
        BOOST_PP_REPEAT(N, HPX_UTIL_BIND_SERIALIZE_MEMBER, _)
    }

    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void serialize(hpx::util::portable_binary_oarchive& ar
      , BOOST_PP_CAT(hpx::util::detail::bound, N)<
            F, BOOST_PP_ENUM_PARAMS(N, Arg)
        >& bound
      , unsigned int const)
    {
        ar& bound.f;
        BOOST_PP_REPEAT(N, HPX_UTIL_BIND_SERIALIZE_MEMBER, _)
    }

#undef HPX_UTIL_BIND_SERIALIZE_MEMBER
}}

#undef HPX_UTIL_BIND_INIT_MEMBER
#undef HPX_UTIL_BIND_MEMBER_TYPE
#undef HPX_UTIL_BIND_MEMBER
#undef HPX_UTIL_BIND_INIT_COPY_MEMBER
#undef HPX_UTIL_BIND_INIT_MOVE_MEMBER
#undef HPX_UTIL_BIND_ASSIGN_MEMBER
#undef HPX_UTIL_BIND_MOVE_MEMBER

#undef NN
#undef N

#endif
