//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_BIND_HPP
#define HPX_UTIL_BIND_HPP

#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_bind_expression.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_placeholder.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/result_of.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct cannot_be_called {};

        struct not_enough_arguments {};

        template <typename F>
        class one_shot_wrapper;

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename T>
        struct bind_eval_bound_impl
        {
            typedef T& type;

            template <typename UnboundArgs>
            static BOOST_FORCEINLINE
            type call(T& t, UnboundArgs && /*unbound_args*/)
            {
                    return t;
            }
        };

        template <typename F, typename T>
        struct bind_eval_bound_impl<one_shot_wrapper<F>, T>
        {
            typedef T && type;

            template <typename UnboundArgs>
            static BOOST_FORCEINLINE
            type call(T& t, UnboundArgs && /*unbound_args*/)
            {
                return std::move(t);
            }
        };

        template <
            typename F
          , typename T, typename UnboundArgs
          , typename Enable = void
        >
        struct bind_eval_impl
          : bind_eval_bound_impl<F, T>
        {};

        template <std::size_t I, typename UnboundArgs, typename Enable = void>
        struct bind_eval_placeholder_impl
        {
            typedef
                typename util::tuple_element<
                    I
                  , typename util::decay<UnboundArgs>::type
                >::type&&
                type;

            template <typename T>
            static BOOST_FORCEINLINE
            type call(T& /*t*/, UnboundArgs && unbound_args)
            {
                return util::get<I>(std::forward<UnboundArgs>(unbound_args));
            }
        };
        template <std::size_t I, typename UnboundArgs>
        struct bind_eval_placeholder_impl<
            I, UnboundArgs
          , typename boost::enable_if_c<
                I >= util::tuple_size<UnboundArgs>::value
            >::type
        >
        {
            typedef not_enough_arguments type;

            template <typename T>
            static BOOST_FORCEINLINE
            type call(T& /*t*/, UnboundArgs && /*unbound_args*/)
            {
                return not_enough_arguments();
            }
        };

        template <typename F, typename T, typename UnboundArgs>
        struct bind_eval_impl<
            F, T, UnboundArgs
          , typename boost::enable_if_c<
                traits::is_placeholder<
                    typename boost::remove_const<T>::type
                >::value != 0
            >::type
        > : bind_eval_placeholder_impl<
                traits::is_placeholder<
                    typename boost::remove_const<T>::type
                >::value - 1
              , UnboundArgs
            >
        {};

        template <typename F, typename T, typename UnboundArgs>
        struct bind_eval_impl<
            F, T, UnboundArgs
          , typename boost::enable_if_c<
                traits::is_bind_expression<
                    typename boost::remove_const<T>::type
                >::value
            >::type
        >
        {
            typedef
                typename util::invoke_fused_result_of<T&(UnboundArgs)>::type
                type;

            static BOOST_FORCEINLINE
            type call(T& t, UnboundArgs && unbound_args)
            {
                return util::invoke_fused
                    (t, std::forward<UnboundArgs>(unbound_args));
            }
        };

        template <typename F, typename T, typename UnboundArgs>
        struct bind_eval_impl<
            F, T, UnboundArgs
          , typename boost::enable_if_c<
                boost::is_reference_wrapper<
                    typename boost::remove_const<T>::type
                >::value
            >::type
        >
        {
            typedef
                typename boost::unwrap_reference<T>::type&
                type;

            static BOOST_FORCEINLINE
            type call(T& t, UnboundArgs && /*unbound_args*/)
            {
                return t.get();
            }
        };

        template <typename F, typename T, typename UnboundArgs>
        BOOST_FORCEINLINE
        typename bind_eval_impl<F, T, UnboundArgs>::type
        bind_eval(T& t, UnboundArgs && unbound_args)
        {
            return
                bind_eval_impl<F, T, UnboundArgs>::call
                    (t, std::forward<UnboundArgs>(unbound_args));
        }

        ///////////////////////////////////////////////////////////////////////
        template <
            typename F, typename BoundArgs, typename UnboundArgs
          , typename Enable = void
        >
        struct bind_invoke_impl;

        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl<
            F, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 0
            >::type
        >
        {
            typedef
                typename util::detail::result_of_or<
                    F()
                  , cannot_be_called
                >::type
                type;

            static BOOST_FORCEINLINE
            type call(
                F& f, BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return util::invoke_r<type>(f);
            }
        };

        template <typename F, typename BoundArgs, typename UnboundArgs>
        BOOST_FORCEINLINE
        typename bind_invoke_impl<F, BoundArgs, UnboundArgs>::type
        bind_invoke(
            F& f, BoundArgs& bound_args
          , UnboundArgs && unbound_args
        )
        {
            return
                bind_invoke_impl<F, BoundArgs, UnboundArgs>::call(
                    f, bound_args
                  , std::forward<UnboundArgs>(unbound_args)
                );
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F>
        class one_shot_wrapper
        {
        public:
#           if !defined(HPX_DISABLE_ASSERTS)
            // default constructor is needed for serialization
            one_shot_wrapper()
              : _called(false)
            {}

            explicit one_shot_wrapper(F const& f)
              : _f(f)
              , _called(false)
            {}
            explicit one_shot_wrapper(F && f)
              : _f(std::move(f))
              , _called(false)
            {}

            one_shot_wrapper(one_shot_wrapper const& other)
              : _f(other._f)
              , _called(other._called)
            {}
            one_shot_wrapper(one_shot_wrapper && other)
              : _f(std::move(other._f))
              , _called(other._called)
            {
                other._called = true;
            }

            void check_call()
            {
                HPX_ASSERT(!_called);

                _called = true;
            }
#           else
            // default constructor is needed for serialization
            one_shot_wrapper()
            {}

            explicit one_shot_wrapper(F const& f)
              : _f(f)
            {}
            explicit one_shot_wrapper(F && f)
              : _f(std::move(f))
            {}

            one_shot_wrapper(one_shot_wrapper const& other)
              : _f(other._f)
            {}
            one_shot_wrapper(one_shot_wrapper && other)
              : _f(std::move(other._f))
            {}

            void check_call()
            {}
#           endif

            template <typename T>
            struct result;

            template <typename This>
            struct result<This()>
              : util::detail::result_of_or<F(), cannot_be_called>
            {};

            template <typename This>
            struct result<This const()>
              : boost::mpl::identity<cannot_be_called>
            {};

            BOOST_FORCEINLINE
            typename result<one_shot_wrapper()>::type
            operator()()
            {
                typedef typename result<one_shot_wrapper()>::type result_type;

                check_call();
                return util::invoke_r<result_type>(_f);
            }

#           define HPX_UTIL_BIND_ONE_SHOT_WRAPPER_FUNCTION_OP(Z, N, D)        \
            template <typename This, BOOST_PP_ENUM_PARAMS(N, typename T)>     \
            struct result<This(BOOST_PP_ENUM_PARAMS(N, T))>                   \
              : util::detail::result_of_or<                                   \
                    F(BOOST_PP_ENUM_PARAMS(N, T))                             \
                  , cannot_be_called                                          \
                >                                                             \
            {};                                                               \
                                                                              \
            template <typename This, BOOST_PP_ENUM_PARAMS(N, typename T)>     \
            struct result<This const(BOOST_PP_ENUM_PARAMS(N, T))>             \
              : boost::mpl::identity<cannot_be_called>                        \
            {};                                                               \
                                                                              \
            template <BOOST_PP_ENUM_PARAMS(N, typename T)>                    \
            BOOST_FORCEINLINE                                                 \
            typename result<one_shot_wrapper(BOOST_PP_ENUM_PARAMS(N, T))>::type\
            operator()(HPX_ENUM_FWD_ARGS(N, T, t))                            \
            {                                                                 \
                typedef typename result<                                      \
                    one_shot_wrapper(BOOST_PP_ENUM_PARAMS(N, T))              \
                >::type result_type;                                          \
                check_call();                                                 \
                return util::invoke_r<result_type>(_f,                        \
                    HPX_ENUM_FORWARD_ARGS(N, T, t));                          \
            }                                                                 \
            /**/

            BOOST_PP_REPEAT_FROM_TO(
                1, HPX_FUNCTION_ARGUMENT_LIMIT
              , HPX_UTIL_BIND_ONE_SHOT_WRAPPER_FUNCTION_OP, _
            );

#           undef HPX_UTIL_BIND_ONE_SHOT_WRAPPER_FUNCTION_OP

        public: // exposition-only
            F _f;
#           if !defined(HPX_DISABLE_ASSERTS)
            bool _called;
#           endif
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename BoundArgs>
        class bound
        {
        public:
            // default constructor is needed for serialization
            bound()
            {}

            template <typename F_, typename BoundArgs_>
            explicit bound(
                F_ && f
              , BoundArgs_ && bound_args
            ) : _f(std::forward<F_>(f))
              , _bound_args(std::forward<BoundArgs_>(bound_args))
            {}

            bound(bound const& other)
              : _f(other._f)
              , _bound_args(other._bound_args)
            {}
            bound(bound && other)
              : _f(std::move(other._f))
              , _bound_args(std::move(other._bound_args))
            {}

            template <typename>
            struct result;

            // bring in the definition for all overloads for operator()
            #include <hpx/util/detail/define_bind_function_operators.hpp>

        public: // exposition-only
            F _f;
            BoundArgs _bound_args;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <std::size_t I>
        struct placeholder
        {
            static std::size_t const value = I;
        };
    }

    namespace placeholders
    {
#       define HPX_UTIL_BIND_PLACEHOLDER(Z, N, D)                             \
        detail::placeholder<N> const BOOST_PP_CAT(_, N) = {};                 \
        /**/

        BOOST_PP_REPEAT_FROM_TO(
            1, HPX_FUNCTION_ARGUMENT_LIMIT
          , HPX_UTIL_BIND_PLACEHOLDER, _
        );

#       undef HPX_UTIL_BIND_PLACEHOLDER
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    typename boost::disable_if_c<
        traits::is_action<typename boost::remove_reference<F>::type>::value
      , detail::bound<
            typename util::decay<F>::type
          , util::tuple<>
        >
    >::type
    bind(F && f)
    {
        typedef
            detail::bound<
                typename util::decay<F>::type
              , util::tuple<>
            >
            result_type;

        return result_type(std::forward<F>(f), util::forward_as_tuple());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    detail::one_shot_wrapper<typename util::decay<F>::type>
    one_shot(F && f)
    {
        typedef
            detail::one_shot_wrapper<typename util::decay<F>::type>
            result_type;

        return result_type(std::forward<F>(f));
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename F, typename BoundArgs>
    struct is_bind_expression<util::detail::bound<F, BoundArgs> >
      : boost::mpl::true_
    {};

    template <std::size_t I>
    struct is_placeholder<util::detail::placeholder<I> >
      : util::detail::placeholder<I>
    {};
}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    // serialization of the bound object
    template <typename F, typename BoundArgs>
    void serialize(
        ::hpx::util::portable_binary_iarchive& ar
      , ::hpx::util::detail::bound<F, BoundArgs>& bound
      , unsigned int const /*version*/)
    {
        ar >> bound._f;
        ar >> bound._bound_args;
    }

    template <typename F, typename BoundArgs>
    void serialize(
        ::hpx::util::portable_binary_oarchive& ar
      , ::hpx::util::detail::bound<F, BoundArgs>& bound
      , unsigned int const /*version*/)
    {
        ar << bound._f;
        ar << bound._bound_args;
    }

    // serialization of the bound object
    template <typename F>
    void serialize(
        ::hpx::util::portable_binary_iarchive& ar
      , ::hpx::util::detail::one_shot_wrapper<F>& one_shot_wrapper
      , unsigned int const /*version*/)
    {
        ar >> one_shot_wrapper._f;
        ar >> one_shot_wrapper._called;
    }

    template <typename F>
    void serialize(
        ::hpx::util::portable_binary_oarchive& ar
      , ::hpx::util::detail::one_shot_wrapper<F>& one_shot_wrapper
      , unsigned int const /*version*/)
    {
        ar << one_shot_wrapper._f;
        ar << one_shot_wrapper._called;
    }

    // serialization of placeholders is trivial, just provide empty functions
    template <std::size_t I>
    void serialize(
        ::hpx::util::portable_binary_iarchive& ar
      , ::hpx::util::detail::placeholder<I>& bound
      , unsigned int const /*version*/)
    {}

    template <std::size_t I>
    void serialize(
        ::hpx::util::portable_binary_oarchive& ar
      , ::hpx::util::detail::placeholder<I>& bound
      , unsigned int const /*version*/)
    {}
}}

#   if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#       include <hpx/util/preprocessed/bind.hpp>
#   else
#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(preserve: 1, line: 0, output: "preprocessed/bind_" HPX_LIMIT_STR ".hpp")
#       endif

        ///////////////////////////////////////////////////////////////////////
#       define BOOST_PP_ITERATION_PARAMS_1                                    \
        (                                                                     \
            3                                                                 \
          , (                                                                 \
                1                                                             \
              , HPX_FUNCTION_ARGUMENT_LIMIT                                   \
              , <hpx/util/bind.hpp>                                           \
            )                                                                 \
        )                                                                     \
        /**/
#       include BOOST_PP_ITERATE()

#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(output: null)
#       endif
#   endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util
{
    namespace detail
    {
#       define HPX_UTIL_BIND_EVAL_TYPE(Z, N, D)                               \
        typename detail::bind_eval_impl<                                      \
            F, typename util::tuple_element<N, BoundArgs>::type               \
          , UnboundArgs                                                       \
        >::type                                                               \
        /**/
#       define HPX_UTIL_BIND_EVAL(Z, N, D)                                    \
        detail::bind_eval<F>(                                                 \
            util::get<N>(bound_args)                                          \
          , std::forward<UnboundArgs>(unbound_args)                         \
        )                                                                     \
        /**/
        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl<
            F, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == N
            >::type
        >
        {
            typedef
                typename util::detail::result_of_or<
                    F(BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL_TYPE, _))
                  , cannot_be_called
                >::type
                type;

            static BOOST_FORCEINLINE
            type call(
                F& f, BoundArgs& bound_args
              , UnboundArgs && unbound_args
            )
            {
                return util::invoke_r<type>(f,
                    BOOST_PP_ENUM(N, HPX_UTIL_BIND_EVAL, _));
            }
        };

        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl<
            one_shot_wrapper<F> const, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == N
            >::type
        >
        {
            typedef cannot_be_called type;
        };
#       undef HPX_UTIL_BIND_EVAL_TYPE
#       undef HPX_UTIL_BIND_EVAL
    }

#   define HPX_UTIL_BIND_DECAY(Z, N, D)                                       \
    typename util::decay<BOOST_PP_CAT(T, N)>::type                            \
    /**/
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename T)>
    typename boost::disable_if_c<
        traits::is_action<typename boost::remove_reference<F>::type>::value
      , detail::bound<
            typename util::decay<F>::type
          , util::tuple<BOOST_PP_ENUM(N, HPX_UTIL_BIND_DECAY, _)>
        >
    >::type
    bind(F && f, HPX_ENUM_FWD_ARGS(N, T, t))
    {
        typedef
            detail::bound<
                typename util::decay<F>::type
              , util::tuple<BOOST_PP_ENUM(N, HPX_UTIL_BIND_DECAY, _)>
            >
            result_type;

        return
            result_type(
                std::forward<F>(f)
              , util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, T, t))
            );
    }
#   undef HPX_UTIL_BIND_DECAY
}}

#undef N

#endif
