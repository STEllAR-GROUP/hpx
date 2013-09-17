//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_INVOKE_HPP
#define HPX_UTIL_INVOKE_HPP

#include <hpx/config.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/util/detail/qualify_as.hpp>

#include <boost/mem_fn.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/is_function.hpp>
#include <boost/type_traits/is_member_function_pointer.hpp>
#include <boost/type_traits/is_member_object_pointer.hpp>
#include <boost/type_traits/is_member_pointer.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/result_of.hpp>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Q = int, typename Enable = void>
        struct get_member_pointer_object
        {
            typedef
                typename detail::qualify_as<
                    typename get_member_pointer_object<T>::type
                  , typename boost::mpl::if_<
                        boost::is_pointer<
                            typename boost::remove_reference<Q>::type>
                      , typename boost::remove_pointer<
                            typename boost::remove_reference<Q>::type>::type&
                      , Q
                    >::type
                >::type
                type;
        };
        
        template <typename T, typename Q>
        struct get_member_pointer_object<T, Q
          , typename boost::enable_if<
                boost::is_reference_wrapper<typename util::decay<Q>::type>>::type
        > : get_member_pointer_object<T
              , typename boost::unwrap_reference<typename util::decay<Q>::type>::type&>
        {};
        
        // Note: boost::mem_fn is not rvref-aware
        template <typename T, typename Q>
        struct get_member_pointer_object<T, Q&&>
          : get_member_pointer_object<T, Q const&>
        {};

        template <typename T, typename C>
        struct get_member_pointer_object<T C::*>
        {
            typedef T type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename FD, typename F, typename Enable = void>
        struct invoke_result_of_impl
          : boost::result_of<F>
        {};

#       define HPX_UTIL_INVOKE_RESULT_OF_IMPL(z, n, data)                     \
        template <typename FD                                                 \
          , typename F BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Arg)>        \
        struct invoke_result_of_impl<FD, F(BOOST_PP_ENUM_PARAMS(n, Arg))      \
          , typename boost::enable_if<boost::is_reference_wrapper<FD>>::type  \
        > : boost::result_of<                                                 \
                typename boost::unwrap_reference<FD>::type&                   \
                    (BOOST_PP_ENUM_PARAMS(n, Arg))                            \
            >                                                                 \
        {};                                                                   \
                                                                              \
        /* workaround for tricking result_of into using decltype */           \
        template <typename FD                                                 \
          , typename F BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Arg)>        \
        struct invoke_result_of_impl<FD, F(BOOST_PP_ENUM_PARAMS(n, Arg))      \
          , typename boost::enable_if<                                        \
                boost::mpl::or_<                                              \
                    boost::is_function<typename boost::remove_pointer<FD>::type>\
                  , boost::is_member_function_pointer<FD>                     \
                >                                                             \
            >::type                                                           \
        > : boost::result_of<                                                 \
                FD(BOOST_PP_ENUM_PARAMS(n, Arg))                              \
            >                                                                 \
        {};                                                                   \
        /**/

        BOOST_PP_REPEAT(
            HPX_PP_ROUND_UP_ADD3(HPX_FUNCTION_ARGUMENT_LIMIT)
          , HPX_UTIL_INVOKE_RESULT_OF_IMPL, _
        )

#       undef HPX_UTIL_INVOKE_RESULT_OF_IMPL

        // Note: boost::result_of differs form std::result_of,
        // ignoring member-object-ptrs
        template <typename FD, typename F, typename Class>
        struct invoke_result_of_impl<FD, F(Class)
          , typename boost::enable_if<boost::is_member_object_pointer<FD>>::type
        > : detail::get_member_pointer_object<FD, Class>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    struct invoke_result_of;

#   define HPX_UTIL_INVOKE_RESULT_OF(z, n, data)                              \
    template <typename F BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Arg)>      \
    struct invoke_result_of<F(BOOST_PP_ENUM_PARAMS(n, Arg))>                  \
      : detail::invoke_result_of_impl<                                        \
            typename util::decay<F>::type                                     \
          , F(BOOST_PP_ENUM_PARAMS(n, Arg))                                   \
        >                                                                     \
    {};                                                                       \
    /**/

    BOOST_PP_REPEAT(
        HPX_PP_ROUND_UP_ADD3(HPX_FUNCTION_ARGUMENT_LIMIT)
      , HPX_UTIL_INVOKE_RESULT_OF, _
    )

#   undef HPX_UTIL_INVOKE_RESULT_OF

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::mpl::or_<
            boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , invoke_result_of<F()>
    >::type
    invoke(BOOST_FWD_REF(F) f)
    {
        return boost::forward<F>(f)();
    };

    template <typename F>
    BOOST_FORCEINLINE
    typename boost::lazy_enable_if<
        boost::is_member_pointer<typename util::decay<F>::type>
      , invoke_result_of<F()>
    >::type
    invoke(BOOST_FWD_REF(F) f)
    {
        return boost::mem_fn(boost::forward<F>(f))();
    };

    template <typename F>
    BOOST_FORCEINLINE
    typename boost::lazy_enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , invoke_result_of<F()>
    >::type
    invoke(BOOST_FWD_REF(F) f)
    {
        return (f.get())();
    };

    template <typename R, typename F>
    BOOST_FORCEINLINE
    R
    invoke_r(BOOST_FWD_REF(F) f)
    {
        return util::void_guard<R>(), util::invoke(boost::forward<F>(f));
    };
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/preprocessed/invoke.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/invoke_" HPX_LIMIT_STR ".hpp")
#endif

///////////////////////////////////////////////////////////////////////////////
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (                                                                         \
        3                                                                     \
      , (                                                                     \
            1                                                                 \
          , HPX_PP_ROUND_UP_ADD3(HPX_FUNCTION_ARGUMENT_LIMIT)                 \
          , <hpx/util/invoke.hpp>                                             \
        )                                                                     \
    )                                                                         \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::mpl::or_<
            boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , invoke_result_of<
            F(HPX_ENUM_FWD_ARGS(N, Arg, BOOST_PP_INTERCEPT))
        >
    >::type
    invoke(BOOST_FWD_REF(F) f, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            boost::forward<F>(f)
                (HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    };

    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    typename boost::lazy_enable_if<
        boost::is_member_pointer<typename util::decay<F>::type>
      , invoke_result_of<
            F(HPX_ENUM_FWD_ARGS(N, Arg, BOOST_PP_INTERCEPT))
        >
    >::type
    invoke(BOOST_FWD_REF(F) f, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            boost::mem_fn(boost::forward<F>(f))
                (HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    };

    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    typename boost::lazy_enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , invoke_result_of<
            F(HPX_ENUM_FWD_ARGS(N, Arg, BOOST_PP_INTERCEPT))
        >
    >::type
    invoke(BOOST_FWD_REF(F) f, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            (f.get())
                (HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    };

    template <typename R, typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    BOOST_FORCEINLINE
    R
    invoke_r(BOOST_FWD_REF(F) f, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            util::void_guard<R>(), util::invoke(
                boost::forward<F>(f)
              , HPX_ENUM_FORWARD_ARGS(N, Arg, arg)
            );
    };
}}

#undef N

#endif
