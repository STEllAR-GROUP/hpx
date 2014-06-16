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
#include <hpx/util/result_of.hpp>
#include <hpx/util/void_guard.hpp>

#include <boost/get_pointer.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp> // for get_pointer(shared_ptr<X>)
#include <boost/type_traits/is_function.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_member_pointer.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/utility/enable_if.hpp>

#if !defined(HPX_GCC_VERSION)
#   define HPX_UTIL_INVOKE_FORCEINLINE BOOST_FORCEINLINE
#else
#   define HPX_UTIL_INVOKE_FORCEINLINE inline
#endif

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    struct invoke_result_of
      : hpx::util::result_of<F>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, typename FR>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)())
    {
        return util::void_guard<R>(), f();
    }

    template <typename R, typename FR, typename C, typename T>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR C::*f, T && t)
    {
        return util::void_guard<R>(), (std::forward<T>(t).*f);
    }
    template <typename R, typename FR, typename C, typename T>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR C::*f, T && t)
    {
        using boost::get_pointer;
        return util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f);
    }

    template <typename R, typename FR, typename C, typename T>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(), T && t)
    {
        return util::void_guard<R>(), (std::forward<T>(t).*f)();
    }
    template <typename R, typename FR, typename C, typename T>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(), T && t)
    {
        using boost::get_pointer;
        return util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)();
    }
    template <typename R, typename FR, typename C, typename T>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)() const, T && t)
    {
        return util::void_guard<R>(), (std::forward<T>(t).*f)();
    }
    template <typename R, typename FR, typename C, typename T>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)() const, T && t)
    {
        using boost::get_pointer;
        return util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)();
    }

    template <typename R, typename F>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f)
    {
        return util::void_guard<R>(), (f.get())();
    }

    template <typename R, typename F>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f)
    {
        return util::void_guard<R>(), std::forward<F>(f)();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)())
    {
        return util::invoke_r<R>(f);
    }

    template <typename R, typename C, typename T>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename invoke_result_of<R C::*(T)>::type
    invoke(R C::*f, T && t)
    {
        typedef
            typename invoke_result_of<R C::*(T)>::type
            result_type;

        return util::invoke_r<result_type>(f, std::forward<T>(t));
    }

    template <typename R, typename C, typename T>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(), T && t)
    {
        return util::invoke_r<R>(f, std::forward<T>(t));
    }
    template <typename R, typename C, typename T>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)() const, T && t)
    {
        return util::invoke_r<R>(f, std::forward<T>(t));
    }

    template <typename F>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F()>::type
    >::type
    invoke(F && f)
    {
        typedef
            typename invoke_result_of<F()>::type
            result_type;

        return util::invoke_r<result_type>(std::forward<F>(f));
    }
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
          , HPX_FUNCTION_ARGUMENT_LIMIT                                       \
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

#undef HPX_UTIL_INVOKE_FORCEINLINE

#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename R, typename FR, BOOST_PP_ENUM_PARAMS(N, typename A)
      , BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(BOOST_PP_ENUM_PARAMS(N, A))
      , HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            util::void_guard<R>(), f
                (HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

#   if N < HPX_FUNCTION_ARGUMENT_LIMIT
    template <typename R, typename FR
      , typename C, BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename T, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(BOOST_PP_ENUM_PARAMS(N, A))
      , T && t, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }
    template <typename R, typename FR
      , typename C, BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename T, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(BOOST_PP_ENUM_PARAMS(N, A))
      , T && t, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }
    template <typename R, typename FR
      , typename C, BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename T, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(BOOST_PP_ENUM_PARAMS(N, A)) const
      , T && t, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }
    template <typename R, typename FR
      , typename C, BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename T, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(BOOST_PP_ENUM_PARAMS(N, A)) const
      , T && t, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }
#   endif /*N < HPX_FUNCTION_ARGUMENT_LIMIT*/

    template <typename R, typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            util::void_guard<R>(), (f.get())
                (HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename R, typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, BOOST_PP_ENUM_PARAMS(N, typename A)
      , BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(BOOST_PP_ENUM_PARAMS(N, A))
      , HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            util::invoke_r<R>(
                f
              , HPX_ENUM_FORWARD_ARGS(N, Arg, arg)
            );
    }

#   if N < HPX_FUNCTION_ARGUMENT_LIMIT
    template <typename R, typename C, BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename T, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(BOOST_PP_ENUM_PARAMS(N, A))
      , T && t, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , HPX_ENUM_FORWARD_ARGS(N, Arg, arg)
            );
    }
    template <typename R, typename C, BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename T, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(BOOST_PP_ENUM_PARAMS(N, A)) const
      , T && t, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , HPX_ENUM_FORWARD_ARGS(N, Arg, arg)
            );
    }
#   endif /*N < HPX_FUNCTION_ARGUMENT_LIMIT*/

    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(BOOST_PP_ENUM_PARAMS(N, Arg))>::type
    >::type
    invoke(F && f, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        typedef
            typename invoke_result_of<F(BOOST_PP_ENUM_PARAMS(N, Arg))>::type
            result_type;

        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , HPX_ENUM_FORWARD_ARGS(N, Arg, arg)
            );
    }
}}

#undef N

#endif
