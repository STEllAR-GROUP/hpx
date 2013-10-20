// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0
      , typename Arg0>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0)
      , BOOST_FWD_REF(Arg0) arg0)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ));
    }
    template <typename R, typename FR
      , typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ));
    }
    template <typename R, typename FR
      , typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ));
    }
    template <typename R, typename FR
      , typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ));
    }
    template <typename R, typename FR
      , typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ));
    }
    template <typename R, typename F, typename Arg0>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ));
    }
    template <typename R, typename F, typename Arg0>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ));
    }
    
    template <typename R, typename A0
      , typename Arg0>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0)
      , BOOST_FWD_REF(Arg0) arg0)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 )
            );
    }
    template <typename R, typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 )
            );
    }
    template <typename R, typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 )
            );
    }
    template <typename F, typename Arg0>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0)
    {
        typedef
            typename invoke_result_of<F(Arg0)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1
      , typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    
    template <typename R, typename A0 , typename A1
      , typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2
      , typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2
      , typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
    {
        return
            util::void_guard<R>(), (boost::forward<T>(t).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(boost::forward<T>(t))).*f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    HPX_UTIL_INVOKE_MEM_FUN_PTR_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16) const
      , BOOST_FWD_REF(T) t, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
    {
        return
            util::invoke_r<R>(
                f, boost::forward<T>(t)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17)
    {
        return
            util::void_guard<R>(), f
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ) , boost::forward<Arg17>( arg17 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17)
    {
        return
            util::void_guard<R>(), (f.get())
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ) , boost::forward<Arg17>( arg17 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17)
    {
        return
            util::void_guard<R>(), boost::forward<F>(f)
                (boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ) , boost::forward<Arg17>( arg17 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
      , BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17)
    {
        return
            util::invoke_r<R>(
                f
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ) , boost::forward<Arg17>( arg17 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    BOOST_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17)>::type
    >::type
    invoke(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7 , BOOST_FWD_REF(Arg8) arg8 , BOOST_FWD_REF(Arg9) arg9 , BOOST_FWD_REF(Arg10) arg10 , BOOST_FWD_REF(Arg11) arg11 , BOOST_FWD_REF(Arg12) arg12 , BOOST_FWD_REF(Arg13) arg13 , BOOST_FWD_REF(Arg14) arg14 , BOOST_FWD_REF(Arg15) arg15 , BOOST_FWD_REF(Arg16) arg16 , BOOST_FWD_REF(Arg17) arg17)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12 , Arg13 , Arg14 , Arg15 , Arg16 , Arg17)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                boost::forward<F>(f)
              , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ) , boost::forward<Arg8>( arg8 ) , boost::forward<Arg9>( arg9 ) , boost::forward<Arg10>( arg10 ) , boost::forward<Arg11>( arg11 ) , boost::forward<Arg12>( arg12 ) , boost::forward<Arg13>( arg13 ) , boost::forward<Arg14>( arg14 ) , boost::forward<Arg15>( arg15 ) , boost::forward<Arg16>( arg16 ) , boost::forward<Arg17>( arg17 )
            );
    }
}}
