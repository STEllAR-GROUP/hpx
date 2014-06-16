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
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0)
      , Arg0 && arg0)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ));
    }
    template <typename R, typename FR
      , typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0)
      , T && t, Arg0 && arg0)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ));
    }
    template <typename R, typename FR
      , typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0)
      , T && t, Arg0 && arg0)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ));
    }
    template <typename R, typename FR
      , typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0) const
      , T && t, Arg0 && arg0)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ));
    }
    template <typename R, typename FR
      , typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0) const
      , T && t, Arg0 && arg0)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ));
    }
    template <typename R, typename F, typename Arg0>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ));
    }
    template <typename R, typename F, typename Arg0>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ));
    }
    
    template <typename R, typename A0
      , typename Arg0>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0)
      , Arg0 && arg0)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 )
            );
    }
    template <typename R, typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0)
      , T && t, Arg0 && arg0)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 )
            );
    }
    template <typename R, typename C, typename A0
      , typename T, typename Arg0>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0) const
      , T && t, Arg0 && arg0)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 )
            );
    }
    template <typename F, typename Arg0>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0)>::type
    >::type
    invoke(F && f, Arg0 && arg0)
    {
        typedef
            typename invoke_result_of<F(Arg0)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1
      , typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1)
      , Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1)
      , T && t, Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1)
      , T && t, Arg0 && arg0 , Arg1 && arg1)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1) const
      , T && t, Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1) const
      , T && t, Arg0 && arg0 , Arg1 && arg1)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    
    template <typename R, typename A0 , typename A1
      , typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1)
      , Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1)
      , T && t, Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1) const
      , T && t, Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2
      , typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2
      , typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11)
    {
        return
            util::void_guard<R>(), (std::forward<T>(t).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::is_base_of<C, typename util::decay<T>::type>
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11)
    {
        using boost::get_pointer;
        return
            util::void_guard<R>(), ((*get_pointer(std::forward<T>(t))).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 )
            );
    }
    template <typename R, typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11)
    {
        return
            util::invoke_r<R>(
                f, std::forward<T>(t)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 )
            );
    }
}}
namespace hpx { namespace util
{
    
    template <typename R, typename FR, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::enable_if<
        boost::is_reference_wrapper<typename util::decay<F>::type>
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
          , boost::is_reference_wrapper<typename util::decay<F>::type>
        >
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12)
    {
        return
            util::void_guard<R>(), std::forward<F>(f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ));
    }
    
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    HPX_UTIL_INVOKE_FORCEINLINE
    R
    invoke(R (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12)
    {
        return
            util::invoke_r<R>(
                f
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 )
            );
    }
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    HPX_UTIL_INVOKE_FORCEINLINE
    typename boost::disable_if<
        boost::mpl::or_<
            boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>
          , boost::is_member_pointer<typename util::decay<F>::type>
        >
      , typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12)>::type
    >::type
    invoke(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12)
    {
        typedef
            typename invoke_result_of<F(Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7 , Arg8 , Arg9 , Arg10 , Arg11 , Arg12)>::type
            result_type;
        return
            util::invoke_r<result_type>(
                std::forward<F>(f)
              , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 )
            );
    }
}}
