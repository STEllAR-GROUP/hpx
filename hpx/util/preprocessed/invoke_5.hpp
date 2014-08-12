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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0)
      , T && t, Arg0 && arg0)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ));
    }
    template <typename R, typename FR
      , typename C, typename A0
      , typename T, typename Arg0>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0)
      , T && t, Arg0 && arg0)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ));
    }
    template <typename R, typename FR
      , typename C, typename A0
      , typename T, typename Arg0>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0) const
      , T && t, Arg0 && arg0)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ));
    }
    template <typename R, typename FR
      , typename C, typename A0
      , typename T, typename Arg0>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0) const
      , T && t, Arg0 && arg0)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ));
    }
    template <typename R, typename F, typename Arg0>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<F>::type>::value
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ));
    }
    template <typename R, typename F, typename Arg0>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
     || boost::is_reference_wrapper<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1)
      , T && t, Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1)
      , T && t, Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1) const
      , T && t, Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1
      , typename T, typename Arg0 , typename Arg1>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1) const
      , T && t, Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<F>::type>::value
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
     || boost::is_reference_wrapper<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2
      , typename T, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<F>::type>::value
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
     || boost::is_reference_wrapper<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<F>::type>::value
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
     || boost::is_reference_wrapper<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<F>::type>::value
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
     || boost::is_reference_wrapper<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<F>::type>::value
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
     || boost::is_reference_wrapper<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::void_guard<R>(), ((*std::forward<T>(t)).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename R, typename FR
      , typename C, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename T, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6) const
      , T && t, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<F>::type>::value
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
     || boost::is_reference_wrapper<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
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
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
    R
    invoke_r(FR (*f)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return
            util::void_guard<R>(), f
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<F>::type>::value
      , R
    >::type
    invoke_r(F && f, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return
            util::void_guard<R>(), (f.get())
                (std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename R, typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
     || boost::is_reference_wrapper<typename util::decay<F>::type>::value
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
    HPX_MAYBE_FORCEINLINE
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
    template <typename F, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
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
