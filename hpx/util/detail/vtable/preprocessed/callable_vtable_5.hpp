// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0>
    struct callable_vtable<R(A0)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ));
        }
        typedef R (*invoke_t)(void**,
            A0 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1>
    struct callable_vtable<R(A0 , A1)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2>
    struct callable_vtable<R(A0 , A1 , A2)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3>
    struct callable_vtable<R(A0 , A1 , A2 , A3)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 &&);
    };
}}}
