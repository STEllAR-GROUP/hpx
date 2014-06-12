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
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 &&);
    };
}}}
