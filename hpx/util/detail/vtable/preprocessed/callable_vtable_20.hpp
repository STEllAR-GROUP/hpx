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
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 && , A13 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 && , A13 && , A14 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 && , A13 && , A14 && , A15 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 && , A13 && , A14 && , A15 && , A16 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 && , A13 && , A14 && , A15 && , A16 && , A17 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 && , A13 && , A14 && , A15 && , A16 && , A17 && , A18 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 && , A13 && , A14 && , A15 && , A16 && , A17 && , A18 && , A19 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19 , A20 && a20)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 ) , std::forward<A20>( a20 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 && , A13 && , A14 && , A15 && , A16 && , A17 && , A18 && , A19 && , A20 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19 , A20 && a20 , A21 && a21)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 ) , std::forward<A20>( a20 ) , std::forward<A21>( a21 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 && , A13 && , A14 && , A15 && , A16 && , A17 && , A18 && , A19 && , A20 && , A21 &&);
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22>
    struct callable_vtable<R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21 , A22)>
    {
        template <typename T>
        BOOST_FORCEINLINE static R invoke(void** f,
            A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19 , A20 && a20 , A21 && a21 , A22 && a22)
        {
            return util::invoke_r<R>(vtable::get<T>(f),
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 ) , std::forward<A20>( a20 ) , std::forward<A21>( a21 ) , std::forward<A22>( a22 ));
        }
        typedef R (*invoke_t)(void**,
            A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 && , A13 && , A14 && , A15 && , A16 && , A17 && , A18 && , A19 && , A20 && , A21 && , A22 &&);
    };
}}}
