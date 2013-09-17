// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        template <
            typename Functor
          , typename R
          
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R()
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R()
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R()
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f )
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) );
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16);
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f , A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f)) , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17);
            }
        };
