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
            invoke(void ** f
                )
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                     );
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type , typename util::detail::add_rvalue_reference<A7>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type , typename util::detail::add_rvalue_reference<A7>::type , typename util::detail::add_rvalue_reference<A8>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type , typename util::detail::add_rvalue_reference<A7>::type , typename util::detail::add_rvalue_reference<A8>::type , typename util::detail::add_rvalue_reference<A9>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type , typename util::detail::add_rvalue_reference<A7>::type , typename util::detail::add_rvalue_reference<A8>::type , typename util::detail::add_rvalue_reference<A9>::type , typename util::detail::add_rvalue_reference<A10>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type , typename util::detail::add_rvalue_reference<A7>::type , typename util::detail::add_rvalue_reference<A8>::type , typename util::detail::add_rvalue_reference<A9>::type , typename util::detail::add_rvalue_reference<A10>::type , typename util::detail::add_rvalue_reference<A11>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type , typename util::detail::add_rvalue_reference<A7>::type , typename util::detail::add_rvalue_reference<A8>::type , typename util::detail::add_rvalue_reference<A9>::type , typename util::detail::add_rvalue_reference<A10>::type , typename util::detail::add_rvalue_reference<A11>::type , typename util::detail::add_rvalue_reference<A12>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type , typename util::detail::add_rvalue_reference<A7>::type , typename util::detail::add_rvalue_reference<A8>::type , typename util::detail::add_rvalue_reference<A9>::type , typename util::detail::add_rvalue_reference<A10>::type , typename util::detail::add_rvalue_reference<A11>::type , typename util::detail::add_rvalue_reference<A12>::type , typename util::detail::add_rvalue_reference<A13>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type , typename util::detail::add_rvalue_reference<A7>::type , typename util::detail::add_rvalue_reference<A8>::type , typename util::detail::add_rvalue_reference<A9>::type , typename util::detail::add_rvalue_reference<A10>::type , typename util::detail::add_rvalue_reference<A11>::type , typename util::detail::add_rvalue_reference<A12>::type , typename util::detail::add_rvalue_reference<A13>::type , typename util::detail::add_rvalue_reference<A14>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type , typename util::detail::add_rvalue_reference<A7>::type , typename util::detail::add_rvalue_reference<A8>::type , typename util::detail::add_rvalue_reference<A9>::type , typename util::detail::add_rvalue_reference<A10>::type , typename util::detail::add_rvalue_reference<A11>::type , typename util::detail::add_rvalue_reference<A12>::type , typename util::detail::add_rvalue_reference<A13>::type , typename util::detail::add_rvalue_reference<A14>::type , typename util::detail::add_rvalue_reference<A15>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type , typename util::detail::add_rvalue_reference<A7>::type , typename util::detail::add_rvalue_reference<A8>::type , typename util::detail::add_rvalue_reference<A9>::type , typename util::detail::add_rvalue_reference<A10>::type , typename util::detail::add_rvalue_reference<A11>::type , typename util::detail::add_rvalue_reference<A12>::type , typename util::detail::add_rvalue_reference<A13>::type , typename util::detail::add_rvalue_reference<A14>::type , typename util::detail::add_rvalue_reference<A15>::type , typename util::detail::add_rvalue_reference<A16>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16);
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
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type , typename util::detail::add_rvalue_reference<A1>::type , typename util::detail::add_rvalue_reference<A2>::type , typename util::detail::add_rvalue_reference<A3>::type , typename util::detail::add_rvalue_reference<A4>::type , typename util::detail::add_rvalue_reference<A5>::type , typename util::detail::add_rvalue_reference<A6>::type , typename util::detail::add_rvalue_reference<A7>::type , typename util::detail::add_rvalue_reference<A8>::type , typename util::detail::add_rvalue_reference<A9>::type , typename util::detail::add_rvalue_reference<A10>::type , typename util::detail::add_rvalue_reference<A11>::type , typename util::detail::add_rvalue_reference<A12>::type , typename util::detail::add_rvalue_reference<A13>::type , typename util::detail::add_rvalue_reference<A14>::type , typename util::detail::add_rvalue_reference<A15>::type , typename util::detail::add_rvalue_reference<A16>::type , typename util::detail::add_rvalue_reference<A17>::type)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17);
            }
        };
