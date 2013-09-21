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
                , typename util::detail::add_rvalue_reference<A0>::type a0)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8 , typename util::detail::add_rvalue_reference<A9>::type a9)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8 , typename util::detail::add_rvalue_reference<A9>::type a9 , typename util::detail::add_rvalue_reference<A10>::type a10)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8 , typename util::detail::add_rvalue_reference<A9>::type a9 , typename util::detail::add_rvalue_reference<A10>::type a10 , typename util::detail::add_rvalue_reference<A11>::type a11)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8 , typename util::detail::add_rvalue_reference<A9>::type a9 , typename util::detail::add_rvalue_reference<A10>::type a10 , typename util::detail::add_rvalue_reference<A11>::type a11 , typename util::detail::add_rvalue_reference<A12>::type a12)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8 , typename util::detail::add_rvalue_reference<A9>::type a9 , typename util::detail::add_rvalue_reference<A10>::type a10 , typename util::detail::add_rvalue_reference<A11>::type a11 , typename util::detail::add_rvalue_reference<A12>::type a12 , typename util::detail::add_rvalue_reference<A13>::type a13)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8 , typename util::detail::add_rvalue_reference<A9>::type a9 , typename util::detail::add_rvalue_reference<A10>::type a10 , typename util::detail::add_rvalue_reference<A11>::type a11 , typename util::detail::add_rvalue_reference<A12>::type a12 , typename util::detail::add_rvalue_reference<A13>::type a13 , typename util::detail::add_rvalue_reference<A14>::type a14)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8 , typename util::detail::add_rvalue_reference<A9>::type a9 , typename util::detail::add_rvalue_reference<A10>::type a10 , typename util::detail::add_rvalue_reference<A11>::type a11 , typename util::detail::add_rvalue_reference<A12>::type a12 , typename util::detail::add_rvalue_reference<A13>::type a13 , typename util::detail::add_rvalue_reference<A14>::type a14 , typename util::detail::add_rvalue_reference<A15>::type a15)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8 , typename util::detail::add_rvalue_reference<A9>::type a9 , typename util::detail::add_rvalue_reference<A10>::type a10 , typename util::detail::add_rvalue_reference<A11>::type a11 , typename util::detail::add_rvalue_reference<A12>::type a12 , typename util::detail::add_rvalue_reference<A13>::type a13 , typename util::detail::add_rvalue_reference<A14>::type a14 , typename util::detail::add_rvalue_reference<A15>::type a15 , typename util::detail::add_rvalue_reference<A16>::type a16)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ));
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
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8 , typename util::detail::add_rvalue_reference<A9>::type a9 , typename util::detail::add_rvalue_reference<A10>::type a10 , typename util::detail::add_rvalue_reference<A11>::type a11 , typename util::detail::add_rvalue_reference<A12>::type a12 , typename util::detail::add_rvalue_reference<A13>::type a13 , typename util::detail::add_rvalue_reference<A14>::type a14 , typename util::detail::add_rvalue_reference<A15>::type a15 , typename util::detail::add_rvalue_reference<A16>::type a16 , typename util::detail::add_rvalue_reference<A17>::type a17)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ));
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8 , typename util::detail::add_rvalue_reference<A9>::type a9 , typename util::detail::add_rvalue_reference<A10>::type a10 , typename util::detail::add_rvalue_reference<A11>::type a11 , typename util::detail::add_rvalue_reference<A12>::type a12 , typename util::detail::add_rvalue_reference<A13>::type a13 , typename util::detail::add_rvalue_reference<A14>::type a14 , typename util::detail::add_rvalue_reference<A15>::type a15 , typename util::detail::add_rvalue_reference<A16>::type a16 , typename util::detail::add_rvalue_reference<A17>::type a17 , typename util::detail::add_rvalue_reference<A18>::type a18)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ));
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
          , typename IArchive
          , typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)
          , IArchive
          , OArchive
        >
            : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)
              , IArchive
              , OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)
                    >::template get<IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f
                , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7 , typename util::detail::add_rvalue_reference<A8>::type a8 , typename util::detail::add_rvalue_reference<A9>::type a9 , typename util::detail::add_rvalue_reference<A10>::type a10 , typename util::detail::add_rvalue_reference<A11>::type a11 , typename util::detail::add_rvalue_reference<A12>::type a12 , typename util::detail::add_rvalue_reference<A13>::type a13 , typename util::detail::add_rvalue_reference<A14>::type a14 , typename util::detail::add_rvalue_reference<A15>::type a15 , typename util::detail::add_rvalue_reference<A16>::type a16 , typename util::detail::add_rvalue_reference<A17>::type a17 , typename util::detail::add_rvalue_reference<A18>::type a18 , typename util::detail::add_rvalue_reference<A19>::type a19)
            {
                return util::invoke_r<R>((**reinterpret_cast<Functor**>(f))
                    , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ));
            }
        };
