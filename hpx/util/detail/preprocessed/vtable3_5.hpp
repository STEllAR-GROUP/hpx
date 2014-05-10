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
          
          , typename IArchive, typename OArchive
        >
        struct type<
            Functor
          , R()
          , IArchive, OArchive
        > : type_base<Functor>
        {
            static vtable_ptr_base<
                R()
              , IArchive, OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R()
                    >::template get<true, IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f
                )
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f))
                     );
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0
          , typename IArchive, typename OArchive
        >
        struct type<
            Functor
          , R(A0)
          , IArchive, OArchive
        > : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0)
              , IArchive, OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0)
                    >::template get<true, IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f
                , A0 && a0)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f))
                    , std::forward<A0>( a0 ));
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1
          , typename IArchive, typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1)
          , IArchive, OArchive
        > : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1)
              , IArchive, OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1)
                    >::template get<true, IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f
                , A0 && a0 , A1 && a1)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f))
                    , std::forward<A0>( a0 ) , std::forward<A1>( a1 ));
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2
          , typename IArchive, typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2)
          , IArchive, OArchive
        > : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2)
              , IArchive, OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2)
                    >::template get<true, IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f
                , A0 && a0 , A1 && a1 , A2 && a2)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f))
                    , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ));
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3
          , typename IArchive, typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3)
          , IArchive, OArchive
        > : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3)
              , IArchive, OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3)
                    >::template get<true, IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f
                , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f))
                    , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ));
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
          , typename IArchive, typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4)
          , IArchive, OArchive
        > : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4)
              , IArchive, OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4)
                    >::template get<true, IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f
                , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f))
                    , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ));
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
          , typename IArchive, typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5)
          , IArchive, OArchive
        > : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5)
              , IArchive, OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5)
                    >::template get<true, IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f
                , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f))
                    , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ));
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
          , typename IArchive, typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , IArchive, OArchive
        > : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
              , IArchive, OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
                    >::template get<true, IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f
                , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f))
                    , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ));
            }
        };
        template <
            typename Functor
          , typename R
          , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
          , typename IArchive, typename OArchive
        >
        struct type<
            Functor
          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , IArchive, OArchive
        > : type_base<Functor>
        {
            static vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
              , IArchive, OArchive
            > *get_ptr()
            {
                return
                    get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
                    >::template get<true, IArchive, OArchive>();
            }
            BOOST_FORCEINLINE static R
            invoke(void ** f
                , A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7)
            {
                return util::invoke_r<R>((*reinterpret_cast<Functor*>(f))
                    , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ));
            }
        };
