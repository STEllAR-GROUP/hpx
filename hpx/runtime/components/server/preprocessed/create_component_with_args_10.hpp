// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        template <typename Component, typename A0>
        struct component_constructor_functor1
        {
            typedef void result_type;
            component_constructor_functor1(
                component_constructor_functor1 const & other)
              : a0(other. a0)
            {}
            component_constructor_functor1(
                component_constructor_functor1 && other)
              : a0(std::move(other. a0))
            {}
            template <typename T0>
            explicit
            component_constructor_functor1(
                T0 && t0
              , typename ::boost::disable_if<
                    typename boost::is_same<
                        component_constructor_functor1
                      , typename util::decay<T0>::type
                    >::type
                >::type * dummy = 0
            )
              : a0(std::forward<T0> (t0))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(std::move(a0));
            }
            typename util::decay<A0>::type a0;
        };
        template <typename Component, typename A0>
        naming::gid_type create_with_args(A0 && a0)
        {
            return server::create<Component>(
                component_constructor_functor1<
                    Component, A0>(
                        std::forward<A0>( a0 ))
            );
        }
        template <typename Component, typename A0 , typename A1>
        struct component_constructor_functor2
        {
            typedef void result_type;
            component_constructor_functor2(
                component_constructor_functor2 const & other)
              : a0(other. a0) , a1(other. a1)
            {}
            component_constructor_functor2(
                component_constructor_functor2 && other)
              : a0(std::move(other. a0)) , a1(std::move(other. a1))
            {}
            template <typename T0 , typename T1>
            component_constructor_functor2(
                T0 && t0 , T1 && t1
            )
              : a0(std::forward<T0> (t0)) , a1(std::forward<T1> (t1))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(std::move(a0) , std::move(a1));
            }
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1;
        };
        template <typename Component, typename A0 , typename A1>
        naming::gid_type create_with_args(A0 && a0 , A1 && a1)
        {
            return server::create<Component>(
                component_constructor_functor2<
                    Component, A0 , A1>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ))
            );
        }
        template <typename Component, typename A0 , typename A1 , typename A2>
        struct component_constructor_functor3
        {
            typedef void result_type;
            component_constructor_functor3(
                component_constructor_functor3 const & other)
              : a0(other. a0) , a1(other. a1) , a2(other. a2)
            {}
            component_constructor_functor3(
                component_constructor_functor3 && other)
              : a0(std::move(other. a0)) , a1(std::move(other. a1)) , a2(std::move(other. a2))
            {}
            template <typename T0 , typename T1 , typename T2>
            component_constructor_functor3(
                T0 && t0 , T1 && t1 , T2 && t2
            )
              : a0(std::forward<T0> (t0)) , a1(std::forward<T1> (t1)) , a2(std::forward<T2> (t2))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(std::move(a0) , std::move(a1) , std::move(a2));
            }
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1; typename util::decay<A2>::type a2;
        };
        template <typename Component, typename A0 , typename A1 , typename A2>
        naming::gid_type create_with_args(A0 && a0 , A1 && a1 , A2 && a2)
        {
            return server::create<Component>(
                component_constructor_functor3<
                    Component, A0 , A1 , A2>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ))
            );
        }
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3>
        struct component_constructor_functor4
        {
            typedef void result_type;
            component_constructor_functor4(
                component_constructor_functor4 const & other)
              : a0(other. a0) , a1(other. a1) , a2(other. a2) , a3(other. a3)
            {}
            component_constructor_functor4(
                component_constructor_functor4 && other)
              : a0(std::move(other. a0)) , a1(std::move(other. a1)) , a2(std::move(other. a2)) , a3(std::move(other. a3))
            {}
            template <typename T0 , typename T1 , typename T2 , typename T3>
            component_constructor_functor4(
                T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3
            )
              : a0(std::forward<T0> (t0)) , a1(std::forward<T1> (t1)) , a2(std::forward<T2> (t2)) , a3(std::forward<T3> (t3))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(std::move(a0) , std::move(a1) , std::move(a2) , std::move(a3));
            }
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1; typename util::decay<A2>::type a2; typename util::decay<A3>::type a3;
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3>
        naming::gid_type create_with_args(A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3)
        {
            return server::create<Component>(
                component_constructor_functor4<
                    Component, A0 , A1 , A2 , A3>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ))
            );
        }
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        struct component_constructor_functor5
        {
            typedef void result_type;
            component_constructor_functor5(
                component_constructor_functor5 const & other)
              : a0(other. a0) , a1(other. a1) , a2(other. a2) , a3(other. a3) , a4(other. a4)
            {}
            component_constructor_functor5(
                component_constructor_functor5 && other)
              : a0(std::move(other. a0)) , a1(std::move(other. a1)) , a2(std::move(other. a2)) , a3(std::move(other. a3)) , a4(std::move(other. a4))
            {}
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
            component_constructor_functor5(
                T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4
            )
              : a0(std::forward<T0> (t0)) , a1(std::forward<T1> (t1)) , a2(std::forward<T2> (t2)) , a3(std::forward<T3> (t3)) , a4(std::forward<T4> (t4))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(std::move(a0) , std::move(a1) , std::move(a2) , std::move(a3) , std::move(a4));
            }
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1; typename util::decay<A2>::type a2; typename util::decay<A3>::type a3; typename util::decay<A4>::type a4;
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        naming::gid_type create_with_args(A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4)
        {
            return server::create<Component>(
                component_constructor_functor5<
                    Component, A0 , A1 , A2 , A3 , A4>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ))
            );
        }
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        struct component_constructor_functor6
        {
            typedef void result_type;
            component_constructor_functor6(
                component_constructor_functor6 const & other)
              : a0(other. a0) , a1(other. a1) , a2(other. a2) , a3(other. a3) , a4(other. a4) , a5(other. a5)
            {}
            component_constructor_functor6(
                component_constructor_functor6 && other)
              : a0(std::move(other. a0)) , a1(std::move(other. a1)) , a2(std::move(other. a2)) , a3(std::move(other. a3)) , a4(std::move(other. a4)) , a5(std::move(other. a5))
            {}
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
            component_constructor_functor6(
                T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5
            )
              : a0(std::forward<T0> (t0)) , a1(std::forward<T1> (t1)) , a2(std::forward<T2> (t2)) , a3(std::forward<T3> (t3)) , a4(std::forward<T4> (t4)) , a5(std::forward<T5> (t5))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(std::move(a0) , std::move(a1) , std::move(a2) , std::move(a3) , std::move(a4) , std::move(a5));
            }
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1; typename util::decay<A2>::type a2; typename util::decay<A3>::type a3; typename util::decay<A4>::type a4; typename util::decay<A5>::type a5;
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        naming::gid_type create_with_args(A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5)
        {
            return server::create<Component>(
                component_constructor_functor6<
                    Component, A0 , A1 , A2 , A3 , A4 , A5>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ))
            );
        }
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        struct component_constructor_functor7
        {
            typedef void result_type;
            component_constructor_functor7(
                component_constructor_functor7 const & other)
              : a0(other. a0) , a1(other. a1) , a2(other. a2) , a3(other. a3) , a4(other. a4) , a5(other. a5) , a6(other. a6)
            {}
            component_constructor_functor7(
                component_constructor_functor7 && other)
              : a0(std::move(other. a0)) , a1(std::move(other. a1)) , a2(std::move(other. a2)) , a3(std::move(other. a3)) , a4(std::move(other. a4)) , a5(std::move(other. a5)) , a6(std::move(other. a6))
            {}
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
            component_constructor_functor7(
                T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6
            )
              : a0(std::forward<T0> (t0)) , a1(std::forward<T1> (t1)) , a2(std::forward<T2> (t2)) , a3(std::forward<T3> (t3)) , a4(std::forward<T4> (t4)) , a5(std::forward<T5> (t5)) , a6(std::forward<T6> (t6))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(std::move(a0) , std::move(a1) , std::move(a2) , std::move(a3) , std::move(a4) , std::move(a5) , std::move(a6));
            }
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1; typename util::decay<A2>::type a2; typename util::decay<A3>::type a3; typename util::decay<A4>::type a4; typename util::decay<A5>::type a5; typename util::decay<A6>::type a6;
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        naming::gid_type create_with_args(A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6)
        {
            return server::create<Component>(
                component_constructor_functor7<
                    Component, A0 , A1 , A2 , A3 , A4 , A5 , A6>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ))
            );
        }
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        struct component_constructor_functor8
        {
            typedef void result_type;
            component_constructor_functor8(
                component_constructor_functor8 const & other)
              : a0(other. a0) , a1(other. a1) , a2(other. a2) , a3(other. a3) , a4(other. a4) , a5(other. a5) , a6(other. a6) , a7(other. a7)
            {}
            component_constructor_functor8(
                component_constructor_functor8 && other)
              : a0(std::move(other. a0)) , a1(std::move(other. a1)) , a2(std::move(other. a2)) , a3(std::move(other. a3)) , a4(std::move(other. a4)) , a5(std::move(other. a5)) , a6(std::move(other. a6)) , a7(std::move(other. a7))
            {}
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
            component_constructor_functor8(
                T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7
            )
              : a0(std::forward<T0> (t0)) , a1(std::forward<T1> (t1)) , a2(std::forward<T2> (t2)) , a3(std::forward<T3> (t3)) , a4(std::forward<T4> (t4)) , a5(std::forward<T5> (t5)) , a6(std::forward<T6> (t6)) , a7(std::forward<T7> (t7))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(std::move(a0) , std::move(a1) , std::move(a2) , std::move(a3) , std::move(a4) , std::move(a5) , std::move(a6) , std::move(a7));
            }
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1; typename util::decay<A2>::type a2; typename util::decay<A3>::type a3; typename util::decay<A4>::type a4; typename util::decay<A5>::type a5; typename util::decay<A6>::type a6; typename util::decay<A7>::type a7;
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        naming::gid_type create_with_args(A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7)
        {
            return server::create<Component>(
                component_constructor_functor8<
                    Component, A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ))
            );
        }
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        struct component_constructor_functor9
        {
            typedef void result_type;
            component_constructor_functor9(
                component_constructor_functor9 const & other)
              : a0(other. a0) , a1(other. a1) , a2(other. a2) , a3(other. a3) , a4(other. a4) , a5(other. a5) , a6(other. a6) , a7(other. a7) , a8(other. a8)
            {}
            component_constructor_functor9(
                component_constructor_functor9 && other)
              : a0(std::move(other. a0)) , a1(std::move(other. a1)) , a2(std::move(other. a2)) , a3(std::move(other. a3)) , a4(std::move(other. a4)) , a5(std::move(other. a5)) , a6(std::move(other. a6)) , a7(std::move(other. a7)) , a8(std::move(other. a8))
            {}
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
            component_constructor_functor9(
                T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8
            )
              : a0(std::forward<T0> (t0)) , a1(std::forward<T1> (t1)) , a2(std::forward<T2> (t2)) , a3(std::forward<T3> (t3)) , a4(std::forward<T4> (t4)) , a5(std::forward<T5> (t5)) , a6(std::forward<T6> (t6)) , a7(std::forward<T7> (t7)) , a8(std::forward<T8> (t8))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(std::move(a0) , std::move(a1) , std::move(a2) , std::move(a3) , std::move(a4) , std::move(a5) , std::move(a6) , std::move(a7) , std::move(a8));
            }
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1; typename util::decay<A2>::type a2; typename util::decay<A3>::type a3; typename util::decay<A4>::type a4; typename util::decay<A5>::type a5; typename util::decay<A6>::type a6; typename util::decay<A7>::type a7; typename util::decay<A8>::type a8;
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        naming::gid_type create_with_args(A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8)
        {
            return server::create<Component>(
                component_constructor_functor9<
                    Component, A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ))
            );
        }
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        struct component_constructor_functor10
        {
            typedef void result_type;
            component_constructor_functor10(
                component_constructor_functor10 const & other)
              : a0(other. a0) , a1(other. a1) , a2(other. a2) , a3(other. a3) , a4(other. a4) , a5(other. a5) , a6(other. a6) , a7(other. a7) , a8(other. a8) , a9(other. a9)
            {}
            component_constructor_functor10(
                component_constructor_functor10 && other)
              : a0(std::move(other. a0)) , a1(std::move(other. a1)) , a2(std::move(other. a2)) , a3(std::move(other. a3)) , a4(std::move(other. a4)) , a5(std::move(other. a5)) , a6(std::move(other. a6)) , a7(std::move(other. a7)) , a8(std::move(other. a8)) , a9(std::move(other. a9))
            {}
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
            component_constructor_functor10(
                T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9
            )
              : a0(std::forward<T0> (t0)) , a1(std::forward<T1> (t1)) , a2(std::forward<T2> (t2)) , a3(std::forward<T3> (t3)) , a4(std::forward<T4> (t4)) , a5(std::forward<T5> (t5)) , a6(std::forward<T6> (t6)) , a7(std::forward<T7> (t7)) , a8(std::forward<T8> (t8)) , a9(std::forward<T9> (t9))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(std::move(a0) , std::move(a1) , std::move(a2) , std::move(a3) , std::move(a4) , std::move(a5) , std::move(a6) , std::move(a7) , std::move(a8) , std::move(a9));
            }
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1; typename util::decay<A2>::type a2; typename util::decay<A3>::type a3; typename util::decay<A4>::type a4; typename util::decay<A5>::type a5; typename util::decay<A6>::type a6; typename util::decay<A7>::type a7; typename util::decay<A8>::type a8; typename util::decay<A9>::type a9;
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        naming::gid_type create_with_args(A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9)
        {
            return server::create<Component>(
                component_constructor_functor10<
                    Component, A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ))
            );
        }
