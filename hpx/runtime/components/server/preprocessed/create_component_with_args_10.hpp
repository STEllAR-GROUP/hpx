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
                BOOST_RV_REF(component_constructor_functor1) other)
              : a0(boost::move(other. a0))
            {}
            component_constructor_functor1 & operator=(
                BOOST_COPY_ASSIGN_REF(component_constructor_functor1) other)
            {
                a0 = other. a0;
                return *this;
            }
            component_constructor_functor1 & operator=(
                BOOST_RV_REF(component_constructor_functor1) other)
            {
                a0 = boost::move(other. a0);
                return *this;
            }
            template <typename T0>
            explicit
            component_constructor_functor1(
                BOOST_FWD_REF(T0) t0
              , typename ::boost::disable_if<
                    typename boost::is_same<
                        component_constructor_functor1
                      , typename boost::remove_const<
                            typename hpx::util::detail::remove_reference<
                                T0
                            >::type
                        >::type
                    >::type
                >::type * dummy = 0
            )
              : a0(boost::forward<T0> (t0))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(boost::move(a0));
            }
            typename boost::remove_const<typename hpx::util::detail::remove_reference<A0>::type >::type a0;
            private:
                BOOST_COPYABLE_AND_MOVABLE(component_constructor_functor1)
        };
        template <typename Component, typename A0>
        naming::gid_type create_with_args(BOOST_FWD_REF(A0) a0)
        {
            return server::create<Component>(
                component_constructor_functor1<
                    Component, A0>(
                        boost::forward<A0>( a0 ))
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
                BOOST_RV_REF(component_constructor_functor2) other)
              : a0(boost::move(other. a0)) , a1(boost::move(other. a1))
            {}
            component_constructor_functor2 & operator=(
                BOOST_COPY_ASSIGN_REF(component_constructor_functor2) other)
            {
                a0 = other. a0; a1 = other. a1;
                return *this;
            }
            component_constructor_functor2 & operator=(
                BOOST_RV_REF(component_constructor_functor2) other)
            {
                a0 = boost::move(other. a0); a1 = boost::move(other. a1);
                return *this;
            }
            template <typename T0 , typename T1>
            component_constructor_functor2(
                BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1
            )
              : a0(boost::forward<T0> (t0)) , a1(boost::forward<T1> (t1))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(boost::move(a0) , boost::move(a1));
            }
            typename boost::remove_const<typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const<typename hpx::util::detail::remove_reference<A1>::type >::type a1;
            private:
                BOOST_COPYABLE_AND_MOVABLE(component_constructor_functor2)
        };
        template <typename Component, typename A0 , typename A1>
        naming::gid_type create_with_args(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
        {
            return server::create<Component>(
                component_constructor_functor2<
                    Component, A0 , A1>(
                        boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ))
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
                BOOST_RV_REF(component_constructor_functor3) other)
              : a0(boost::move(other. a0)) , a1(boost::move(other. a1)) , a2(boost::move(other. a2))
            {}
            component_constructor_functor3 & operator=(
                BOOST_COPY_ASSIGN_REF(component_constructor_functor3) other)
            {
                a0 = other. a0; a1 = other. a1; a2 = other. a2;
                return *this;
            }
            component_constructor_functor3 & operator=(
                BOOST_RV_REF(component_constructor_functor3) other)
            {
                a0 = boost::move(other. a0); a1 = boost::move(other. a1); a2 = boost::move(other. a2);
                return *this;
            }
            template <typename T0 , typename T1 , typename T2>
            component_constructor_functor3(
                BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2
            )
              : a0(boost::forward<T0> (t0)) , a1(boost::forward<T1> (t1)) , a2(boost::forward<T2> (t2))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(boost::move(a0) , boost::move(a1) , boost::move(a2));
            }
            typename boost::remove_const<typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const<typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const<typename hpx::util::detail::remove_reference<A2>::type >::type a2;
            private:
                BOOST_COPYABLE_AND_MOVABLE(component_constructor_functor3)
        };
        template <typename Component, typename A0 , typename A1 , typename A2>
        naming::gid_type create_with_args(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
        {
            return server::create<Component>(
                component_constructor_functor3<
                    Component, A0 , A1 , A2>(
                        boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ))
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
                BOOST_RV_REF(component_constructor_functor4) other)
              : a0(boost::move(other. a0)) , a1(boost::move(other. a1)) , a2(boost::move(other. a2)) , a3(boost::move(other. a3))
            {}
            component_constructor_functor4 & operator=(
                BOOST_COPY_ASSIGN_REF(component_constructor_functor4) other)
            {
                a0 = other. a0; a1 = other. a1; a2 = other. a2; a3 = other. a3;
                return *this;
            }
            component_constructor_functor4 & operator=(
                BOOST_RV_REF(component_constructor_functor4) other)
            {
                a0 = boost::move(other. a0); a1 = boost::move(other. a1); a2 = boost::move(other. a2); a3 = boost::move(other. a3);
                return *this;
            }
            template <typename T0 , typename T1 , typename T2 , typename T3>
            component_constructor_functor4(
                BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3
            )
              : a0(boost::forward<T0> (t0)) , a1(boost::forward<T1> (t1)) , a2(boost::forward<T2> (t2)) , a3(boost::forward<T3> (t3))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(boost::move(a0) , boost::move(a1) , boost::move(a2) , boost::move(a3));
            }
            typename boost::remove_const<typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const<typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const<typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const<typename hpx::util::detail::remove_reference<A3>::type >::type a3;
            private:
                BOOST_COPYABLE_AND_MOVABLE(component_constructor_functor4)
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3>
        naming::gid_type create_with_args(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
        {
            return server::create<Component>(
                component_constructor_functor4<
                    Component, A0 , A1 , A2 , A3>(
                        boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ))
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
                BOOST_RV_REF(component_constructor_functor5) other)
              : a0(boost::move(other. a0)) , a1(boost::move(other. a1)) , a2(boost::move(other. a2)) , a3(boost::move(other. a3)) , a4(boost::move(other. a4))
            {}
            component_constructor_functor5 & operator=(
                BOOST_COPY_ASSIGN_REF(component_constructor_functor5) other)
            {
                a0 = other. a0; a1 = other. a1; a2 = other. a2; a3 = other. a3; a4 = other. a4;
                return *this;
            }
            component_constructor_functor5 & operator=(
                BOOST_RV_REF(component_constructor_functor5) other)
            {
                a0 = boost::move(other. a0); a1 = boost::move(other. a1); a2 = boost::move(other. a2); a3 = boost::move(other. a3); a4 = boost::move(other. a4);
                return *this;
            }
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
            component_constructor_functor5(
                BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4
            )
              : a0(boost::forward<T0> (t0)) , a1(boost::forward<T1> (t1)) , a2(boost::forward<T2> (t2)) , a3(boost::forward<T3> (t3)) , a4(boost::forward<T4> (t4))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(boost::move(a0) , boost::move(a1) , boost::move(a2) , boost::move(a3) , boost::move(a4));
            }
            typename boost::remove_const<typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const<typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const<typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const<typename hpx::util::detail::remove_reference<A3>::type >::type a3; typename boost::remove_const<typename hpx::util::detail::remove_reference<A4>::type >::type a4;
            private:
                BOOST_COPYABLE_AND_MOVABLE(component_constructor_functor5)
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
        naming::gid_type create_with_args(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
        {
            return server::create<Component>(
                component_constructor_functor5<
                    Component, A0 , A1 , A2 , A3 , A4>(
                        boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ))
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
                BOOST_RV_REF(component_constructor_functor6) other)
              : a0(boost::move(other. a0)) , a1(boost::move(other. a1)) , a2(boost::move(other. a2)) , a3(boost::move(other. a3)) , a4(boost::move(other. a4)) , a5(boost::move(other. a5))
            {}
            component_constructor_functor6 & operator=(
                BOOST_COPY_ASSIGN_REF(component_constructor_functor6) other)
            {
                a0 = other. a0; a1 = other. a1; a2 = other. a2; a3 = other. a3; a4 = other. a4; a5 = other. a5;
                return *this;
            }
            component_constructor_functor6 & operator=(
                BOOST_RV_REF(component_constructor_functor6) other)
            {
                a0 = boost::move(other. a0); a1 = boost::move(other. a1); a2 = boost::move(other. a2); a3 = boost::move(other. a3); a4 = boost::move(other. a4); a5 = boost::move(other. a5);
                return *this;
            }
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
            component_constructor_functor6(
                BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5
            )
              : a0(boost::forward<T0> (t0)) , a1(boost::forward<T1> (t1)) , a2(boost::forward<T2> (t2)) , a3(boost::forward<T3> (t3)) , a4(boost::forward<T4> (t4)) , a5(boost::forward<T5> (t5))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(boost::move(a0) , boost::move(a1) , boost::move(a2) , boost::move(a3) , boost::move(a4) , boost::move(a5));
            }
            typename boost::remove_const<typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const<typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const<typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const<typename hpx::util::detail::remove_reference<A3>::type >::type a3; typename boost::remove_const<typename hpx::util::detail::remove_reference<A4>::type >::type a4; typename boost::remove_const<typename hpx::util::detail::remove_reference<A5>::type >::type a5;
            private:
                BOOST_COPYABLE_AND_MOVABLE(component_constructor_functor6)
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
        naming::gid_type create_with_args(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
        {
            return server::create<Component>(
                component_constructor_functor6<
                    Component, A0 , A1 , A2 , A3 , A4 , A5>(
                        boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ))
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
                BOOST_RV_REF(component_constructor_functor7) other)
              : a0(boost::move(other. a0)) , a1(boost::move(other. a1)) , a2(boost::move(other. a2)) , a3(boost::move(other. a3)) , a4(boost::move(other. a4)) , a5(boost::move(other. a5)) , a6(boost::move(other. a6))
            {}
            component_constructor_functor7 & operator=(
                BOOST_COPY_ASSIGN_REF(component_constructor_functor7) other)
            {
                a0 = other. a0; a1 = other. a1; a2 = other. a2; a3 = other. a3; a4 = other. a4; a5 = other. a5; a6 = other. a6;
                return *this;
            }
            component_constructor_functor7 & operator=(
                BOOST_RV_REF(component_constructor_functor7) other)
            {
                a0 = boost::move(other. a0); a1 = boost::move(other. a1); a2 = boost::move(other. a2); a3 = boost::move(other. a3); a4 = boost::move(other. a4); a5 = boost::move(other. a5); a6 = boost::move(other. a6);
                return *this;
            }
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
            component_constructor_functor7(
                BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6
            )
              : a0(boost::forward<T0> (t0)) , a1(boost::forward<T1> (t1)) , a2(boost::forward<T2> (t2)) , a3(boost::forward<T3> (t3)) , a4(boost::forward<T4> (t4)) , a5(boost::forward<T5> (t5)) , a6(boost::forward<T6> (t6))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(boost::move(a0) , boost::move(a1) , boost::move(a2) , boost::move(a3) , boost::move(a4) , boost::move(a5) , boost::move(a6));
            }
            typename boost::remove_const<typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const<typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const<typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const<typename hpx::util::detail::remove_reference<A3>::type >::type a3; typename boost::remove_const<typename hpx::util::detail::remove_reference<A4>::type >::type a4; typename boost::remove_const<typename hpx::util::detail::remove_reference<A5>::type >::type a5; typename boost::remove_const<typename hpx::util::detail::remove_reference<A6>::type >::type a6;
            private:
                BOOST_COPYABLE_AND_MOVABLE(component_constructor_functor7)
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
        naming::gid_type create_with_args(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
        {
            return server::create<Component>(
                component_constructor_functor7<
                    Component, A0 , A1 , A2 , A3 , A4 , A5 , A6>(
                        boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ))
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
                BOOST_RV_REF(component_constructor_functor8) other)
              : a0(boost::move(other. a0)) , a1(boost::move(other. a1)) , a2(boost::move(other. a2)) , a3(boost::move(other. a3)) , a4(boost::move(other. a4)) , a5(boost::move(other. a5)) , a6(boost::move(other. a6)) , a7(boost::move(other. a7))
            {}
            component_constructor_functor8 & operator=(
                BOOST_COPY_ASSIGN_REF(component_constructor_functor8) other)
            {
                a0 = other. a0; a1 = other. a1; a2 = other. a2; a3 = other. a3; a4 = other. a4; a5 = other. a5; a6 = other. a6; a7 = other. a7;
                return *this;
            }
            component_constructor_functor8 & operator=(
                BOOST_RV_REF(component_constructor_functor8) other)
            {
                a0 = boost::move(other. a0); a1 = boost::move(other. a1); a2 = boost::move(other. a2); a3 = boost::move(other. a3); a4 = boost::move(other. a4); a5 = boost::move(other. a5); a6 = boost::move(other. a6); a7 = boost::move(other. a7);
                return *this;
            }
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
            component_constructor_functor8(
                BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7
            )
              : a0(boost::forward<T0> (t0)) , a1(boost::forward<T1> (t1)) , a2(boost::forward<T2> (t2)) , a3(boost::forward<T3> (t3)) , a4(boost::forward<T4> (t4)) , a5(boost::forward<T5> (t5)) , a6(boost::forward<T6> (t6)) , a7(boost::forward<T7> (t7))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(boost::move(a0) , boost::move(a1) , boost::move(a2) , boost::move(a3) , boost::move(a4) , boost::move(a5) , boost::move(a6) , boost::move(a7));
            }
            typename boost::remove_const<typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const<typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const<typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const<typename hpx::util::detail::remove_reference<A3>::type >::type a3; typename boost::remove_const<typename hpx::util::detail::remove_reference<A4>::type >::type a4; typename boost::remove_const<typename hpx::util::detail::remove_reference<A5>::type >::type a5; typename boost::remove_const<typename hpx::util::detail::remove_reference<A6>::type >::type a6; typename boost::remove_const<typename hpx::util::detail::remove_reference<A7>::type >::type a7;
            private:
                BOOST_COPYABLE_AND_MOVABLE(component_constructor_functor8)
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
        naming::gid_type create_with_args(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
        {
            return server::create<Component>(
                component_constructor_functor8<
                    Component, A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>(
                        boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ))
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
                BOOST_RV_REF(component_constructor_functor9) other)
              : a0(boost::move(other. a0)) , a1(boost::move(other. a1)) , a2(boost::move(other. a2)) , a3(boost::move(other. a3)) , a4(boost::move(other. a4)) , a5(boost::move(other. a5)) , a6(boost::move(other. a6)) , a7(boost::move(other. a7)) , a8(boost::move(other. a8))
            {}
            component_constructor_functor9 & operator=(
                BOOST_COPY_ASSIGN_REF(component_constructor_functor9) other)
            {
                a0 = other. a0; a1 = other. a1; a2 = other. a2; a3 = other. a3; a4 = other. a4; a5 = other. a5; a6 = other. a6; a7 = other. a7; a8 = other. a8;
                return *this;
            }
            component_constructor_functor9 & operator=(
                BOOST_RV_REF(component_constructor_functor9) other)
            {
                a0 = boost::move(other. a0); a1 = boost::move(other. a1); a2 = boost::move(other. a2); a3 = boost::move(other. a3); a4 = boost::move(other. a4); a5 = boost::move(other. a5); a6 = boost::move(other. a6); a7 = boost::move(other. a7); a8 = boost::move(other. a8);
                return *this;
            }
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
            component_constructor_functor9(
                BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8
            )
              : a0(boost::forward<T0> (t0)) , a1(boost::forward<T1> (t1)) , a2(boost::forward<T2> (t2)) , a3(boost::forward<T3> (t3)) , a4(boost::forward<T4> (t4)) , a5(boost::forward<T5> (t5)) , a6(boost::forward<T6> (t6)) , a7(boost::forward<T7> (t7)) , a8(boost::forward<T8> (t8))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(boost::move(a0) , boost::move(a1) , boost::move(a2) , boost::move(a3) , boost::move(a4) , boost::move(a5) , boost::move(a6) , boost::move(a7) , boost::move(a8));
            }
            typename boost::remove_const<typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const<typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const<typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const<typename hpx::util::detail::remove_reference<A3>::type >::type a3; typename boost::remove_const<typename hpx::util::detail::remove_reference<A4>::type >::type a4; typename boost::remove_const<typename hpx::util::detail::remove_reference<A5>::type >::type a5; typename boost::remove_const<typename hpx::util::detail::remove_reference<A6>::type >::type a6; typename boost::remove_const<typename hpx::util::detail::remove_reference<A7>::type >::type a7; typename boost::remove_const<typename hpx::util::detail::remove_reference<A8>::type >::type a8;
            private:
                BOOST_COPYABLE_AND_MOVABLE(component_constructor_functor9)
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
        naming::gid_type create_with_args(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8)
        {
            return server::create<Component>(
                component_constructor_functor9<
                    Component, A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>(
                        boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ))
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
                BOOST_RV_REF(component_constructor_functor10) other)
              : a0(boost::move(other. a0)) , a1(boost::move(other. a1)) , a2(boost::move(other. a2)) , a3(boost::move(other. a3)) , a4(boost::move(other. a4)) , a5(boost::move(other. a5)) , a6(boost::move(other. a6)) , a7(boost::move(other. a7)) , a8(boost::move(other. a8)) , a9(boost::move(other. a9))
            {}
            component_constructor_functor10 & operator=(
                BOOST_COPY_ASSIGN_REF(component_constructor_functor10) other)
            {
                a0 = other. a0; a1 = other. a1; a2 = other. a2; a3 = other. a3; a4 = other. a4; a5 = other. a5; a6 = other. a6; a7 = other. a7; a8 = other. a8; a9 = other. a9;
                return *this;
            }
            component_constructor_functor10 & operator=(
                BOOST_RV_REF(component_constructor_functor10) other)
            {
                a0 = boost::move(other. a0); a1 = boost::move(other. a1); a2 = boost::move(other. a2); a3 = boost::move(other. a3); a4 = boost::move(other. a4); a5 = boost::move(other. a5); a6 = boost::move(other. a6); a7 = boost::move(other. a7); a8 = boost::move(other. a8); a9 = boost::move(other. a9);
                return *this;
            }
            template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
            component_constructor_functor10(
                BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7 , BOOST_FWD_REF(T8) t8 , BOOST_FWD_REF(T9) t9
            )
              : a0(boost::forward<T0> (t0)) , a1(boost::forward<T1> (t1)) , a2(boost::forward<T2> (t2)) , a3(boost::forward<T3> (t3)) , a4(boost::forward<T4> (t4)) , a5(boost::forward<T5> (t5)) , a6(boost::forward<T6> (t6)) , a7(boost::forward<T7> (t7)) , a8(boost::forward<T8> (t8)) , a9(boost::forward<T9> (t9))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(boost::move(a0) , boost::move(a1) , boost::move(a2) , boost::move(a3) , boost::move(a4) , boost::move(a5) , boost::move(a6) , boost::move(a7) , boost::move(a8) , boost::move(a9));
            }
            typename boost::remove_const<typename hpx::util::detail::remove_reference<A0>::type >::type a0; typename boost::remove_const<typename hpx::util::detail::remove_reference<A1>::type >::type a1; typename boost::remove_const<typename hpx::util::detail::remove_reference<A2>::type >::type a2; typename boost::remove_const<typename hpx::util::detail::remove_reference<A3>::type >::type a3; typename boost::remove_const<typename hpx::util::detail::remove_reference<A4>::type >::type a4; typename boost::remove_const<typename hpx::util::detail::remove_reference<A5>::type >::type a5; typename boost::remove_const<typename hpx::util::detail::remove_reference<A6>::type >::type a6; typename boost::remove_const<typename hpx::util::detail::remove_reference<A7>::type >::type a7; typename boost::remove_const<typename hpx::util::detail::remove_reference<A8>::type >::type a8; typename boost::remove_const<typename hpx::util::detail::remove_reference<A9>::type >::type a9;
            private:
                BOOST_COPYABLE_AND_MOVABLE(component_constructor_functor10)
        };
        template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
        naming::gid_type create_with_args(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9)
        {
            return server::create<Component>(
                component_constructor_functor10<
                    Component, A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>(
                        boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ))
            );
        }
