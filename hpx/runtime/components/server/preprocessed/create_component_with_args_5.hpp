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
                      , typename util::decay<T0>::type
                    >::type
                >::type * dummy = 0
            )
              : a0(boost::forward<T0> (t0))
            {}
            result_type operator()(void* p)
            {
                new (p) typename Component::derived_type(boost::move(a0));
            }
            typename util::decay<A0>::type a0;
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
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1;
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
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1; typename util::decay<A2>::type a2;
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
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1; typename util::decay<A2>::type a2; typename util::decay<A3>::type a3;
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
            typename util::decay<A0>::type a0; typename util::decay<A1>::type a1; typename util::decay<A2>::type a2; typename util::decay<A3>::type a3; typename util::decay<A4>::type a4;
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
