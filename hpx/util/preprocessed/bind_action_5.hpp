// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util
{
    
    
    namespace detail
    {
        template <
            typename Action
          , typename Arg0
        >
        struct bound_action1
        {
            typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::result_type
            >::type result_type;
            
            bound_action1()
            {}
            template <typename A0>
            bound_action1(
                BOOST_FWD_REF(A0) a0
            )
                : arg0(boost::forward<A0>(a0))
            {}
            bound_action1(
                    bound_action1 const & other)
                : arg0(other.arg0)
            {}
            bound_action1(BOOST_RV_REF(
                    bound_action1) other)
                : arg0(boost::move(other.arg0))
            {}
            bound_action1 & operator=(
                BOOST_COPY_ASSIGN_REF(bound_action1) other)
            {
                arg0 = other.arg0;
                return *this;
            }
            bound_action1 & operator=(
                BOOST_RV_REF(bound_action1) other)
            {
                arg0 = boost::move(other.arg0);
                return *this;
            }
            
            bool apply()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  
                        );
            }
            bool apply() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  
                        );
            }
            template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) ); }
            
            
            hpx::lcos::future<result_type> async()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  
                        );
            }
            hpx::lcos::future<result_type> async() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  
                        );
            }
            template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) ); }
            
            BOOST_FORCEINLINE result_type operator()()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  
                        ).get();
            }
            BOOST_FORCEINLINE result_type operator()() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  
                        ).get();
            }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0)
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0) const
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7) const
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          
                ).get();
    }
            Arg0 arg0;
        };
        
        template <
            typename Env
          , typename Action
          , typename Arg0
        >
        typename detail::bound_action1<
                Action , Arg0
        >::result_type
        eval(
            Env & env
          , detail::bound_action1<
                Action
              , Arg0
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    detail::bound_action1<
                        Action
                      , Arg0
                    >
                >(f)(
                    env
                 );
        }
    }
    
    template <
        typename Action
      , typename A0
    >
    detail::bound_action1<
        Action
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type
    >
    bind(
        BOOST_FWD_REF(A0) a0
    )
    {
        return
            detail::bound_action1<
                Action
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type
            > (boost::forward<A0>( a0 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived
      , typename A0
    >
    detail::bound_action1<
        Derived
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , BOOST_FWD_REF(A0) a0
    )
    {
        return
            detail::bound_action1<
                Derived
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type
            > (boost::forward<A0>( a0 ));
    }
}}
namespace boost { namespace serialization
{
    
    template <
        typename Action
      , typename Arg0
    >
    void serialize(hpx::util::portable_binary_iarchive& ar
      , hpx::util::detail::bound_action1<
            Action
          , Arg0
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0;
    }
    template <
        typename Action
      , typename Arg0
    >
    void serialize(hpx::util::portable_binary_oarchive& ar
      , hpx::util::detail::bound_action1<
            Action
          , Arg0
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0;
    }
}}
namespace hpx { namespace util
{
    
    
    namespace detail
    {
        template <
            typename Action
          , typename Arg0 , typename Arg1
        >
        struct bound_action2
        {
            typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::result_type
            >::type result_type;
            
            bound_action2()
            {}
            template <typename A0 , typename A1>
            bound_action2(
                BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1
            )
                : arg0(boost::forward<A0>(a0)) , arg1(boost::forward<A1>(a1))
            {}
            bound_action2(
                    bound_action2 const & other)
                : arg0(other.arg0) , arg1(other.arg1)
            {}
            bound_action2(BOOST_RV_REF(
                    bound_action2) other)
                : arg0(boost::move(other.arg0)) , arg1(boost::move(other.arg1))
            {}
            bound_action2 & operator=(
                BOOST_COPY_ASSIGN_REF(bound_action2) other)
            {
                arg0 = other.arg0; arg1 = other.arg1;
                return *this;
            }
            bound_action2 & operator=(
                BOOST_RV_REF(bound_action2) other)
            {
                arg0 = boost::move(other.arg0); arg1 = boost::move(other.arg1);
                return *this;
            }
            
            bool apply()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1));
            }
            bool apply() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1));
            }
            template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); }
            
            
            hpx::lcos::future<result_type> async()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1));
            }
            hpx::lcos::future<result_type> async() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1));
            }
            template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1)); }
            
            BOOST_FORCEINLINE result_type operator()()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1)).get();
            }
            BOOST_FORCEINLINE result_type operator()() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1)).get();
            }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0)
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0) const
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7) const
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1)).get();
    }
            Arg0 arg0; Arg1 arg1;
        };
        
        template <
            typename Env
          , typename Action
          , typename Arg0 , typename Arg1
        >
        typename detail::bound_action2<
                Action , Arg0 , Arg1
        >::result_type
        eval(
            Env & env
          , detail::bound_action2<
                Action
              , Arg0 , Arg1
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    detail::bound_action2<
                        Action
                      , Arg0 , Arg1
                    >
                >(f)(
                    env
                 );
        }
    }
    
    template <
        typename Action
      , typename A0 , typename A1
    >
    detail::bound_action2<
        Action
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type
    >
    bind(
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1
    )
    {
        return
            detail::bound_action2<
                Action
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived
      , typename A0 , typename A1
    >
    detail::bound_action2<
        Derived
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1
    )
    {
        return
            detail::bound_action2<
                Derived
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
    }
}}
namespace boost { namespace serialization
{
    
    template <
        typename Action
      , typename Arg0 , typename Arg1
    >
    void serialize(hpx::util::portable_binary_iarchive& ar
      , hpx::util::detail::bound_action2<
            Action
          , Arg0 , Arg1
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1;
    }
    template <
        typename Action
      , typename Arg0 , typename Arg1
    >
    void serialize(hpx::util::portable_binary_oarchive& ar
      , hpx::util::detail::bound_action2<
            Action
          , Arg0 , Arg1
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1;
    }
}}
namespace hpx { namespace util
{
    
    
    namespace detail
    {
        template <
            typename Action
          , typename Arg0 , typename Arg1 , typename Arg2
        >
        struct bound_action3
        {
            typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::result_type
            >::type result_type;
            
            bound_action3()
            {}
            template <typename A0 , typename A1 , typename A2>
            bound_action3(
                BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2
            )
                : arg0(boost::forward<A0>(a0)) , arg1(boost::forward<A1>(a1)) , arg2(boost::forward<A2>(a2))
            {}
            bound_action3(
                    bound_action3 const & other)
                : arg0(other.arg0) , arg1(other.arg1) , arg2(other.arg2)
            {}
            bound_action3(BOOST_RV_REF(
                    bound_action3) other)
                : arg0(boost::move(other.arg0)) , arg1(boost::move(other.arg1)) , arg2(boost::move(other.arg2))
            {}
            bound_action3 & operator=(
                BOOST_COPY_ASSIGN_REF(bound_action3) other)
            {
                arg0 = other.arg0; arg1 = other.arg1; arg2 = other.arg2;
                return *this;
            }
            bound_action3 & operator=(
                BOOST_RV_REF(bound_action3) other)
            {
                arg0 = boost::move(other.arg0); arg1 = boost::move(other.arg1); arg2 = boost::move(other.arg2);
                return *this;
            }
            
            bool apply()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2));
            }
            bool apply() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2));
            }
            template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); }
            
            
            hpx::lcos::future<result_type> async()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2));
            }
            hpx::lcos::future<result_type> async() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2));
            }
            template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)); }
            
            BOOST_FORCEINLINE result_type operator()()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
            }
            BOOST_FORCEINLINE result_type operator()() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
            }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0)
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0) const
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7) const
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2)).get();
    }
            Arg0 arg0; Arg1 arg1; Arg2 arg2;
        };
        
        template <
            typename Env
          , typename Action
          , typename Arg0 , typename Arg1 , typename Arg2
        >
        typename detail::bound_action3<
                Action , Arg0 , Arg1 , Arg2
        >::result_type
        eval(
            Env & env
          , detail::bound_action3<
                Action
              , Arg0 , Arg1 , Arg2
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    detail::bound_action3<
                        Action
                      , Arg0 , Arg1 , Arg2
                    >
                >(f)(
                    env
                 );
        }
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2
    >
    detail::bound_action3<
        Action
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type
    >
    bind(
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2
    )
    {
        return
            detail::bound_action3<
                Action
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived
      , typename A0 , typename A1 , typename A2
    >
    detail::bound_action3<
        Derived
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2
    )
    {
        return
            detail::bound_action3<
                Derived
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
    }
}}
namespace boost { namespace serialization
{
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2
    >
    void serialize(hpx::util::portable_binary_iarchive& ar
      , hpx::util::detail::bound_action3<
            Action
          , Arg0 , Arg1 , Arg2
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2;
    }
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2
    >
    void serialize(hpx::util::portable_binary_oarchive& ar
      , hpx::util::detail::bound_action3<
            Action
          , Arg0 , Arg1 , Arg2
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2;
    }
}}
namespace hpx { namespace util
{
    
    
    namespace detail
    {
        template <
            typename Action
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
        >
        struct bound_action4
        {
            typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::result_type
            >::type result_type;
            
            bound_action4()
            {}
            template <typename A0 , typename A1 , typename A2 , typename A3>
            bound_action4(
                BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3
            )
                : arg0(boost::forward<A0>(a0)) , arg1(boost::forward<A1>(a1)) , arg2(boost::forward<A2>(a2)) , arg3(boost::forward<A3>(a3))
            {}
            bound_action4(
                    bound_action4 const & other)
                : arg0(other.arg0) , arg1(other.arg1) , arg2(other.arg2) , arg3(other.arg3)
            {}
            bound_action4(BOOST_RV_REF(
                    bound_action4) other)
                : arg0(boost::move(other.arg0)) , arg1(boost::move(other.arg1)) , arg2(boost::move(other.arg2)) , arg3(boost::move(other.arg3))
            {}
            bound_action4 & operator=(
                BOOST_COPY_ASSIGN_REF(bound_action4) other)
            {
                arg0 = other.arg0; arg1 = other.arg1; arg2 = other.arg2; arg3 = other.arg3;
                return *this;
            }
            bound_action4 & operator=(
                BOOST_RV_REF(bound_action4) other)
            {
                arg0 = boost::move(other.arg0); arg1 = boost::move(other.arg1); arg2 = boost::move(other.arg2); arg3 = boost::move(other.arg3);
                return *this;
            }
            
            bool apply()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3));
            }
            bool apply() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3));
            }
            template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); }
            
            
            hpx::lcos::future<result_type> async()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3));
            }
            hpx::lcos::future<result_type> async() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3));
            }
            template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)); }
            
            BOOST_FORCEINLINE result_type operator()()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
            }
            BOOST_FORCEINLINE result_type operator()() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
            }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0)
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0) const
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7) const
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3)).get();
    }
            Arg0 arg0; Arg1 arg1; Arg2 arg2; Arg3 arg3;
        };
        
        template <
            typename Env
          , typename Action
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
        >
        typename detail::bound_action4<
                Action , Arg0 , Arg1 , Arg2 , Arg3
        >::result_type
        eval(
            Env & env
          , detail::bound_action4<
                Action
              , Arg0 , Arg1 , Arg2 , Arg3
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    detail::bound_action4<
                        Action
                      , Arg0 , Arg1 , Arg2 , Arg3
                    >
                >(f)(
                    env
                 );
        }
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    detail::bound_action4<
        Action
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type
    >
    bind(
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3
    )
    {
        return
            detail::bound_action4<
                Action
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    detail::bound_action4<
        Derived
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3
    )
    {
        return
            detail::bound_action4<
                Derived
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
    }
}}
namespace boost { namespace serialization
{
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
    >
    void serialize(hpx::util::portable_binary_iarchive& ar
      , hpx::util::detail::bound_action4<
            Action
          , Arg0 , Arg1 , Arg2 , Arg3
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2; ar & bound.arg3;
    }
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
    >
    void serialize(hpx::util::portable_binary_oarchive& ar
      , hpx::util::detail::bound_action4<
            Action
          , Arg0 , Arg1 , Arg2 , Arg3
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2; ar & bound.arg3;
    }
}}
namespace hpx { namespace util
{
    
    
    namespace detail
    {
        template <
            typename Action
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
        >
        struct bound_action5
        {
            typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::result_type
            >::type result_type;
            
            bound_action5()
            {}
            template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
            bound_action5(
                BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4
            )
                : arg0(boost::forward<A0>(a0)) , arg1(boost::forward<A1>(a1)) , arg2(boost::forward<A2>(a2)) , arg3(boost::forward<A3>(a3)) , arg4(boost::forward<A4>(a4))
            {}
            bound_action5(
                    bound_action5 const & other)
                : arg0(other.arg0) , arg1(other.arg1) , arg2(other.arg2) , arg3(other.arg3) , arg4(other.arg4)
            {}
            bound_action5(BOOST_RV_REF(
                    bound_action5) other)
                : arg0(boost::move(other.arg0)) , arg1(boost::move(other.arg1)) , arg2(boost::move(other.arg2)) , arg3(boost::move(other.arg3)) , arg4(boost::move(other.arg4))
            {}
            bound_action5 & operator=(
                BOOST_COPY_ASSIGN_REF(bound_action5) other)
            {
                arg0 = other.arg0; arg1 = other.arg1; arg2 = other.arg2; arg3 = other.arg3; arg4 = other.arg4;
                return *this;
            }
            bound_action5 & operator=(
                BOOST_RV_REF(bound_action5) other)
            {
                arg0 = boost::move(other.arg0); arg1 = boost::move(other.arg1); arg2 = boost::move(other.arg2); arg3 = boost::move(other.arg3); arg4 = boost::move(other.arg4);
                return *this;
            }
            
            bool apply()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4));
            }
            bool apply() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4));
            }
            template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); }
            
            
            hpx::lcos::future<result_type> async()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4));
            }
            hpx::lcos::future<result_type> async() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4));
            }
            template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)); }
            
            BOOST_FORCEINLINE result_type operator()()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
            }
            BOOST_FORCEINLINE result_type operator()() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
            }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0)
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0) const
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7) const
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4)).get();
    }
            Arg0 arg0; Arg1 arg1; Arg2 arg2; Arg3 arg3; Arg4 arg4;
        };
        
        template <
            typename Env
          , typename Action
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
        >
        typename detail::bound_action5<
                Action , Arg0 , Arg1 , Arg2 , Arg3 , Arg4
        >::result_type
        eval(
            Env & env
          , detail::bound_action5<
                Action
              , Arg0 , Arg1 , Arg2 , Arg3 , Arg4
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    detail::bound_action5<
                        Action
                      , Arg0 , Arg1 , Arg2 , Arg3 , Arg4
                    >
                >(f)(
                    env
                 );
        }
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    detail::bound_action5<
        Action
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type
    >
    bind(
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4
    )
    {
        return
            detail::bound_action5<
                Action
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    detail::bound_action5<
        Derived
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4
    )
    {
        return
            detail::bound_action5<
                Derived
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
    }
}}
namespace boost { namespace serialization
{
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
    >
    void serialize(hpx::util::portable_binary_iarchive& ar
      , hpx::util::detail::bound_action5<
            Action
          , Arg0 , Arg1 , Arg2 , Arg3 , Arg4
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2; ar & bound.arg3; ar & bound.arg4;
    }
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
    >
    void serialize(hpx::util::portable_binary_oarchive& ar
      , hpx::util::detail::bound_action5<
            Action
          , Arg0 , Arg1 , Arg2 , Arg3 , Arg4
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2; ar & bound.arg3; ar & bound.arg4;
    }
}}
namespace hpx { namespace util
{
    
    
    namespace detail
    {
        template <
            typename Action
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
        >
        struct bound_action6
        {
            typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::result_type
            >::type result_type;
            
            bound_action6()
            {}
            template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
            bound_action6(
                BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5
            )
                : arg0(boost::forward<A0>(a0)) , arg1(boost::forward<A1>(a1)) , arg2(boost::forward<A2>(a2)) , arg3(boost::forward<A3>(a3)) , arg4(boost::forward<A4>(a4)) , arg5(boost::forward<A5>(a5))
            {}
            bound_action6(
                    bound_action6 const & other)
                : arg0(other.arg0) , arg1(other.arg1) , arg2(other.arg2) , arg3(other.arg3) , arg4(other.arg4) , arg5(other.arg5)
            {}
            bound_action6(BOOST_RV_REF(
                    bound_action6) other)
                : arg0(boost::move(other.arg0)) , arg1(boost::move(other.arg1)) , arg2(boost::move(other.arg2)) , arg3(boost::move(other.arg3)) , arg4(boost::move(other.arg4)) , arg5(boost::move(other.arg5))
            {}
            bound_action6 & operator=(
                BOOST_COPY_ASSIGN_REF(bound_action6) other)
            {
                arg0 = other.arg0; arg1 = other.arg1; arg2 = other.arg2; arg3 = other.arg3; arg4 = other.arg4; arg5 = other.arg5;
                return *this;
            }
            bound_action6 & operator=(
                BOOST_RV_REF(bound_action6) other)
            {
                arg0 = boost::move(other.arg0); arg1 = boost::move(other.arg1); arg2 = boost::move(other.arg2); arg3 = boost::move(other.arg3); arg4 = boost::move(other.arg4); arg5 = boost::move(other.arg5);
                return *this;
            }
            
            bool apply()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5));
            }
            bool apply() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5));
            }
            template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); }
            
            
            hpx::lcos::future<result_type> async()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5));
            }
            hpx::lcos::future<result_type> async() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5));
            }
            template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)); }
            
            BOOST_FORCEINLINE result_type operator()()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
            }
            BOOST_FORCEINLINE result_type operator()() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
            }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0)
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0) const
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7) const
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5)).get();
    }
            Arg0 arg0; Arg1 arg1; Arg2 arg2; Arg3 arg3; Arg4 arg4; Arg5 arg5;
        };
        
        template <
            typename Env
          , typename Action
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
        >
        typename detail::bound_action6<
                Action , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5
        >::result_type
        eval(
            Env & env
          , detail::bound_action6<
                Action
              , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    detail::bound_action6<
                        Action
                      , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5
                    >
                >(f)(
                    env
                 );
        }
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
    >
    detail::bound_action6<
        Action
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type
    >
    bind(
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5
    )
    {
        return
            detail::bound_action6<
                Action
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
    >
    detail::bound_action6<
        Derived
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5
    )
    {
        return
            detail::bound_action6<
                Derived
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
    }
}}
namespace boost { namespace serialization
{
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
    >
    void serialize(hpx::util::portable_binary_iarchive& ar
      , hpx::util::detail::bound_action6<
            Action
          , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2; ar & bound.arg3; ar & bound.arg4; ar & bound.arg5;
    }
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
    >
    void serialize(hpx::util::portable_binary_oarchive& ar
      , hpx::util::detail::bound_action6<
            Action
          , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2; ar & bound.arg3; ar & bound.arg4; ar & bound.arg5;
    }
}}
namespace hpx { namespace util
{
    
    
    namespace detail
    {
        template <
            typename Action
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
        >
        struct bound_action7
        {
            typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::result_type
            >::type result_type;
            
            bound_action7()
            {}
            template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
            bound_action7(
                BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6
            )
                : arg0(boost::forward<A0>(a0)) , arg1(boost::forward<A1>(a1)) , arg2(boost::forward<A2>(a2)) , arg3(boost::forward<A3>(a3)) , arg4(boost::forward<A4>(a4)) , arg5(boost::forward<A5>(a5)) , arg6(boost::forward<A6>(a6))
            {}
            bound_action7(
                    bound_action7 const & other)
                : arg0(other.arg0) , arg1(other.arg1) , arg2(other.arg2) , arg3(other.arg3) , arg4(other.arg4) , arg5(other.arg5) , arg6(other.arg6)
            {}
            bound_action7(BOOST_RV_REF(
                    bound_action7) other)
                : arg0(boost::move(other.arg0)) , arg1(boost::move(other.arg1)) , arg2(boost::move(other.arg2)) , arg3(boost::move(other.arg3)) , arg4(boost::move(other.arg4)) , arg5(boost::move(other.arg5)) , arg6(boost::move(other.arg6))
            {}
            bound_action7 & operator=(
                BOOST_COPY_ASSIGN_REF(bound_action7) other)
            {
                arg0 = other.arg0; arg1 = other.arg1; arg2 = other.arg2; arg3 = other.arg3; arg4 = other.arg4; arg5 = other.arg5; arg6 = other.arg6;
                return *this;
            }
            bound_action7 & operator=(
                BOOST_RV_REF(bound_action7) other)
            {
                arg0 = boost::move(other.arg0); arg1 = boost::move(other.arg1); arg2 = boost::move(other.arg2); arg3 = boost::move(other.arg3); arg4 = boost::move(other.arg4); arg5 = boost::move(other.arg5); arg6 = boost::move(other.arg6);
                return *this;
            }
            
            bool apply()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6));
            }
            bool apply() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6));
            }
            template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); }
            
            
            hpx::lcos::future<result_type> async()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6));
            }
            hpx::lcos::future<result_type> async() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6));
            }
            template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)); }
            
            BOOST_FORCEINLINE result_type operator()()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
            }
            BOOST_FORCEINLINE result_type operator()() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
            }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0)
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0) const
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7) const
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6)).get();
    }
            Arg0 arg0; Arg1 arg1; Arg2 arg2; Arg3 arg3; Arg4 arg4; Arg5 arg5; Arg6 arg6;
        };
        
        template <
            typename Env
          , typename Action
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
        >
        typename detail::bound_action7<
                Action , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6
        >::result_type
        eval(
            Env & env
          , detail::bound_action7<
                Action
              , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    detail::bound_action7<
                        Action
                      , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6
                    >
                >(f)(
                    env
                 );
        }
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
    >
    detail::bound_action7<
        Action
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type , typename boost::remove_const< typename detail::remove_reference<A6>::type>::type
    >
    bind(
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6
    )
    {
        return
            detail::bound_action7<
                Action
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type , typename boost::remove_const< typename detail::remove_reference<A6>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
    >
    detail::bound_action7<
        Derived
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type , typename boost::remove_const< typename detail::remove_reference<A6>::type>::type
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6
    )
    {
        return
            detail::bound_action7<
                Derived
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type , typename boost::remove_const< typename detail::remove_reference<A6>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
    }
}}
namespace boost { namespace serialization
{
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
    >
    void serialize(hpx::util::portable_binary_iarchive& ar
      , hpx::util::detail::bound_action7<
            Action
          , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2; ar & bound.arg3; ar & bound.arg4; ar & bound.arg5; ar & bound.arg6;
    }
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
    >
    void serialize(hpx::util::portable_binary_oarchive& ar
      , hpx::util::detail::bound_action7<
            Action
          , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2; ar & bound.arg3; ar & bound.arg4; ar & bound.arg5; ar & bound.arg6;
    }
}}
namespace hpx { namespace util
{
    
    
    namespace detail
    {
        template <
            typename Action
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
        >
        struct bound_action8
        {
            typedef typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::result_type
            >::type result_type;
            
            bound_action8()
            {}
            template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
            bound_action8(
                BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7
            )
                : arg0(boost::forward<A0>(a0)) , arg1(boost::forward<A1>(a1)) , arg2(boost::forward<A2>(a2)) , arg3(boost::forward<A3>(a3)) , arg4(boost::forward<A4>(a4)) , arg5(boost::forward<A5>(a5)) , arg6(boost::forward<A6>(a6)) , arg7(boost::forward<A7>(a7))
            {}
            bound_action8(
                    bound_action8 const & other)
                : arg0(other.arg0) , arg1(other.arg1) , arg2(other.arg2) , arg3(other.arg3) , arg4(other.arg4) , arg5(other.arg5) , arg6(other.arg6) , arg7(other.arg7)
            {}
            bound_action8(BOOST_RV_REF(
                    bound_action8) other)
                : arg0(boost::move(other.arg0)) , arg1(boost::move(other.arg1)) , arg2(boost::move(other.arg2)) , arg3(boost::move(other.arg3)) , arg4(boost::move(other.arg4)) , arg5(boost::move(other.arg5)) , arg6(boost::move(other.arg6)) , arg7(boost::move(other.arg7))
            {}
            bound_action8 & operator=(
                BOOST_COPY_ASSIGN_REF(bound_action8) other)
            {
                arg0 = other.arg0; arg1 = other.arg1; arg2 = other.arg2; arg3 = other.arg3; arg4 = other.arg4; arg5 = other.arg5; arg6 = other.arg6; arg7 = other.arg7;
                return *this;
            }
            bound_action8 & operator=(
                BOOST_RV_REF(bound_action8) other)
            {
                arg0 = boost::move(other.arg0); arg1 = boost::move(other.arg1); arg2 = boost::move(other.arg2); arg3 = boost::move(other.arg3); arg4 = boost::move(other.arg4); arg5 = boost::move(other.arg5); arg6 = boost::move(other.arg6); arg7 = boost::move(other.arg7);
                return *this;
            }
            
            bool apply()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7));
            }
            bool apply() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::apply<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7));
            }
            template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0> bool apply(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> bool apply(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::apply<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); }
            
            
            hpx::lcos::future<result_type> async()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7));
            }
            hpx::lcos::future<result_type> async() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7));
            }
            template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0) const { typedef hpx::util::tuple1< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type > env_type; env_type env(boost::forward<A0>( a0 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const { typedef hpx::util::tuple2< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const { typedef hpx::util::tuple3< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const { typedef hpx::util::tuple4< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const { typedef hpx::util::tuple5< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const { typedef hpx::util::tuple6< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); } template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6> hpx::lcos::future<result_type> async(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const { typedef hpx::util::tuple7< typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type > env_type; env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )); return hpx::async<Action>( hpx::util::detail::eval(env, arg0) , hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)); }
            
            BOOST_FORCEINLINE result_type operator()()
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
            }
            BOOST_FORCEINLINE result_type operator()() const
            {
                typedef hpx::util::tuple0<> env_type;
                env_type env;
                return hpx::async<Action>(
                    hpx::util::detail::eval(env, arg0)
                  ,
                        hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
            }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0)
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0) const
    {
        typedef
            hpx::util::tuple1<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1) const
    {
        typedef
            hpx::util::tuple2<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2) const
    {
        typedef
            hpx::util::tuple3<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3) const
    {
        typedef
            hpx::util::tuple4<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4) const
    {
        typedef
            hpx::util::tuple5<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5) const
    {
        typedef
            hpx::util::tuple6<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6) const
    {
        typedef
            hpx::util::tuple7<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    BOOST_FORCEINLINE result_type operator()(BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7) const
    {
        typedef
            hpx::util::tuple8<
                typename detail::env_value_type<BOOST_FWD_REF(A0)>::type , typename detail::env_value_type<BOOST_FWD_REF(A1)>::type , typename detail::env_value_type<BOOST_FWD_REF(A2)>::type , typename detail::env_value_type<BOOST_FWD_REF(A3)>::type , typename detail::env_value_type<BOOST_FWD_REF(A4)>::type , typename detail::env_value_type<BOOST_FWD_REF(A5)>::type , typename detail::env_value_type<BOOST_FWD_REF(A6)>::type , typename detail::env_value_type<BOOST_FWD_REF(A7)>::type
            >
            env_type;
        env_type env(boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          ,
                hpx::util::detail::eval(env, arg1) , hpx::util::detail::eval(env, arg2) , hpx::util::detail::eval(env, arg3) , hpx::util::detail::eval(env, arg4) , hpx::util::detail::eval(env, arg5) , hpx::util::detail::eval(env, arg6) , hpx::util::detail::eval(env, arg7)).get();
    }
            Arg0 arg0; Arg1 arg1; Arg2 arg2; Arg3 arg3; Arg4 arg4; Arg5 arg5; Arg6 arg6; Arg7 arg7;
        };
        
        template <
            typename Env
          , typename Action
          , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
        >
        typename detail::bound_action8<
                Action , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7
        >::result_type
        eval(
            Env & env
          , detail::bound_action8<
                Action
              , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7
            > const & f
        )
        {
            return
                boost::fusion::fused<
                    detail::bound_action8<
                        Action
                      , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7
                    >
                >(f)(
                    env
                 );
        }
    }
    
    template <
        typename Action
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
    >
    detail::bound_action8<
        Action
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type , typename boost::remove_const< typename detail::remove_reference<A6>::type>::type , typename boost::remove_const< typename detail::remove_reference<A7>::type>::type
    >
    bind(
        BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7
    )
    {
        return
            detail::bound_action8<
                Action
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type , typename boost::remove_const< typename detail::remove_reference<A6>::type>::type , typename boost::remove_const< typename detail::remove_reference<A7>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
    >
    detail::bound_action8<
        Derived
      , typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type , typename boost::remove_const< typename detail::remove_reference<A6>::type>::type , typename boost::remove_const< typename detail::remove_reference<A7>::type>::type
    >
    bind(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7
    )
    {
        return
            detail::bound_action8<
                Derived
              ,
                  typename boost::remove_const< typename detail::remove_reference<A0>::type>::type , typename boost::remove_const< typename detail::remove_reference<A1>::type>::type , typename boost::remove_const< typename detail::remove_reference<A2>::type>::type , typename boost::remove_const< typename detail::remove_reference<A3>::type>::type , typename boost::remove_const< typename detail::remove_reference<A4>::type>::type , typename boost::remove_const< typename detail::remove_reference<A5>::type>::type , typename boost::remove_const< typename detail::remove_reference<A6>::type>::type , typename boost::remove_const< typename detail::remove_reference<A7>::type>::type
            > (boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
    }
}}
namespace boost { namespace serialization
{
    
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
    >
    void serialize(hpx::util::portable_binary_iarchive& ar
      , hpx::util::detail::bound_action8<
            Action
          , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2; ar & bound.arg3; ar & bound.arg4; ar & bound.arg5; ar & bound.arg6; ar & bound.arg7;
    }
    template <
        typename Action
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
    >
    void serialize(hpx::util::portable_binary_oarchive& ar
      , hpx::util::detail::bound_action8<
            Action
          , Arg0 , Arg1 , Arg2 , Arg3 , Arg4 , Arg5 , Arg6 , Arg7
        >& bound
      , unsigned int const)
    {
        ar & bound.arg0; ar & bound.arg1; ar & bound.arg2; ar & bound.arg3; ar & bound.arg4; ar & bound.arg5; ar & bound.arg6; ar & bound.arg7;
    }
}}
