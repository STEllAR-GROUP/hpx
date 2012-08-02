// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    
    
    template <typename T, typename A0>
    lcos::future<object<T> >
    new_(naming::id_type const & target_id, BOOST_FWD_REF(A0) a0)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0
            >(
                boost::forward<A0>( a0 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1>
    lcos::future<object<T> >
    new_(naming::id_type const & target_id, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1
            >(
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2>
    lcos::future<object<T> >
    new_(naming::id_type const & target_id, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2
            >(
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3>
    lcos::future<object<T> >
    new_(naming::id_type const & target_id, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2 , A3
            >(
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    lcos::future<object<T> >
    new_(naming::id_type const & target_id, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2 , A3 , A4
            >(
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    lcos::future<object<T> >
    new_(naming::id_type const & target_id, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2 , A3 , A4 , A5
            >(
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    lcos::future<object<T> >
    new_(naming::id_type const & target_id, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2 , A3 , A4 , A5 , A6
            >(
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    lcos::future<object<T> >
    new_(naming::id_type const & target_id, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7
            >(
                boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
