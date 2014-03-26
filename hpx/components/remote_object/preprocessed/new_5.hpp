// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    
    
    template <typename T, typename A0>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            launch::async
          , target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0
            >(
                std::forward<A0>( a0 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            launch::async
          , target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            launch::async
          , target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            launch::async
          , target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2 , A3
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            launch::async
          , target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2 , A3 , A4
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            launch::async
          , target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2 , A3 , A4 , A5
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            launch::async
          , target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2 , A3 , A4 , A5 , A6
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7)
    {
        lcos::packaged_action<
            remote_object::new_impl_action
          , object<T>
        > p;
        p.apply(
            launch::async
          , target_id
          , target_id
          , remote_object::ctor_fun<
                T
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
