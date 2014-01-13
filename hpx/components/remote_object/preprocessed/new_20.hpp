// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    
    
    template <typename T, typename A0>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
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
        traits::is_component<T>, lcos::unique_future<object<T> >
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
        traits::is_component<T>, lcos::unique_future<object<T> >
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
        traits::is_component<T>, lcos::unique_future<object<T> >
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
        traits::is_component<T>, lcos::unique_future<object<T> >
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
        traits::is_component<T>, lcos::unique_future<object<T> >
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
        traits::is_component<T>, lcos::unique_future<object<T> >
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
        traits::is_component<T>, lcos::unique_future<object<T> >
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19 , A20 && a20)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 ) , std::forward<A20>( a20 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19 , A20 && a20 , A21 && a21)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 ) , std::forward<A20>( a20 ) , std::forward<A21>( a21 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22>
    inline typename boost::disable_if<
        traits::is_component<T>, lcos::unique_future<object<T> >
    >::type
    new_(naming::id_type const & target_id, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19 , A20 && a20 , A21 && a21 , A22 && a22)
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
              , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21 , A22
            >(
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 ) , std::forward<A20>( a20 ) , std::forward<A21>( a21 ) , std::forward<A22>( a22 )
            )
          , remote_object::dtor_fun<T>()
        );
        return p.get_future();
    }
