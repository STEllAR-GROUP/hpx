// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    template <typename Component, typename Arg0>
    inline typename boost::enable_if<
        traits::is_component<Component>, 
        lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, BOOST_FWD_REF(Arg0) arg0)
    {
        return components::stub_base<Component>::create_async(locality,
            boost::forward<Arg0>( arg0 ));
    }
    template <typename Component, typename Arg0 , typename Arg1>
    inline typename boost::enable_if<
        traits::is_component<Component>, 
        lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return components::stub_base<Component>::create_async(locality,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    template <typename Component, typename Arg0 , typename Arg1 , typename Arg2>
    inline typename boost::enable_if<
        traits::is_component<Component>, 
        lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return components::stub_base<Component>::create_async(locality,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    template <typename Component, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline typename boost::enable_if<
        traits::is_component<Component>, 
        lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return components::stub_base<Component>::create_async(locality,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    template <typename Component, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline typename boost::enable_if<
        traits::is_component<Component>, 
        lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return components::stub_base<Component>::create_async(locality,
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
