// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    template <typename Component, typename Arg0>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, Arg0 && arg0)
    {
        return components::stub_base<Component>::create_async(locality,
            std::forward<Arg0>( arg0 ));
    }
    template <typename Component, typename Arg0>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_colocated(id_type const& id, Arg0 && arg0)
    {
        return components::stub_base<Component>::create_colocated_async(id,
            std::forward<Arg0>( arg0 ));
    }
    template <typename Component, typename Arg0 , typename Arg1>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, Arg0 && arg0 , Arg1 && arg1)
    {
        return components::stub_base<Component>::create_async(locality,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Component, typename Arg0 , typename Arg1>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_colocated(id_type const& id, Arg0 && arg0 , Arg1 && arg1)
    {
        return components::stub_base<Component>::create_colocated_async(id,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Component, typename Arg0 , typename Arg1 , typename Arg2>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return components::stub_base<Component>::create_async(locality,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Component, typename Arg0 , typename Arg1 , typename Arg2>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_colocated(id_type const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return components::stub_base<Component>::create_colocated_async(id,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Component, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return components::stub_base<Component>::create_async(locality,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Component, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_colocated(id_type const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return components::stub_base<Component>::create_colocated_async(id,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Component, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return components::stub_base<Component>::create_async(locality,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Component, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_colocated(id_type const& id, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return components::stub_base<Component>::create_colocated_async(id,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
