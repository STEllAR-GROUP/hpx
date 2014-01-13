// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename Action, typename Arg0,
        typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 1,
        bool
    >::type
    apply_continue(naming::id_type const& gid, Arg0 && arg0,
        F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Arg0>( arg0 ));
    }
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0, typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 1,
        bool
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, Arg0 && arg0,
        F && f)
    {
        return apply_continue<Derived>(gid, std::forward<Arg0>( arg0 ),
            std::forward<F>(f));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1,
        typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 2,
        bool
    >::type
    apply_continue(naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1,
        F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1, typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 2,
        bool
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1,
        F && f)
    {
        return apply_continue<Derived>(gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ),
            std::forward<F>(f));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2,
        typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 3,
        bool
    >::type
    apply_continue(naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2,
        F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2, typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 3,
        bool
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2,
        F && f)
    {
        return apply_continue<Derived>(gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ),
            std::forward<F>(f));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3,
        typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 4,
        bool
    >::type
    apply_continue(naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3,
        F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3, typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 4,
        bool
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3,
        F && f)
    {
        return apply_continue<Derived>(gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ),
            std::forward<F>(f));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4,
        typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 5,
        bool
    >::type
    apply_continue(naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4,
        F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4, typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 5,
        bool
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4,
        F && f)
    {
        return apply_continue<Derived>(gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ),
            std::forward<F>(f));
    }
}
