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
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<
            typename Action::arguments_type>::value == 1>
    >::type
    apply_continue(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0,
        BOOST_FWD_REF(F) f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type remote_result_type;
        apply<Action>(
            new hpx::actions::typed_continuation<remote_result_type>(
                boost::forward<F>(f))
          , gid
          , boost::forward<Arg0>( arg0 ));
    }
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 1>
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0,
        BOOST_FWD_REF(F) f)
    {
        return apply_continue<Derived>(gid, boost::forward<Arg0>( arg0 ),
            boost::forward<F>(f));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1,
        typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<
            typename Action::arguments_type>::value == 2>
    >::type
    apply_continue(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1,
        BOOST_FWD_REF(F) f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type remote_result_type;
        apply<Action>(
            new hpx::actions::typed_continuation<remote_result_type>(
                boost::forward<F>(f))
          , gid
          , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 2>
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1,
        BOOST_FWD_REF(F) f)
    {
        return apply_continue<Derived>(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ),
            boost::forward<F>(f));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2,
        typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<
            typename Action::arguments_type>::value == 3>
    >::type
    apply_continue(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2,
        BOOST_FWD_REF(F) f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type remote_result_type;
        apply<Action>(
            new hpx::actions::typed_continuation<remote_result_type>(
                boost::forward<F>(f))
          , gid
          , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 3>
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2,
        BOOST_FWD_REF(F) f)
    {
        return apply_continue<Derived>(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ),
            boost::forward<F>(f));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3,
        typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<
            typename Action::arguments_type>::value == 4>
    >::type
    apply_continue(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3,
        BOOST_FWD_REF(F) f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type remote_result_type;
        apply<Action>(
            new hpx::actions::typed_continuation<remote_result_type>(
                boost::forward<F>(f))
          , gid
          , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 4>
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3,
        BOOST_FWD_REF(F) f)
    {
        return apply_continue<Derived>(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ),
            boost::forward<F>(f));
    }
}
namespace hpx
{
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4,
        typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<
            typename Action::arguments_type>::value == 5>
    >::type
    apply_continue(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4,
        BOOST_FWD_REF(F) f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type remote_result_type;
        apply<Action>(
            new hpx::actions::typed_continuation<remote_result_type>(
                boost::forward<F>(f))
          , gid
          , boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4, typename F>
    typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 5>
    >::type
    apply_continue(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4,
        BOOST_FWD_REF(F) f)
    {
        return apply_continue<Derived>(gid, boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ),
            boost::forward<F>(f));
    }
}
