// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <
        typename Action
      , typename Callback
       
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 0,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
       
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
           );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
       
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 0,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
       
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
           
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
       >
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 0,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
       
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
           );
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
       >
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 0,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
       
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
           
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 1,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 1,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 1,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 1,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 2,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 2,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 2,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 2,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 3,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 3,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 3,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 3,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 4,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 4,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 4,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 4,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 5,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 5,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 5,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 5,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 6,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 6,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 6,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 6,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 7,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 7,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 7,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 7,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 8,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 8,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 8,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 8,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 9,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 9,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 9,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 9,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 10,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 10,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 10,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 10,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 11,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 11,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 11,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 11,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 12,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 12,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 12,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 12,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 13,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 13,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 13,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 13,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 14,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 14,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 14,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 14,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 15,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 15,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 15,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 15,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 16,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 16,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 16,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 16,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 17,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 17,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 17,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 17,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 18,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 18,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 18,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 18,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 19,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 19,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 19,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 19,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 20,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18 , Arg19 && arg19
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ) , std::forward<Arg19>( arg19 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19
      , typename F>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 20,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18 , Arg19 && arg19
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ) , std::forward<Arg19>( arg19 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 20,
        bool
    >::type
    apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18 , Arg19 && arg19
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ) , std::forward<Arg19>( arg19 ));
    }
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9 , typename Arg10 , typename Arg11 , typename Arg12 , typename Arg13 , typename Arg14 , typename Arg15 , typename Arg16 , typename Arg17 , typename Arg18 , typename Arg19>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 20,
        bool
    >::type
    apply_continue_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9 , Arg10 && arg10 , Arg11 && arg11 , Arg12 && arg12 , Arg13 && arg13 , Arg14 && arg14 , Arg15 && arg15 , Arg16 && arg16 , Arg17 && arg17 , Arg18 && arg18 , Arg19 && arg19
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ) , std::forward<Arg10>( arg10 ) , std::forward<Arg11>( arg11 ) , std::forward<Arg12>( arg12 ) , std::forward<Arg13>( arg13 ) , std::forward<Arg14>( arg14 ) , std::forward<Arg15>( arg15 ) , std::forward<Arg16>( arg16 ) , std::forward<Arg17>( arg17 ) , std::forward<Arg18>( arg18 ) , std::forward<Arg19>( arg19 )
          , cont);
    }
}
