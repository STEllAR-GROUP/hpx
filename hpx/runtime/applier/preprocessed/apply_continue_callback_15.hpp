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
    bool apply_continue_cb(
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
        typename Component, typename Signature, typename Derived
      , typename Callback
       
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
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
    bool apply_continue_cb(
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
        typename Component, typename Signature, typename Derived
      , typename Callback
       >
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
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
      , typename T0
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 )
          , cont);
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14
      , typename F>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14
      , F && f)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                std::forward<F>(f))
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14
      , typename F>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14
      , F && f)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 )
          , std::forward<F>(f));
    }
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    bool apply_continue_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14
      , naming::id_type const& cont)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;
        return apply_cb<Action>(
            new hpx::actions::typed_continuation<result_type>(
                cont, make_continuation())
          , gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ));
    }
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14
      , naming::id_type const& cont)
    {
        return apply_continue_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 )
          , cont);
    }
}
