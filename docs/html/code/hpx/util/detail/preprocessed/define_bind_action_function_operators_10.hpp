// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        
    template <typename U0>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ))
            );
    }
    
    template <typename U0>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ))
            );
    }
    template <typename U0>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ))
            );
    }
    template <typename U0 , typename U1>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ))
            );
    }
    
    template <typename U0 , typename U1>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ))
            );
    }
    template <typename U0 , typename U1>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ))
            );
    }
    template <typename U0 , typename U1 , typename U2>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ))
            );
    }
    
    template <typename U0 , typename U1 , typename U2>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ))
            );
    }
    template <typename U0 , typename U1 , typename U2>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ))
            );
    }
    
    template <typename U0 , typename U1 , typename U2 , typename U3>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ))
            );
    }
    
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ))
            );
    }
    
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ))
            );
    }
    
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ))
            );
    }
    
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ))
            );
    }
    
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ))
            );
    }
    
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ))
            );
    }
    
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ))
            );
    }
    
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12>
    BOOST_FORCEINLINE
    bool
    apply(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ))
            );
    }
    
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12>
    BOOST_FORCEINLINE
    hpx::lcos::future<result_type>
    async(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12>
    BOOST_FORCEINLINE
    result_type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ))
            );
    }
