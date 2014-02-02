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
    apply(U0 && u0) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ))
            );
    }
    template <typename U0>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ))
            );
    }
    template <typename U0>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ))
            );
    }
    template <typename U0 , typename U1>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ))
            );
    }
    template <typename U0 , typename U1>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ))
            );
    }
    template <typename U0 , typename U1>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ))
            );
    }
    template <typename U0 , typename U1 , typename U2>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ))
            );
    }
    template <typename U0 , typename U1 , typename U2>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ))
            );
    }
    template <typename U0 , typename U1 , typename U2>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19 , typename U20>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19 , U20 && u20) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ) , std::forward<U20>( u20 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19 , typename U20>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19 , U20 && u20) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ) , std::forward<U20>( u20 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19 , typename U20>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19 , U20 && u20) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ) , std::forward<U20>( u20 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19 , typename U20 , typename U21>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19 , U20 && u20 , U21 && u21) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ) , std::forward<U20>( u20 ) , std::forward<U21>( u21 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19 , typename U20 , typename U21>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19 , U20 && u20 , U21 && u21) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ) , std::forward<U20>( u20 ) , std::forward<U21>( u21 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19 , typename U20 , typename U21>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19 , U20 && u20 , U21 && u21) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ) , std::forward<U20>( u20 ) , std::forward<U21>( u21 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19 , typename U20 , typename U21 , typename U22>
    BOOST_FORCEINLINE
    bool
    apply(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19 , U20 && u20 , U21 && u21 , U22 && u22) const
    {
        return
            detail::bind_action_apply<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ) , std::forward<U20>( u20 ) , std::forward<U21>( u21 ) , std::forward<U22>( u22 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19 , typename U20 , typename U21 , typename U22>
    BOOST_FORCEINLINE
    hpx::lcos::unique_future<result_type>
    async(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19 , U20 && u20 , U21 && u21 , U22 && u22) const
    {
        return
            detail::bind_action_async<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ) , std::forward<U20>( u20 ) , std::forward<U21>( u21 ) , std::forward<U22>( u22 ))
            );
    }
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17 , typename U18 , typename U19 , typename U20 , typename U21 , typename U22>
    BOOST_FORCEINLINE
    result_type
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17 , U18 && u18 , U19 && u19 , U20 && u20 , U21 && u21 , U22 && u22) const
    {
        return
            detail::bind_action_invoke<Action>(
                _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ) , std::forward<U8>( u8 ) , std::forward<U9>( u9 ) , std::forward<U10>( u10 ) , std::forward<U11>( u11 ) , std::forward<U12>( u12 ) , std::forward<U13>( u13 ) , std::forward<U14>( u14 ) , std::forward<U15>( u15 ) , std::forward<U16>( u16 ) , std::forward<U17>( u17 ) , std::forward<U18>( u18 ) , std::forward<U19>( u19 ) , std::forward<U20>( u20 ) , std::forward<U21>( u21 ) , std::forward<U22>( u22 ))
            );
    }
