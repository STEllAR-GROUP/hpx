// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        
    template <typename This, typename U0>
    struct result<This(U0)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type>
        >
    {};
    template <typename U0>
    BOOST_FORCEINLINE
    typename result<bound(U0)>::type
    operator()(BOOST_FWD_REF(U0) u0)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ))
            );
    }
    template <typename This, typename U0>
    struct result<This const(U0)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type>
        >
    {};
    template <typename U0>
    BOOST_FORCEINLINE
    typename result<bound const(U0)>::type
    operator()(BOOST_FWD_REF(U0) u0) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ))
            );
    }
    template <typename This, typename U0 , typename U1>
    struct result<This(U0 , U1)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type>
        >
    {};
    template <typename U0 , typename U1>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ))
            );
    }
    template <typename This, typename U0 , typename U1>
    struct result<This const(U0 , U1)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type>
        >
    {};
    template <typename U0 , typename U1>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2>
    struct result<This(U0 , U1 , U2)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2>
    struct result<This const(U0 , U1 , U2)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3>
    struct result<This(U0 , U1 , U2 , U3)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3>
    struct result<This const(U0 , U1 , U2 , U3)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4>
    struct result<This(U0 , U1 , U2 , U3 , U4)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4>
    struct result<This const(U0 , U1 , U2 , U3 , U4)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type , typename util::add_rvalue_reference<U13>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12 , BOOST_FWD_REF(U13) u13)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ) , boost::forward<U13>( u13 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type , typename util::add_rvalue_reference<U13>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12 , BOOST_FWD_REF(U13) u13) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ) , boost::forward<U13>( u13 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type , typename util::add_rvalue_reference<U13>::type , typename util::add_rvalue_reference<U14>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12 , BOOST_FWD_REF(U13) u13 , BOOST_FWD_REF(U14) u14)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ) , boost::forward<U13>( u13 ) , boost::forward<U14>( u14 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type , typename util::add_rvalue_reference<U13>::type , typename util::add_rvalue_reference<U14>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12 , BOOST_FWD_REF(U13) u13 , BOOST_FWD_REF(U14) u14) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ) , boost::forward<U13>( u13 ) , boost::forward<U14>( u14 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type , typename util::add_rvalue_reference<U13>::type , typename util::add_rvalue_reference<U14>::type , typename util::add_rvalue_reference<U15>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12 , BOOST_FWD_REF(U13) u13 , BOOST_FWD_REF(U14) u14 , BOOST_FWD_REF(U15) u15)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ) , boost::forward<U13>( u13 ) , boost::forward<U14>( u14 ) , boost::forward<U15>( u15 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type , typename util::add_rvalue_reference<U13>::type , typename util::add_rvalue_reference<U14>::type , typename util::add_rvalue_reference<U15>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12 , BOOST_FWD_REF(U13) u13 , BOOST_FWD_REF(U14) u14 , BOOST_FWD_REF(U15) u15) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ) , boost::forward<U13>( u13 ) , boost::forward<U14>( u14 ) , boost::forward<U15>( u15 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15 , U16)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type , typename util::add_rvalue_reference<U13>::type , typename util::add_rvalue_reference<U14>::type , typename util::add_rvalue_reference<U15>::type , typename util::add_rvalue_reference<U16>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15 , U16)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12 , BOOST_FWD_REF(U13) u13 , BOOST_FWD_REF(U14) u14 , BOOST_FWD_REF(U15) u15 , BOOST_FWD_REF(U16) u16)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ) , boost::forward<U13>( u13 ) , boost::forward<U14>( u14 ) , boost::forward<U15>( u15 ) , boost::forward<U16>( u16 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15 , U16)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type , typename util::add_rvalue_reference<U13>::type , typename util::add_rvalue_reference<U14>::type , typename util::add_rvalue_reference<U15>::type , typename util::add_rvalue_reference<U16>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15 , U16)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12 , BOOST_FWD_REF(U13) u13 , BOOST_FWD_REF(U14) u14 , BOOST_FWD_REF(U15) u15 , BOOST_FWD_REF(U16) u16) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ) , boost::forward<U13>( u13 ) , boost::forward<U14>( u14 ) , boost::forward<U15>( u15 ) , boost::forward<U16>( u16 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17>
    struct result<This(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15 , U16 , U17)>
      : bind_invoke_impl<
            F, BoundArgs
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type , typename util::add_rvalue_reference<U13>::type , typename util::add_rvalue_reference<U14>::type , typename util::add_rvalue_reference<U15>::type , typename util::add_rvalue_reference<U16>::type , typename util::add_rvalue_reference<U17>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17>
    BOOST_FORCEINLINE
    typename result<bound(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15 , U16 , U17)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12 , BOOST_FWD_REF(U13) u13 , BOOST_FWD_REF(U14) u14 , BOOST_FWD_REF(U15) u15 , BOOST_FWD_REF(U16) u16 , BOOST_FWD_REF(U17) u17)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ) , boost::forward<U13>( u13 ) , boost::forward<U14>( u14 ) , boost::forward<U15>( u15 ) , boost::forward<U16>( u16 ) , boost::forward<U17>( u17 ))
            );
    }
    template <typename This, typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17>
    struct result<This const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15 , U16 , U17)>
      : bind_invoke_impl<
            F const, BoundArgs const
          , util::tuple<typename util::add_rvalue_reference<U0>::type , typename util::add_rvalue_reference<U1>::type , typename util::add_rvalue_reference<U2>::type , typename util::add_rvalue_reference<U3>::type , typename util::add_rvalue_reference<U4>::type , typename util::add_rvalue_reference<U5>::type , typename util::add_rvalue_reference<U6>::type , typename util::add_rvalue_reference<U7>::type , typename util::add_rvalue_reference<U8>::type , typename util::add_rvalue_reference<U9>::type , typename util::add_rvalue_reference<U10>::type , typename util::add_rvalue_reference<U11>::type , typename util::add_rvalue_reference<U12>::type , typename util::add_rvalue_reference<U13>::type , typename util::add_rvalue_reference<U14>::type , typename util::add_rvalue_reference<U15>::type , typename util::add_rvalue_reference<U16>::type , typename util::add_rvalue_reference<U17>::type>
        >
    {};
    template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17>
    BOOST_FORCEINLINE
    typename result<bound const(U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15 , U16 , U17)>::type
    operator()(BOOST_FWD_REF(U0) u0 , BOOST_FWD_REF(U1) u1 , BOOST_FWD_REF(U2) u2 , BOOST_FWD_REF(U3) u3 , BOOST_FWD_REF(U4) u4 , BOOST_FWD_REF(U5) u5 , BOOST_FWD_REF(U6) u6 , BOOST_FWD_REF(U7) u7 , BOOST_FWD_REF(U8) u8 , BOOST_FWD_REF(U9) u9 , BOOST_FWD_REF(U10) u10 , BOOST_FWD_REF(U11) u11 , BOOST_FWD_REF(U12) u12 , BOOST_FWD_REF(U13) u13 , BOOST_FWD_REF(U14) u14 , BOOST_FWD_REF(U15) u15 , BOOST_FWD_REF(U16) u16 , BOOST_FWD_REF(U17) u17) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(boost::forward<U0>( u0 ) , boost::forward<U1>( u1 ) , boost::forward<U2>( u2 ) , boost::forward<U3>( u3 ) , boost::forward<U4>( u4 ) , boost::forward<U5>( u5 ) , boost::forward<U6>( u6 ) , boost::forward<U7>( u7 ) , boost::forward<U8>( u8 ) , boost::forward<U9>( u9 ) , boost::forward<U10>( u10 ) , boost::forward<U11>( u11 ) , boost::forward<U12>( u12 ) , boost::forward<U13>( u13 ) , boost::forward<U14>( u14 ) , boost::forward<U15>( u15 ) , boost::forward<U16>( u16 ) , boost::forward<U17>( u17 ))
            );
    }
