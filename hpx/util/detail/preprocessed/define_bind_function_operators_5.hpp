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
    operator()(U0 && u0)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ))
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
    operator()(U0 && u0) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ))
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
    operator()(U0 && u0 , U1 && u1)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ))
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
    operator()(U0 && u0 , U1 && u1) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7)
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ))
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
    operator()(U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7) const
    {
        return
            detail::bind_invoke(
                _f, _bound_args
              , util::forward_as_tuple(std::forward<U0>( u0 ) , std::forward<U1>( u1 ) , std::forward<U2>( u2 ) , std::forward<U3>( u3 ) , std::forward<U4>( u4 ) , std::forward<U5>( u5 ) , std::forward<U6>( u6 ) , std::forward<U7>( u7 ))
            );
    }
