// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl<
            F, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 1
            >::type
        >
        {
            typedef
                typename boost::mpl::eval_if<
                    traits::is_callable<F, typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type>
                  , util::invoke_result_of<F(typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type)>
                  , boost::mpl::identity<cannot_be_called>
                >::type
                type;
            static BOOST_FORCEINLINE
            type call(
                F& f, BoundArgs& bound_args
              , BOOST_FWD_REF(UnboundArgs) unbound_args
            )
            {
                return util::invoke(f, detail::bind_eval<F>( util::get< 0>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ));
            }
        };
    }
    template <typename F, typename T0>
    typename boost::disable_if_c<
        traits::is_action<typename util::remove_reference<F>::type>::value
      , detail::bound<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type>
        >
    >::type
    bind(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0)
    {
        typedef
            detail::bound<
                typename util::decay<F>::type
              , util::tuple<typename util::decay<T0>::type>
            >
            result_type;
        return
            result_type(
                boost::forward<F>(f)
              , util::forward_as_tuple(boost::forward<T0>( t0 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl<
            F, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 2
            >::type
        >
        {
            typedef
                typename boost::mpl::eval_if<
                    traits::is_callable<F, typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type>
                  , util::invoke_result_of<F(typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type)>
                  , boost::mpl::identity<cannot_be_called>
                >::type
                type;
            static BOOST_FORCEINLINE
            type call(
                F& f, BoundArgs& bound_args
              , BOOST_FWD_REF(UnboundArgs) unbound_args
            )
            {
                return util::invoke(f, detail::bind_eval<F>( util::get< 0>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 1>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ));
            }
        };
    }
    template <typename F, typename T0 , typename T1>
    typename boost::disable_if_c<
        traits::is_action<typename util::remove_reference<F>::type>::value
      , detail::bound<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>
        >
    >::type
    bind(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1)
    {
        typedef
            detail::bound<
                typename util::decay<F>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type>
            >
            result_type;
        return
            result_type(
                boost::forward<F>(f)
              , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl<
            F, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 3
            >::type
        >
        {
            typedef
                typename boost::mpl::eval_if<
                    traits::is_callable<F, typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type>
                  , util::invoke_result_of<F(typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type)>
                  , boost::mpl::identity<cannot_be_called>
                >::type
                type;
            static BOOST_FORCEINLINE
            type call(
                F& f, BoundArgs& bound_args
              , BOOST_FWD_REF(UnboundArgs) unbound_args
            )
            {
                return util::invoke(f, detail::bind_eval<F>( util::get< 0>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 1>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 2>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ));
            }
        };
    }
    template <typename F, typename T0 , typename T1 , typename T2>
    typename boost::disable_if_c<
        traits::is_action<typename util::remove_reference<F>::type>::value
      , detail::bound<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
        >
    >::type
    bind(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2)
    {
        typedef
            detail::bound<
                typename util::decay<F>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
            >
            result_type;
        return
            result_type(
                boost::forward<F>(f)
              , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl<
            F, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 4
            >::type
        >
        {
            typedef
                typename boost::mpl::eval_if<
                    traits::is_callable<F, typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 3, BoundArgs>::type , UnboundArgs >::type>
                  , util::invoke_result_of<F(typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 3, BoundArgs>::type , UnboundArgs >::type)>
                  , boost::mpl::identity<cannot_be_called>
                >::type
                type;
            static BOOST_FORCEINLINE
            type call(
                F& f, BoundArgs& bound_args
              , BOOST_FWD_REF(UnboundArgs) unbound_args
            )
            {
                return util::invoke(f, detail::bind_eval<F>( util::get< 0>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 1>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 2>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 3>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ));
            }
        };
    }
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3>
    typename boost::disable_if_c<
        traits::is_action<typename util::remove_reference<F>::type>::value
      , detail::bound<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
        >
    >::type
    bind(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3)
    {
        typedef
            detail::bound<
                typename util::decay<F>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
            >
            result_type;
        return
            result_type(
                boost::forward<F>(f)
              , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl<
            F, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 5
            >::type
        >
        {
            typedef
                typename boost::mpl::eval_if<
                    traits::is_callable<F, typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 3, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 4, BoundArgs>::type , UnboundArgs >::type>
                  , util::invoke_result_of<F(typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 3, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 4, BoundArgs>::type , UnboundArgs >::type)>
                  , boost::mpl::identity<cannot_be_called>
                >::type
                type;
            static BOOST_FORCEINLINE
            type call(
                F& f, BoundArgs& bound_args
              , BOOST_FWD_REF(UnboundArgs) unbound_args
            )
            {
                return util::invoke(f, detail::bind_eval<F>( util::get< 0>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 1>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 2>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 3>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 4>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ));
            }
        };
    }
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    typename boost::disable_if_c<
        traits::is_action<typename util::remove_reference<F>::type>::value
      , detail::bound<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
        >
    >::type
    bind(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4)
    {
        typedef
            detail::bound<
                typename util::decay<F>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
            >
            result_type;
        return
            result_type(
                boost::forward<F>(f)
              , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl<
            F, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 6
            >::type
        >
        {
            typedef
                typename boost::mpl::eval_if<
                    traits::is_callable<F, typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 3, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 4, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 5, BoundArgs>::type , UnboundArgs >::type>
                  , util::invoke_result_of<F(typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 3, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 4, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 5, BoundArgs>::type , UnboundArgs >::type)>
                  , boost::mpl::identity<cannot_be_called>
                >::type
                type;
            static BOOST_FORCEINLINE
            type call(
                F& f, BoundArgs& bound_args
              , BOOST_FWD_REF(UnboundArgs) unbound_args
            )
            {
                return util::invoke(f, detail::bind_eval<F>( util::get< 0>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 1>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 2>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 3>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 4>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 5>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ));
            }
        };
    }
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    typename boost::disable_if_c<
        traits::is_action<typename util::remove_reference<F>::type>::value
      , detail::bound<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
        >
    >::type
    bind(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5)
    {
        typedef
            detail::bound<
                typename util::decay<F>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
            >
            result_type;
        return
            result_type(
                boost::forward<F>(f)
              , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ) , boost::forward<T5>( t5 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl<
            F, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 7
            >::type
        >
        {
            typedef
                typename boost::mpl::eval_if<
                    traits::is_callable<F, typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 3, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 4, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 5, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 6, BoundArgs>::type , UnboundArgs >::type>
                  , util::invoke_result_of<F(typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 3, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 4, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 5, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 6, BoundArgs>::type , UnboundArgs >::type)>
                  , boost::mpl::identity<cannot_be_called>
                >::type
                type;
            static BOOST_FORCEINLINE
            type call(
                F& f, BoundArgs& bound_args
              , BOOST_FWD_REF(UnboundArgs) unbound_args
            )
            {
                return util::invoke(f, detail::bind_eval<F>( util::get< 0>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 1>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 2>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 3>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 4>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 5>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 6>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ));
            }
        };
    }
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    typename boost::disable_if_c<
        traits::is_action<typename util::remove_reference<F>::type>::value
      , detail::bound<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
        >
    >::type
    bind(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6)
    {
        typedef
            detail::bound<
                typename util::decay<F>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
            >
            result_type;
        return
            result_type(
                boost::forward<F>(f)
              , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ) , boost::forward<T5>( t5 ) , boost::forward<T6>( t6 ))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename F, typename BoundArgs, typename UnboundArgs>
        struct bind_invoke_impl<
            F, BoundArgs, UnboundArgs
          , typename boost::enable_if_c<
                util::tuple_size<BoundArgs>::value == 8
            >::type
        >
        {
            typedef
                typename boost::mpl::eval_if<
                    traits::is_callable<F, typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 3, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 4, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 5, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 6, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 7, BoundArgs>::type , UnboundArgs >::type>
                  , util::invoke_result_of<F(typename detail::bind_eval_impl< F, typename util::tuple_element< 0, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 1, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 2, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 3, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 4, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 5, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 6, BoundArgs>::type , UnboundArgs >::type , typename detail::bind_eval_impl< F, typename util::tuple_element< 7, BoundArgs>::type , UnboundArgs >::type)>
                  , boost::mpl::identity<cannot_be_called>
                >::type
                type;
            static BOOST_FORCEINLINE
            type call(
                F& f, BoundArgs& bound_args
              , BOOST_FWD_REF(UnboundArgs) unbound_args
            )
            {
                return util::invoke(f, detail::bind_eval<F>( util::get< 0>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 1>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 2>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 3>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 4>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 5>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 6>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ) , detail::bind_eval<F>( util::get< 7>(bound_args) , boost::forward<UnboundArgs>(unbound_args) ));
            }
        };
    }
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    typename boost::disable_if_c<
        traits::is_action<typename util::remove_reference<F>::type>::value
      , detail::bound<
            typename util::decay<F>::type
          , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
        >
    >::type
    bind(BOOST_FWD_REF(F) f, BOOST_FWD_REF(T0) t0 , BOOST_FWD_REF(T1) t1 , BOOST_FWD_REF(T2) t2 , BOOST_FWD_REF(T3) t3 , BOOST_FWD_REF(T4) t4 , BOOST_FWD_REF(T5) t5 , BOOST_FWD_REF(T6) t6 , BOOST_FWD_REF(T7) t7)
    {
        typedef
            detail::bound<
                typename util::decay<F>::type
              , util::tuple<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
            >
            result_type;
        return
            result_type(
                boost::forward<F>(f)
              , util::forward_as_tuple(boost::forward<T0>( t0 ) , boost::forward<T1>( t1 ) , boost::forward<T2>( t2 ) , boost::forward<T3>( t3 ) , boost::forward<T4>( t4 ) , boost::forward<T5>( t5 ) , boost::forward<T6>( t6 ) , boost::forward<T7>( t7 ))
            );
    }
}}
