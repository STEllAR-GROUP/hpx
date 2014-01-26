// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util
{
    
    template <typename T0 , typename T1>
    typename boost::lazy_enable_if_c<
        traits::is_future_tuple<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type
        > >::value
      , detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type
        > >
    >::type unwrapped(T0 && f0 , T1 && f1)
    {
        typedef detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type
        > > unwrap_impl_t;
        return unwrap_impl_t::call(util::forward_as_tuple(
            std::forward<T0>( f0 ) , std::forward<T1>( f1 )));
    }
}}
namespace hpx { namespace util
{
    
    template <typename T0 , typename T1 , typename T2>
    typename boost::lazy_enable_if_c<
        traits::is_future_tuple<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type
        > >::value
      , detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type
        > >
    >::type unwrapped(T0 && f0 , T1 && f1 , T2 && f2)
    {
        typedef detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type
        > > unwrap_impl_t;
        return unwrap_impl_t::call(util::forward_as_tuple(
            std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 )));
    }
}}
namespace hpx { namespace util
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    typename boost::lazy_enable_if_c<
        traits::is_future_tuple<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type
        > >::value
      , detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type
        > >
    >::type unwrapped(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3)
    {
        typedef detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type
        > > unwrap_impl_t;
        return unwrap_impl_t::call(util::forward_as_tuple(
            std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 )));
    }
}}
namespace hpx { namespace util
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    typename boost::lazy_enable_if_c<
        traits::is_future_tuple<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type
        > >::value
      , detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type
        > >
    >::type unwrapped(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4)
    {
        typedef detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type
        > > unwrap_impl_t;
        return unwrap_impl_t::call(util::forward_as_tuple(
            std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 )));
    }
}}
namespace hpx { namespace util
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    typename boost::lazy_enable_if_c<
        traits::is_future_tuple<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type
        > >::value
      , detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type
        > >
    >::type unwrapped(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5)
    {
        typedef detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type
        > > unwrap_impl_t;
        return unwrap_impl_t::call(util::forward_as_tuple(
            std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 )));
    }
}}
namespace hpx { namespace util
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    typename boost::lazy_enable_if_c<
        traits::is_future_tuple<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type
        > >::value
      , detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type
        > >
    >::type unwrapped(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6)
    {
        typedef detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type
        > > unwrap_impl_t;
        return unwrap_impl_t::call(util::forward_as_tuple(
            std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 )));
    }
}}
namespace hpx { namespace util
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    typename boost::lazy_enable_if_c<
        traits::is_future_tuple<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type
        > >::value
      , detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type
        > >
    >::type unwrapped(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7)
    {
        typedef detail::unwrap_impl<util::tuple<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type
        > > unwrap_impl_t;
        return unwrap_impl_t::call(util::forward_as_tuple(
            std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ) , std::forward<T7>( f7 )));
    }
}}
