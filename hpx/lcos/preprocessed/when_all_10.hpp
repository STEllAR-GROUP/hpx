// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace lcos
{
    
    template <typename T0>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type> >
    when_all(T0 && f0, error_code& ec = throws)
    {
        return lcos::when_some(1, std::forward<T0>( f0 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type> >
    when_all(T0 && f0 , T1 && f1, error_code& ec = throws)
    {
        return lcos::when_some(2, std::forward<T0>( f0 ) , std::forward<T1>( f1 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2, error_code& ec = throws)
    {
        return lcos::when_some(3, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3, error_code& ec = throws)
    {
        return lcos::when_some(4, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4, error_code& ec = throws)
    {
        return lcos::when_some(5, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5, error_code& ec = throws)
    {
        return lcos::when_some(6, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6, error_code& ec = throws)
    {
        return lcos::when_some(7, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7, error_code& ec = throws)
    {
        return lcos::when_some(8, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ) , std::forward<T7>( f7 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8, error_code& ec = throws)
    {
        return lcos::when_some(9, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ) , std::forward<T7>( f7 ) , std::forward<T8>( f8 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9, error_code& ec = throws)
    {
        return lcos::when_some(10, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ) , std::forward<T7>( f7 ) , std::forward<T8>( f8 ) , std::forward<T9>( f9 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10, error_code& ec = throws)
    {
        return lcos::when_some(11, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ) , std::forward<T7>( f7 ) , std::forward<T8>( f8 ) , std::forward<T9>( f9 ) , std::forward<T10>( f10 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11, error_code& ec = throws)
    {
        return lcos::when_some(12, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ) , std::forward<T7>( f7 ) , std::forward<T8>( f8 ) , std::forward<T9>( f9 ) , std::forward<T10>( f10 ) , std::forward<T11>( f11 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    lcos::future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12, error_code& ec = throws)
    {
        return lcos::when_some(13, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ) , std::forward<T7>( f7 ) , std::forward<T8>( f8 ) , std::forward<T9>( f9 ) , std::forward<T10>( f10 ) , std::forward<T11>( f11 ) , std::forward<T12>( f12 ), ec);
    }
}}
