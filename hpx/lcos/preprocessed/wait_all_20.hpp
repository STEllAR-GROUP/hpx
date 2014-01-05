// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename T0>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type> >
    when_all(BOOST_FWD_REF(T0) f0, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ), ec);
    }
    template <typename T0>
    void
    wait_all(BOOST_FWD_REF(T0) f0, error_code& ec = throws)
    {
        return wait_n(1, boost::forward<T0>( f0 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1, error_code& ec = throws)
    {
        return when_n(2, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ), ec);
    }
    template <typename T0 , typename T1>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1, error_code& ec = throws)
    {
        return wait_n(2, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2, error_code& ec = throws)
    {
        return when_n(3, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ), ec);
    }
    template <typename T0 , typename T1 , typename T2>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2, error_code& ec = throws)
    {
        return wait_n(3, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3, error_code& ec = throws)
    {
        return when_n(4, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3, error_code& ec = throws)
    {
        return wait_n(4, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4, error_code& ec = throws)
    {
        return when_n(5, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4, error_code& ec = throws)
    {
        return wait_n(5, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5, error_code& ec = throws)
    {
        return when_n(6, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5, error_code& ec = throws)
    {
        return wait_n(6, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6, error_code& ec = throws)
    {
        return when_n(7, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6, error_code& ec = throws)
    {
        return wait_n(7, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7, error_code& ec = throws)
    {
        return when_n(8, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7, error_code& ec = throws)
    {
        return wait_n(8, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8, error_code& ec = throws)
    {
        return when_n(9, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8, error_code& ec = throws)
    {
        return wait_n(9, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9, error_code& ec = throws)
    {
        return when_n(10, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9, error_code& ec = throws)
    {
        return wait_n(10, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10, error_code& ec = throws)
    {
        return when_n(11, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10, error_code& ec = throws)
    {
        return wait_n(11, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11, error_code& ec = throws)
    {
        return when_n(12, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11, error_code& ec = throws)
    {
        return wait_n(12, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12, error_code& ec = throws)
    {
        return when_n(13, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12, error_code& ec = throws)
    {
        return wait_n(13, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13, error_code& ec = throws)
    {
        return when_n(14, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13, error_code& ec = throws)
    {
        return wait_n(14, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14, error_code& ec = throws)
    {
        return when_n(15, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14, error_code& ec = throws)
    {
        return wait_n(15, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15, error_code& ec = throws)
    {
        return when_n(16, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15, error_code& ec = throws)
    {
        return wait_n(16, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16, error_code& ec = throws)
    {
        return when_n(17, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16, error_code& ec = throws)
    {
        return wait_n(17, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17, error_code& ec = throws)
    {
        return when_n(18, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17, error_code& ec = throws)
    {
        return wait_n(18, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18, error_code& ec = throws)
    {
        return when_n(19, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18, error_code& ec = throws)
    {
        return wait_n(19, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19, error_code& ec = throws)
    {
        return when_n(20, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19, error_code& ec = throws)
    {
        return wait_n(20, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20, error_code& ec = throws)
    {
        return when_n(21, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20, error_code& ec = throws)
    {
        return wait_n(21, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20 , BOOST_FWD_REF(T21) f21, error_code& ec = throws)
    {
        return when_n(22, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ) , boost::forward<T21>( f21 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20 , BOOST_FWD_REF(T21) f21, error_code& ec = throws)
    {
        return wait_n(22, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ) , boost::forward<T21>( f21 ), ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21 , typename T22>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type , typename util::decay<T22>::type> >
    when_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20 , BOOST_FWD_REF(T21) f21 , BOOST_FWD_REF(T22) f22, error_code& ec = throws)
    {
        return when_n(23, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ) , boost::forward<T21>( f21 ) , boost::forward<T22>( f22 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21 , typename T22>
    void
    wait_all(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20 , BOOST_FWD_REF(T21) f21 , BOOST_FWD_REF(T22) f22, error_code& ec = throws)
    {
        return wait_n(23, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ) , boost::forward<T21>( f21 ) , boost::forward<T22>( f22 ), ec);
    }
}
