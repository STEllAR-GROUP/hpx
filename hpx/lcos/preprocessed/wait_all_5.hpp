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
