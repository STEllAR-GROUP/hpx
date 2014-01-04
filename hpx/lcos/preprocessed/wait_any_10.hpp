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
    when_any(BOOST_FWD_REF(T0) f0, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ), ec);
    }
    template <typename T0>
    HPX_STD_TUPLE<typename util::decay<T0>::type>
    wait_any(BOOST_FWD_REF(T0) f0, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ), ec);
    }
    template <typename T0 , typename T1>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ), ec);
    }
    template <typename T0 , typename T1 , typename T2>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
