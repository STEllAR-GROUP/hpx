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
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ), ec);
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ), ec);
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ), ec);
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ), ec);
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ), ec);
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ), ec);
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ), ec);
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ), ec);
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20 , BOOST_FWD_REF(T21) f21, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ) , boost::forward<T21>( f21 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20 , BOOST_FWD_REF(T21) f21, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ) , boost::forward<T21>( f21 ), ec);
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21 , typename T22>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type , typename util::decay<T22>::type> >
    when_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20 , BOOST_FWD_REF(T21) f21 , BOOST_FWD_REF(T22) f22, error_code& ec = throws)
    {
        return when_n(1, boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ) , boost::forward<T21>( f21 ) , boost::forward<T22>( f22 ), ec);
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21 , typename T22>
    HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type , typename util::decay<T22>::type>
    wait_any(BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17 , BOOST_FWD_REF(T18) f18 , BOOST_FWD_REF(T19) f19 , BOOST_FWD_REF(T20) f20 , BOOST_FWD_REF(T21) f21 , BOOST_FWD_REF(T22) f22, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type , typename util::decay<T22>::type> result_type;
        lcos::unique_future<result_type> f =
            when_any(boost::forward<T0>( f0 ) , boost::forward<T1>( f1 ) , boost::forward<T2>( f2 ) , boost::forward<T3>( f3 ) , boost::forward<T4>( f4 ) , boost::forward<T5>( f5 ) , boost::forward<T6>( f6 ) , boost::forward<T7>( f7 ) , boost::forward<T8>( f8 ) , boost::forward<T9>( f9 ) , boost::forward<T10>( f10 ) , boost::forward<T11>( f11 ) , boost::forward<T12>( f12 ) , boost::forward<T13>( f13 ) , boost::forward<T14>( f14 ) , boost::forward<T15>( f15 ) , boost::forward<T16>( f16 ) , boost::forward<T17>( f17 ) , boost::forward<T18>( f18 ) , boost::forward<T19>( f19 ) , boost::forward<T20>( f20 ) , boost::forward<T21>( f21 ) , boost::forward<T22>( f22 ), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
