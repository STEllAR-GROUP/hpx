// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename R0>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> > >
    when_all(lcos::future<R0> f0,
        error_code& ec = throws)
    {
        return when_n(1, f0, ec);
    }
    template <typename R0>
    HPX_STD_TUPLE<lcos::future<R0> >
    wait_all(lcos::future<R0> f0,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> >
            result_type;
        lcos::future<result_type> f = when_all(
            f0, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_all",
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> > >
    when_all(lcos::future<R0> f0 , lcos::future<R1> f1,
        error_code& ec = throws)
    {
        return when_n(2, f0 , f1, ec);
    }
    template <typename R0 , typename R1>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> >
    wait_all(lcos::future<R0> f0 , lcos::future<R1> f1,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> >
            result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_all",
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> > >
    when_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2,
        error_code& ec = throws)
    {
        return when_n(3, f0 , f1 , f2, ec);
    }
    template <typename R0 , typename R1 , typename R2>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> >
    wait_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> >
            result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_all",
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2 , typename R3>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> > >
    when_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3,
        error_code& ec = throws)
    {
        return when_n(4, f0 , f1 , f2 , f3, ec);
    }
    template <typename R0 , typename R1 , typename R2 , typename R3>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> >
    wait_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> >
            result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_all",
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> > >
    when_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4,
        error_code& ec = throws)
    {
        return when_n(5, f0 , f1 , f2 , f3 , f4, ec);
    }
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> >
    wait_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> >
            result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_all",
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> > >
    when_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5,
        error_code& ec = throws)
    {
        return when_n(6, f0 , f1 , f2 , f3 , f4 , f5, ec);
    }
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> >
    wait_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> >
            result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_all",
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5 , typename R6>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> > >
    when_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5 , lcos::future<R6> f6,
        error_code& ec = throws)
    {
        return when_n(7, f0 , f1 , f2 , f3 , f4 , f5 , f6, ec);
    }
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5 , typename R6>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> >
    wait_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5 , lcos::future<R6> f6,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> >
            result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_all",
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5 , typename R6 , typename R7>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> , lcos::future<R7> > >
    when_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5 , lcos::future<R6> f6 , lcos::future<R7> f7,
        error_code& ec = throws)
    {
        return when_n(8, f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7, ec);
    }
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5 , typename R6 , typename R7>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> , lcos::future<R7> >
    wait_all(lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5 , lcos::future<R6> f6 , lcos::future<R7> f7,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> , lcos::future<R7> >
            result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_all",
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
