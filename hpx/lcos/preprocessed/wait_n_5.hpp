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
    when_n(std::size_t n, lcos::future<R0> f0,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> >
            result_type;
        result_type lazy_values(f0);
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 1)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_n",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }
        boost::shared_ptr<detail::when_n<result_type> > f =
            boost::make_shared<detail::when_n<result_type> >(
                boost::move(lazy_values), n);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_n<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename R0>
    HPX_STD_TUPLE<lcos::future<R0> >
    wait_n(std::size_t n, lcos::future<R0> f0,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> >
            result_type;
        lcos::future<result_type> f = when_n(n,
            f0, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n",
                "lcos::when_n didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> > >
    when_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> >
            result_type;
        result_type lazy_values(f0 , f1);
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 2)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_n",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }
        boost::shared_ptr<detail::when_n<result_type> > f =
            boost::make_shared<detail::when_n<result_type> >(
                boost::move(lazy_values), n);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_n<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename R0 , typename R1>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> >
    wait_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> >
            result_type;
        lcos::future<result_type> f = when_n(n,
            f0 , f1, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n",
                "lcos::when_n didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> > >
    when_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> >
            result_type;
        result_type lazy_values(f0 , f1 , f2);
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 3)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_n",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }
        boost::shared_ptr<detail::when_n<result_type> > f =
            boost::make_shared<detail::when_n<result_type> >(
                boost::move(lazy_values), n);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_n<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename R0 , typename R1 , typename R2>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> >
    wait_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> >
            result_type;
        lcos::future<result_type> f = when_n(n,
            f0 , f1 , f2, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n",
                "lcos::when_n didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2 , typename R3>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> > >
    when_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> >
            result_type;
        result_type lazy_values(f0 , f1 , f2 , f3);
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 4)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_n",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }
        boost::shared_ptr<detail::when_n<result_type> > f =
            boost::make_shared<detail::when_n<result_type> >(
                boost::move(lazy_values), n);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_n<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename R0 , typename R1 , typename R2 , typename R3>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> >
    wait_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> >
            result_type;
        lcos::future<result_type> f = when_n(n,
            f0 , f1 , f2 , f3, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n",
                "lcos::when_n didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> > >
    when_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> >
            result_type;
        result_type lazy_values(f0 , f1 , f2 , f3 , f4);
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 5)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_n",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }
        boost::shared_ptr<detail::when_n<result_type> > f =
            boost::make_shared<detail::when_n<result_type> >(
                boost::move(lazy_values), n);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_n<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> >
    wait_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> >
            result_type;
        lcos::future<result_type> f = when_n(n,
            f0 , f1 , f2 , f3 , f4, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n",
                "lcos::when_n didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> > >
    when_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> >
            result_type;
        result_type lazy_values(f0 , f1 , f2 , f3 , f4 , f5);
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 6)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_n",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }
        boost::shared_ptr<detail::when_n<result_type> > f =
            boost::make_shared<detail::when_n<result_type> >(
                boost::move(lazy_values), n);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_n<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> >
    wait_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> >
            result_type;
        lcos::future<result_type> f = when_n(n,
            f0 , f1 , f2 , f3 , f4 , f5, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n",
                "lcos::when_n didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5 , typename R6>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> > >
    when_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5 , lcos::future<R6> f6,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> >
            result_type;
        result_type lazy_values(f0 , f1 , f2 , f3 , f4 , f5 , f6);
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 7)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_n",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }
        boost::shared_ptr<detail::when_n<result_type> > f =
            boost::make_shared<detail::when_n<result_type> >(
                boost::move(lazy_values), n);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_n<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5 , typename R6>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> >
    wait_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5 , lcos::future<R6> f6,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> >
            result_type;
        lcos::future<result_type> f = when_n(n,
            f0 , f1 , f2 , f3 , f4 , f5 , f6, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n",
                "lcos::when_n didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5 , typename R6 , typename R7>
    lcos::future<HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> , lcos::future<R7> > >
    when_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5 , lcos::future<R6> f6 , lcos::future<R7> f7,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> , lcos::future<R7> >
            result_type;
        result_type lazy_values(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7);
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 8)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_n",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }
        boost::shared_ptr<detail::when_n<result_type> > f =
            boost::make_shared<detail::when_n<result_type> >(
                boost::move(lazy_values), n);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_n<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename R0 , typename R1 , typename R2 , typename R3 , typename R4 , typename R5 , typename R6 , typename R7>
    HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> , lcos::future<R7> >
    wait_n(std::size_t n, lcos::future<R0> f0 , lcos::future<R1> f1 , lcos::future<R2> f2 , lcos::future<R3> f3 , lcos::future<R4> f4 , lcos::future<R5> f5 , lcos::future<R6> f6 , lcos::future<R7> f7,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<lcos::future<R0> , lcos::future<R1> , lcos::future<R2> , lcos::future<R3> , lcos::future<R4> , lcos::future<R5> , lcos::future<R6> , lcos::future<R7> >
            result_type;
        lcos::future<result_type> f = when_n(n,
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n",
                "lcos::when_n didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
