// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, lcos::future<T> f0)
    {
        std::vector<lcos::future<T> > lazy_values;
        lazy_values.reserve(1);
        lazy_values.push_back(f0);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, lcos::future<T> f0,
        error_code& ec = throws)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > result_type;
        lcos::future<result_type> f = when_n(
            f0);
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
    
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1)
    {
        std::vector<lcos::future<T> > lazy_values;
        lazy_values.reserve(2);
        lazy_values.push_back(f0); lazy_values.push_back(f1);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1,
        error_code& ec = throws)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > result_type;
        lcos::future<result_type> f = when_n(
            f0 , f1);
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
    
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2)
    {
        std::vector<lcos::future<T> > lazy_values;
        lazy_values.reserve(3);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2,
        error_code& ec = throws)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > result_type;
        lcos::future<result_type> f = when_n(
            f0 , f1 , f2);
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
    
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3)
    {
        std::vector<lcos::future<T> > lazy_values;
        lazy_values.reserve(4);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3,
        error_code& ec = throws)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > result_type;
        lcos::future<result_type> f = when_n(
            f0 , f1 , f2 , f3);
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
    
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4)
    {
        std::vector<lcos::future<T> > lazy_values;
        lazy_values.reserve(5);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4,
        error_code& ec = throws)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > result_type;
        lcos::future<result_type> f = when_n(
            f0 , f1 , f2 , f3 , f4);
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
    
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5)
    {
        std::vector<lcos::future<T> > lazy_values;
        lazy_values.reserve(6);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5,
        error_code& ec = throws)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > result_type;
        lcos::future<result_type> f = when_n(
            f0 , f1 , f2 , f3 , f4 , f5);
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
    
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6)
    {
        std::vector<lcos::future<T> > lazy_values;
        lazy_values.reserve(7);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6,
        error_code& ec = throws)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > result_type;
        lcos::future<result_type> f = when_n(
            f0 , f1 , f2 , f3 , f4 , f5 , f6);
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
    
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7)
    {
        std::vector<lcos::future<T> > lazy_values;
        lazy_values.reserve(8);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7,
        error_code& ec = throws)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > result_type;
        lcos::future<result_type> f = when_n(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7);
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
    
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8)
    {
        std::vector<lcos::future<T> > lazy_values;
        lazy_values.reserve(9);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8,
        error_code& ec = throws)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > result_type;
        lcos::future<result_type> f = when_n(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8);
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
    
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9)
    {
        std::vector<lcos::future<T> > lazy_values;
        lazy_values.reserve(10);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9,
        error_code& ec = throws)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > result_type;
        lcos::future<result_type> f = when_n(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n", 
                "lcos::when_n didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
