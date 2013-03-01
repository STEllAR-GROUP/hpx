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
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(1);
        lazy_values.push_back(f0);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0);
        if (!f.valid()) {
            { if (&ec == &hpx::throws) { HPX_THROW_EXCEPTION( uninitialized_value, "lcos::wait_all", "lcos::when_all didn't return a valid future"); } else { ec = make_error_code(static_cast<hpx::error>( uninitialized_value), "lcos::when_all didn't return a valid future", "lcos::wait_all", "D:/Devel\\hpx\\hpx\\lcos\\wait_all.hpp", 435, (ec.category() == hpx::get_lightweight_hpx_category()) ? hpx::lightweight : hpx::plain); } };
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(2);
        lazy_values.push_back(f0); lazy_values.push_back(f1);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1);
        if (!f.valid()) {
            { if (&ec == &hpx::throws) { HPX_THROW_EXCEPTION( uninitialized_value, "lcos::wait_all", "lcos::when_all didn't return a valid future"); } else { ec = make_error_code(static_cast<hpx::error>( uninitialized_value), "lcos::when_all didn't return a valid future", "lcos::wait_all", "D:/Devel\\hpx\\hpx\\lcos\\wait_all.hpp", 435, (ec.category() == hpx::get_lightweight_hpx_category()) ? hpx::lightweight : hpx::plain); } };
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(3);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2);
        if (!f.valid()) {
            { if (&ec == &hpx::throws) { HPX_THROW_EXCEPTION( uninitialized_value, "lcos::wait_all", "lcos::when_all didn't return a valid future"); } else { ec = make_error_code(static_cast<hpx::error>( uninitialized_value), "lcos::when_all didn't return a valid future", "lcos::wait_all", "D:/Devel\\hpx\\hpx\\lcos\\wait_all.hpp", 435, (ec.category() == hpx::get_lightweight_hpx_category()) ? hpx::lightweight : hpx::plain); } };
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(4);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3);
        if (!f.valid()) {
            { if (&ec == &hpx::throws) { HPX_THROW_EXCEPTION( uninitialized_value, "lcos::wait_all", "lcos::when_all didn't return a valid future"); } else { ec = make_error_code(static_cast<hpx::error>( uninitialized_value), "lcos::when_all didn't return a valid future", "lcos::wait_all", "D:/Devel\\hpx\\hpx\\lcos\\wait_all.hpp", 435, (ec.category() == hpx::get_lightweight_hpx_category()) ? hpx::lightweight : hpx::plain); } };
            return result_type();
        }
        return f.get(ec);
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(5);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4);
        if (!f.valid()) {
            { if (&ec == &hpx::throws) { HPX_THROW_EXCEPTION( uninitialized_value, "lcos::wait_all", "lcos::when_all didn't return a valid future"); } else { ec = make_error_code(static_cast<hpx::error>( uninitialized_value), "lcos::when_all didn't return a valid future", "lcos::wait_all", "D:/Devel\\hpx\\hpx\\lcos\\wait_all.hpp", 435, (ec.category() == hpx::get_lightweight_hpx_category()) ? hpx::lightweight : hpx::plain); } };
            return result_type();
        }
        return f.get(ec);
    }
}
