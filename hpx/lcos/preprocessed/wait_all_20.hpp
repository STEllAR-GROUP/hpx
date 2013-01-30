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
    wait_all(lcos::future<T> f0)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
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
    wait_all(lcos::future<T> f0 , lcos::future<T> f1)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
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
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
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
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
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
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(6);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(7);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(8);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(9);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(10);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(11);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9); lazy_values.push_back(f10);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(12);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9); lazy_values.push_back(f10); lazy_values.push_back(f11);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(13);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9); lazy_values.push_back(f10); lazy_values.push_back(f11); lazy_values.push_back(f12);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(14);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9); lazy_values.push_back(f10); lazy_values.push_back(f11); lazy_values.push_back(f12); lazy_values.push_back(f13);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(15);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9); lazy_values.push_back(f10); lazy_values.push_back(f11); lazy_values.push_back(f12); lazy_values.push_back(f13); lazy_values.push_back(f14);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(16);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9); lazy_values.push_back(f10); lazy_values.push_back(f11); lazy_values.push_back(f12); lazy_values.push_back(f13); lazy_values.push_back(f14); lazy_values.push_back(f15);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(17);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9); lazy_values.push_back(f10); lazy_values.push_back(f11); lazy_values.push_back(f12); lazy_values.push_back(f13); lazy_values.push_back(f14); lazy_values.push_back(f15); lazy_values.push_back(f16);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(18);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9); lazy_values.push_back(f10); lazy_values.push_back(f11); lazy_values.push_back(f12); lazy_values.push_back(f13); lazy_values.push_back(f14); lazy_values.push_back(f15); lazy_values.push_back(f16); lazy_values.push_back(f17);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17 , lcos::future<T> f18)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(19);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9); lazy_values.push_back(f10); lazy_values.push_back(f11); lazy_values.push_back(f12); lazy_values.push_back(f13); lazy_values.push_back(f14); lazy_values.push_back(f15); lazy_values.push_back(f16); lazy_values.push_back(f17); lazy_values.push_back(f18);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17 , lcos::future<T> f18)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17 , f18);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17 , lcos::future<T> f18 , lcos::future<T> f19)
    {
        typedef std::vector<lcos::future<T> > return_type;
        return_type lazy_values;
        lazy_values.reserve(20);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4); lazy_values.push_back(f5); lazy_values.push_back(f6); lazy_values.push_back(f7); lazy_values.push_back(f8); lazy_values.push_back(f9); lazy_values.push_back(f10); lazy_values.push_back(f11); lazy_values.push_back(f12); lazy_values.push_back(f13); lazy_values.push_back(f14); lazy_values.push_back(f15); lazy_values.push_back(f16); lazy_values.push_back(f17); lazy_values.push_back(f18); lazy_values.push_back(f19);
        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17 , lcos::future<T> f18 , lcos::future<T> f19)
    {
        typedef std::vector<lcos::future<T> > result_type;
        lcos::future<result_type> f = when_all(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17 , f18 , f19);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
