// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename T, typename RT>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T, RT> > > >
    wait_n (std::size_t n, lcos::future<T, RT> f0)
    {
        std::vector<lcos::future<T, RT> > lazy_values;
        lazy_values.reserve(1);
        lazy_values.push_back(f0);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::wait_n<T, RT>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
}
namespace hpx
{
    
    template <typename T, typename RT>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T, RT> > > >
    wait_n (std::size_t n, lcos::future<T, RT> f0 , lcos::future<T, RT> f1)
    {
        std::vector<lcos::future<T, RT> > lazy_values;
        lazy_values.reserve(2);
        lazy_values.push_back(f0); lazy_values.push_back(f1);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::wait_n<T, RT>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
}
namespace hpx
{
    
    template <typename T, typename RT>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T, RT> > > >
    wait_n (std::size_t n, lcos::future<T, RT> f0 , lcos::future<T, RT> f1 , lcos::future<T, RT> f2)
    {
        std::vector<lcos::future<T, RT> > lazy_values;
        lazy_values.reserve(3);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::wait_n<T, RT>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
}
namespace hpx
{
    
    template <typename T, typename RT>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T, RT> > > >
    wait_n (std::size_t n, lcos::future<T, RT> f0 , lcos::future<T, RT> f1 , lcos::future<T, RT> f2 , lcos::future<T, RT> f3)
    {
        std::vector<lcos::future<T, RT> > lazy_values;
        lazy_values.reserve(4);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::wait_n<T, RT>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
}
namespace hpx
{
    
    template <typename T, typename RT>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T, RT> > > >
    wait_n (std::size_t n, lcos::future<T, RT> f0 , lcos::future<T, RT> f1 , lcos::future<T, RT> f2 , lcos::future<T, RT> f3 , lcos::future<T, RT> f4)
    {
        std::vector<lcos::future<T, RT> > lazy_values;
        lazy_values.reserve(5);
        lazy_values.push_back(f0); lazy_values.push_back(f1); lazy_values.push_back(f2); lazy_values.push_back(f3); lazy_values.push_back(f4);
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::wait_n<T, RT>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }
}
