// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0)
    {
        return when_any(f0).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1)
    {
        return when_any(f0 , f1).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2)
    {
        return when_any(f0 , f1 , f2).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3)
    {
        return when_any(f0 , f1 , f2 , f3).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4)
    {
        return when_any(f0 , f1 , f2 , f3 , f4).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17 , lcos::future<T> f18)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17 , f18)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17 , lcos::future<T> f18)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17 , f18).get();
    }
}
namespace hpx
{
    
    template <typename T>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17 , lcos::future<T> f18 , lcos::future<T> f19)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T> , lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17 , f18 , f19)));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17 , lcos::future<T> f18 , lcos::future<T> f19)
    {
        return when_any(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17 , f18 , f19).get();
    }
}
