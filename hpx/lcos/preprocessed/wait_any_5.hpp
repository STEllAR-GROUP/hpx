// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename T, typename RT>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
    when_any (lcos::future<T, RT> f0)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T, RT> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T, RT>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T, RT>(
                argument_type(f0)));
        p.apply();
        return p.get_future();
    }
    template <typename T, typename RT>
    HPX_STD_TUPLE<int, lcos::future<T, RT> >
    wait_any (lcos::future<T, RT> f0)
    {
        return when_any(f0).get();
    }
}
namespace hpx
{
    
    template <typename T, typename RT>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
    when_any (lcos::future<T, RT> f0 , lcos::future<T, RT> f1)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T, RT> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T, RT> , lcos::future<T, RT>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T, RT>(
                argument_type(f0 , f1)));
        p.apply();
        return p.get_future();
    }
    template <typename T, typename RT>
    HPX_STD_TUPLE<int, lcos::future<T, RT> >
    wait_any (lcos::future<T, RT> f0 , lcos::future<T, RT> f1)
    {
        return when_any(f0 , f1).get();
    }
}
namespace hpx
{
    
    template <typename T, typename RT>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
    when_any (lcos::future<T, RT> f0 , lcos::future<T, RT> f1 , lcos::future<T, RT> f2)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T, RT> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T, RT> , lcos::future<T, RT> , lcos::future<T, RT>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T, RT>(
                argument_type(f0 , f1 , f2)));
        p.apply();
        return p.get_future();
    }
    template <typename T, typename RT>
    HPX_STD_TUPLE<int, lcos::future<T, RT> >
    wait_any (lcos::future<T, RT> f0 , lcos::future<T, RT> f1 , lcos::future<T, RT> f2)
    {
        return when_any(f0 , f1 , f2).get();
    }
}
namespace hpx
{
    
    template <typename T, typename RT>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
    when_any (lcos::future<T, RT> f0 , lcos::future<T, RT> f1 , lcos::future<T, RT> f2 , lcos::future<T, RT> f3)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T, RT> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T, RT> , lcos::future<T, RT> , lcos::future<T, RT> , lcos::future<T, RT>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T, RT>(
                argument_type(f0 , f1 , f2 , f3)));
        p.apply();
        return p.get_future();
    }
    template <typename T, typename RT>
    HPX_STD_TUPLE<int, lcos::future<T, RT> >
    wait_any (lcos::future<T, RT> f0 , lcos::future<T, RT> f1 , lcos::future<T, RT> f2 , lcos::future<T, RT> f3)
    {
        return when_any(f0 , f1 , f2 , f3).get();
    }
}
namespace hpx
{
    
    template <typename T, typename RT>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
    when_any (lcos::future<T, RT> f0 , lcos::future<T, RT> f1 , lcos::future<T, RT> f2 , lcos::future<T, RT> f3 , lcos::future<T, RT> f4)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T, RT> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T, RT> , lcos::future<T, RT> , lcos::future<T, RT> , lcos::future<T, RT> , lcos::future<T, RT>
        > argument_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T, RT>(
                argument_type(f0 , f1 , f2 , f3 , f4)));
        p.apply();
        return p.get_future();
    }
    template <typename T, typename RT>
    HPX_STD_TUPLE<int, lcos::future<T, RT> >
    wait_any (lcos::future<T, RT> f0 , lcos::future<T, RT> f1 , lcos::future<T, RT> f2 , lcos::future<T, RT> f3 , lcos::future<T, RT> f4)
    {
        return when_any(f0 , f1 , f2 , f3 , f4).get();
    }
}
