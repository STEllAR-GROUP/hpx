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
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T> > >
    when_any (lcos::future<T> f0)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > return_type;
        typedef boost::fusion::tuple<
            lcos::future<T>
        > argument_type;
        lcos::local::futures_factory<return_type()> p((
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0))));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_any", 
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get();
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
        lcos::local::futures_factory<return_type()> p((
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1))));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_any", 
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get();
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
        lcos::local::futures_factory<return_type()> p((
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2))));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_any", 
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get();
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
        lcos::local::futures_factory<return_type()> p((
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3))));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_any", 
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get();
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
        lcos::local::futures_factory<return_type()> p((
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4))));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_any", 
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get();
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
        lcos::local::futures_factory<return_type()> p((
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5))));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_any", 
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get();
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
        lcos::local::futures_factory<return_type()> p((
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6))));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_any", 
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get();
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
        lcos::local::futures_factory<return_type()> p((
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7))));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_any", 
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get();
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
        lcos::local::futures_factory<return_type()> p((
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8))));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_any", 
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get();
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
        lcos::local::futures_factory<return_type()> p((
            detail::when_any_tuple<argument_type, T>(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9))));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_any", 
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get();
    }
}
