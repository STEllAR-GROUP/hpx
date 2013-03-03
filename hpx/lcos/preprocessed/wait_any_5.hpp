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
    wait_any (lcos::future<T> f0,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0);
        if (!f.valid()) {
            { if (&ec == &hpx::throws) { HPX_THROW_EXCEPTION( uninitialized_value, "lcos::wait_any", "lcos::when_any didn't return a valid future"); } else { ec = make_error_code(static_cast<hpx::error>( uninitialized_value), "lcos::when_any didn't return a valid future", "lcos::wait_any", "D:/Devel\\hpx\\hpx\\lcos\\wait_any.hpp", 491, (ec.category() == hpx::get_lightweight_hpx_category()) ? hpx::lightweight : hpx::plain); } };
            return result_type();
        }
        return f.get(ec);
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
    wait_any (lcos::future<T> f0 , lcos::future<T> f1,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1);
        if (!f.valid()) {
            { if (&ec == &hpx::throws) { HPX_THROW_EXCEPTION( uninitialized_value, "lcos::wait_any", "lcos::when_any didn't return a valid future"); } else { ec = make_error_code(static_cast<hpx::error>( uninitialized_value), "lcos::when_any didn't return a valid future", "lcos::wait_any", "D:/Devel\\hpx\\hpx\\lcos\\wait_any.hpp", 491, (ec.category() == hpx::get_lightweight_hpx_category()) ? hpx::lightweight : hpx::plain); } };
            return result_type();
        }
        return f.get(ec);
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
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2);
        if (!f.valid()) {
            { if (&ec == &hpx::throws) { HPX_THROW_EXCEPTION( uninitialized_value, "lcos::wait_any", "lcos::when_any didn't return a valid future"); } else { ec = make_error_code(static_cast<hpx::error>( uninitialized_value), "lcos::when_any didn't return a valid future", "lcos::wait_any", "D:/Devel\\hpx\\hpx\\lcos\\wait_any.hpp", 491, (ec.category() == hpx::get_lightweight_hpx_category()) ? hpx::lightweight : hpx::plain); } };
            return result_type();
        }
        return f.get(ec);
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
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3);
        if (!f.valid()) {
            { if (&ec == &hpx::throws) { HPX_THROW_EXCEPTION( uninitialized_value, "lcos::wait_any", "lcos::when_any didn't return a valid future"); } else { ec = make_error_code(static_cast<hpx::error>( uninitialized_value), "lcos::when_any didn't return a valid future", "lcos::wait_any", "D:/Devel\\hpx\\hpx\\lcos\\wait_any.hpp", 491, (ec.category() == hpx::get_lightweight_hpx_category()) ? hpx::lightweight : hpx::plain); } };
            return result_type();
        }
        return f.get(ec);
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
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4);
        if (!f.valid()) {
            { if (&ec == &hpx::throws) { HPX_THROW_EXCEPTION( uninitialized_value, "lcos::wait_any", "lcos::when_any didn't return a valid future"); } else { ec = make_error_code(static_cast<hpx::error>( uninitialized_value), "lcos::when_any didn't return a valid future", "lcos::wait_any", "D:/Devel\\hpx\\hpx\\lcos\\wait_any.hpp", 491, (ec.category() == hpx::get_lightweight_hpx_category()) ? hpx::lightweight : hpx::plain); } };
            return result_type();
        }
        return f.get(ec);
    }
}
