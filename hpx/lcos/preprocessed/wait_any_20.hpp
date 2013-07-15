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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
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
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
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
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
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
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
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
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
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
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17 , f18));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17 , lcos::future<T> f18,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17 , f18);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
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
        boost::shared_ptr<detail::when_any_tuple<argument_type, T> > f =
            boost::make_shared<detail::when_any_tuple<argument_type, T> >(
                argument_type(f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17 , f18 , f19));
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                util::bind(&detail::when_any_tuple<argument_type, T>::operator(), f));
        p.apply();
        return p.get_future();
    }
    template <typename T>
    HPX_STD_TUPLE<int, lcos::future<T> >
    wait_any (lcos::future<T> f0 , lcos::future<T> f1 , lcos::future<T> f2 , lcos::future<T> f3 , lcos::future<T> f4 , lcos::future<T> f5 , lcos::future<T> f6 , lcos::future<T> f7 , lcos::future<T> f8 , lcos::future<T> f9 , lcos::future<T> f10 , lcos::future<T> f11 , lcos::future<T> f12 , lcos::future<T> f13 , lcos::future<T> f14 , lcos::future<T> f15 , lcos::future<T> f16 , lcos::future<T> f17 , lcos::future<T> f18 , lcos::future<T> f19,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T> > result_type;
        lcos::future<result_type> f = when_any(
            f0 , f1 , f2 , f3 , f4 , f5 , f6 , f7 , f8 , f9 , f10 , f11 , f12 , f13 , f14 , f15 , f16 , f17 , f18 , f19);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }
        return f.get(ec);
    }
}
