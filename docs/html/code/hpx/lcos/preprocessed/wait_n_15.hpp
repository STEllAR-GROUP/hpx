// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename T0>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0));
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
    template <typename T0>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0));
        if (n == 0)
        {
            return;
        }
        if (n > 1)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1));
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
    template <typename T0 , typename T1>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1));
        if (n == 0)
        {
            return;
        }
        if (n > 2)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2));
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
    template <typename T0 , typename T1 , typename T2>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2));
        if (n == 0)
        {
            return;
        }
        if (n > 3)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3));
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
    template <typename T0 , typename T1 , typename T2 , typename T3>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3));
        if (n == 0)
        {
            return;
        }
        if (n > 4)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4));
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4));
        if (n == 0)
        {
            return;
        }
        if (n > 5)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5));
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5));
        if (n == 0)
        {
            return;
        }
        if (n > 6)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6));
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6));
        if (n == 0)
        {
            return;
        }
        if (n > 7)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7));
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6) , lcos::detail::future_access::get_shared_state(f7));
        if (n == 0)
        {
            return;
        }
        if (n > 8)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8));
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 9)
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6) , lcos::detail::future_access::get_shared_state(f7) , lcos::detail::future_access::get_shared_state(f8));
        if (n == 0)
        {
            return;
        }
        if (n > 9)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9));
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 10)
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6) , lcos::detail::future_access::get_shared_state(f7) , lcos::detail::future_access::get_shared_state(f8) , lcos::detail::future_access::get_shared_state(f9));
        if (n == 0)
        {
            return;
        }
        if (n > 10)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10));
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 11)
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6) , lcos::detail::future_access::get_shared_state(f7) , lcos::detail::future_access::get_shared_state(f8) , lcos::detail::future_access::get_shared_state(f9) , lcos::detail::future_access::get_shared_state(f10));
        if (n == 0)
        {
            return;
        }
        if (n > 11)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11));
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 12)
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6) , lcos::detail::future_access::get_shared_state(f7) , lcos::detail::future_access::get_shared_state(f8) , lcos::detail::future_access::get_shared_state(f9) , lcos::detail::future_access::get_shared_state(f10) , lcos::detail::future_access::get_shared_state(f11));
        if (n == 0)
        {
            return;
        }
        if (n > 12)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12));
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 13)
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6) , lcos::detail::future_access::get_shared_state(f7) , lcos::detail::future_access::get_shared_state(f8) , lcos::detail::future_access::get_shared_state(f9) , lcos::detail::future_access::get_shared_state(f10) , lcos::detail::future_access::get_shared_state(f11) , lcos::detail::future_access::get_shared_state(f12));
        if (n == 0)
        {
            return;
        }
        if (n > 13)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13));
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 14)
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6) , lcos::detail::future_access::get_shared_state(f7) , lcos::detail::future_access::get_shared_state(f8) , lcos::detail::future_access::get_shared_state(f9) , lcos::detail::future_access::get_shared_state(f10) , lcos::detail::future_access::get_shared_state(f11) , lcos::detail::future_access::get_shared_state(f12) , lcos::detail::future_access::get_shared_state(f13));
        if (n == 0)
        {
            return;
        }
        if (n > 14)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14));
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 15)
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6) , lcos::detail::future_access::get_shared_state(f7) , lcos::detail::future_access::get_shared_state(f8) , lcos::detail::future_access::get_shared_state(f9) , lcos::detail::future_access::get_shared_state(f10) , lcos::detail::future_access::get_shared_state(f11) , lcos::detail::future_access::get_shared_state(f12) , lcos::detail::future_access::get_shared_state(f13) , lcos::detail::future_access::get_shared_state(f14));
        if (n == 0)
        {
            return;
        }
        if (n > 15)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14) , detail::when_acquire_future<T15>()(f15));
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 16)
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type , typename lcos::detail::shared_state_ptr_for<T15>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6) , lcos::detail::future_access::get_shared_state(f7) , lcos::detail::future_access::get_shared_state(f8) , lcos::detail::future_access::get_shared_state(f9) , lcos::detail::future_access::get_shared_state(f10) , lcos::detail::future_access::get_shared_state(f11) , lcos::detail::future_access::get_shared_state(f12) , lcos::detail::future_access::get_shared_state(f13) , lcos::detail::future_access::get_shared_state(f14) , lcos::detail::future_access::get_shared_state(f15));
        if (n == 0)
        {
            return;
        }
        if (n > 16)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14) , detail::when_acquire_future<T15>()(f15) , detail::when_acquire_future<T16>()(f16));
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 17)
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type , typename lcos::detail::shared_state_ptr_for<T15>::type , typename lcos::detail::shared_state_ptr_for<T16>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6) , lcos::detail::future_access::get_shared_state(f7) , lcos::detail::future_access::get_shared_state(f8) , lcos::detail::future_access::get_shared_state(f9) , lcos::detail::future_access::get_shared_state(f10) , lcos::detail::future_access::get_shared_state(f11) , lcos::detail::future_access::get_shared_state(f12) , lcos::detail::future_access::get_shared_state(f13) , lcos::detail::future_access::get_shared_state(f14) , lcos::detail::future_access::get_shared_state(f15) , lcos::detail::future_access::get_shared_state(f16));
        if (n == 0)
        {
            return;
        }
        if (n > 17)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
namespace hpx
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    lcos::unique_future<HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type> >
    when_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14) , detail::when_acquire_future<T15>()(f15) , detail::when_acquire_future<T16>()(f16) , detail::when_acquire_future<T17>()(f17));
        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }
        if (n > 18)
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
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    void
    wait_n(std::size_t n, BOOST_FWD_REF(T0) f0 , BOOST_FWD_REF(T1) f1 , BOOST_FWD_REF(T2) f2 , BOOST_FWD_REF(T3) f3 , BOOST_FWD_REF(T4) f4 , BOOST_FWD_REF(T5) f5 , BOOST_FWD_REF(T6) f6 , BOOST_FWD_REF(T7) f7 , BOOST_FWD_REF(T8) f8 , BOOST_FWD_REF(T9) f9 , BOOST_FWD_REF(T10) f10 , BOOST_FWD_REF(T11) f11 , BOOST_FWD_REF(T12) f12 , BOOST_FWD_REF(T13) f13 , BOOST_FWD_REF(T14) f14 , BOOST_FWD_REF(T15) f15 , BOOST_FWD_REF(T16) f16 , BOOST_FWD_REF(T17) f17,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type , typename lcos::detail::shared_state_ptr_for<T15>::type , typename lcos::detail::shared_state_ptr_for<T16>::type , typename lcos::detail::shared_state_ptr_for<T17>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::future_access::get_shared_state(f0) , lcos::detail::future_access::get_shared_state(f1) , lcos::detail::future_access::get_shared_state(f2) , lcos::detail::future_access::get_shared_state(f3) , lcos::detail::future_access::get_shared_state(f4) , lcos::detail::future_access::get_shared_state(f5) , lcos::detail::future_access::get_shared_state(f6) , lcos::detail::future_access::get_shared_state(f7) , lcos::detail::future_access::get_shared_state(f8) , lcos::detail::future_access::get_shared_state(f9) , lcos::detail::future_access::get_shared_state(f10) , lcos::detail::future_access::get_shared_state(f11) , lcos::detail::future_access::get_shared_state(f12) , lcos::detail::future_access::get_shared_state(f13) , lcos::detail::future_access::get_shared_state(f14) , lcos::detail::future_access::get_shared_state(f15) , lcos::detail::future_access::get_shared_state(f16) , lcos::detail::future_access::get_shared_state(f17));
        if (n == 0)
        {
            return;
        }
        if (n > 18)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_n",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_n<result_type> > f =
            boost::make_shared<detail::wait_n<result_type> >(
                boost::move(lazy_values_), n);
        return (*f.get())();
    }
}
