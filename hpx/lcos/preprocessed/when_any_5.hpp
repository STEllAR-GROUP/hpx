// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace lcos
{
    
    template <typename T0>
    lcos::future<when_any_result<
        HPX_STD_TUPLE<typename util::decay<T0>::type> > >
    when_any(T0 && f0, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0));
        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values));
        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1>
    lcos::future<when_any_result<
        HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type> > >
    when_any(T0 && f0 , T1 && f1, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1));
        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values));
        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2>
    lcos::future<when_any_result<
        HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2));
        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values));
        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    lcos::future<when_any_result<
        HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3));
        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values));
        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    lcos::future<when_any_result<
        HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4));
        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values));
        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    lcos::future<when_any_result<
        HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5));
        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values));
        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    lcos::future<when_any_result<
        HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6));
        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values));
        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    lcos::future<when_any_result<
        HPX_STD_TUPLE<typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
            result_type;
        result_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7));
        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values));
        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
