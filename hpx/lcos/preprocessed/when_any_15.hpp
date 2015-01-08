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
        util::tuple<typename traits::acquire_future<T0>::type> > >
    when_any(T0 && f0, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0));
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
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type> > >
    when_any(T0 && f0 , T1 && f1, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1));
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
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2));
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
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3));
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
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4));
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
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5));
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
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6));
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
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7));
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    lcos::future<when_any_result<
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8));
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    lcos::future<when_any_result<
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9));
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    lcos::future<when_any_result<
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10));
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    lcos::future<when_any_result<
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11));
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    lcos::future<when_any_result<
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12));
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    lcos::future<when_any_result<
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12) , traits::acquire_future_disp()(f13));
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    lcos::future<when_any_result<
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12) , traits::acquire_future_disp()(f13) , traits::acquire_future_disp()(f14));
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    lcos::future<when_any_result<
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12) , traits::acquire_future_disp()(f13) , traits::acquire_future_disp()(f14) , traits::acquire_future_disp()(f15));
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    lcos::future<when_any_result<
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type , typename traits::acquire_future<T16>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type , typename traits::acquire_future<T16>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12) , traits::acquire_future_disp()(f13) , traits::acquire_future_disp()(f14) , traits::acquire_future_disp()(f15) , traits::acquire_future_disp()(f16));
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
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    lcos::future<when_any_result<
        util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type , typename traits::acquire_future<T16>::type , typename traits::acquire_future<T17>::type> > >
    when_any(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17, error_code& ec = throws)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type , typename traits::acquire_future<T16>::type , typename traits::acquire_future<T17>::type>
            result_type;
        result_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12) , traits::acquire_future_disp()(f13) , traits::acquire_future_disp()(f14) , traits::acquire_future_disp()(f15) , traits::acquire_future_disp()(f16) , traits::acquire_future_disp()(f17));
        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values));
        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
