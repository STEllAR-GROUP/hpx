// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace lcos
{
    
    template <typename T0, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 1);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 2);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 3);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 4);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 5);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 6);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 7);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 8);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 9);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 10);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 11);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 12);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12, F && func)
    {
        typedef util::tuple<
            typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 13);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
