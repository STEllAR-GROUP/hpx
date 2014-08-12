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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11));
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
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 13);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13, F && func)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 14);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14, F && func)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 15);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15, F && func)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14) , detail::when_acquire_future<T15>()(f15));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 16);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16, F && func)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14) , detail::when_acquire_future<T15>()(f15) , detail::when_acquire_future<T16>()(f16));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 17);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17, F && func)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14) , detail::when_acquire_future<T15>()(f15) , detail::when_acquire_future<T16>()(f16) , detail::when_acquire_future<T17>()(f17));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 18);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17 , T18 && f18, F && func)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14) , detail::when_acquire_future<T15>()(f15) , detail::when_acquire_future<T16>()(f16) , detail::when_acquire_future<T17>()(f17) , detail::when_acquire_future<T18>()(f18));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 19);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17 , T18 && f18 , T19 && f19, F && func)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14) , detail::when_acquire_future<T15>()(f15) , detail::when_acquire_future<T16>()(f16) , detail::when_acquire_future<T17>()(f17) , detail::when_acquire_future<T18>()(f18) , detail::when_acquire_future<T19>()(f19));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 20);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17 , T18 && f18 , T19 && f19 , T20 && f20, F && func)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14) , detail::when_acquire_future<T15>()(f15) , detail::when_acquire_future<T16>()(f16) , detail::when_acquire_future<T17>()(f17) , detail::when_acquire_future<T18>()(f18) , detail::when_acquire_future<T19>()(f19) , detail::when_acquire_future<T20>()(f20));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 21);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17 , T18 && f18 , T19 && f19 , T20 && f20 , T21 && f21, F && func)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14) , detail::when_acquire_future<T15>()(f15) , detail::when_acquire_future<T16>()(f16) , detail::when_acquire_future<T17>()(f17) , detail::when_acquire_future<T18>()(f18) , detail::when_acquire_future<T19>()(f19) , detail::when_acquire_future<T20>()(f20) , detail::when_acquire_future<T21>()(f21));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 22);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21 , typename T22, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17 , T18 && f18 , T19 && f19 , T20 && f20 , T21 && f21 , T22 && f22, F && func)
    {
        typedef HPX_STD_TUPLE<
            typename util::decay<T0>::type , typename util::decay<T1>::type , typename util::decay<T2>::type , typename util::decay<T3>::type , typename util::decay<T4>::type , typename util::decay<T5>::type , typename util::decay<T6>::type , typename util::decay<T7>::type , typename util::decay<T8>::type , typename util::decay<T9>::type , typename util::decay<T10>::type , typename util::decay<T11>::type , typename util::decay<T12>::type , typename util::decay<T13>::type , typename util::decay<T14>::type , typename util::decay<T15>::type , typename util::decay<T16>::type , typename util::decay<T17>::type , typename util::decay<T18>::type , typename util::decay<T19>::type , typename util::decay<T20>::type , typename util::decay<T21>::type , typename util::decay<T22>::type>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;
        argument_type lazy_values(detail::when_acquire_future<T0>()(f0) , detail::when_acquire_future<T1>()(f1) , detail::when_acquire_future<T2>()(f2) , detail::when_acquire_future<T3>()(f3) , detail::when_acquire_future<T4>()(f4) , detail::when_acquire_future<T5>()(f5) , detail::when_acquire_future<T6>()(f6) , detail::when_acquire_future<T7>()(f7) , detail::when_acquire_future<T8>()(f8) , detail::when_acquire_future<T9>()(f9) , detail::when_acquire_future<T10>()(f10) , detail::when_acquire_future<T11>()(f11) , detail::when_acquire_future<T12>()(f12) , detail::when_acquire_future<T13>()(f13) , detail::when_acquire_future<T14>()(f14) , detail::when_acquire_future<T15>()(f15) , detail::when_acquire_future<T16>()(f16) , detail::when_acquire_future<T17>()(f17) , detail::when_acquire_future<T18>()(f18) , detail::when_acquire_future<T19>()(f19) , detail::when_acquire_future<T20>()(f20) , detail::when_acquire_future<T21>()(f21) , detail::when_acquire_future<T22>()(f22));
        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), 23);
        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));
        p.apply();
        return p.get_future();
    }
}}
