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
