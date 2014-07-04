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
        >
    >::type
    wait_each(T0 && f0, F && func)
    {
        lcos::when_each(std::forward<T0>( f0 ),
            std::forward<F>(func)).wait();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >
    >::type
    wait_each(T0 && f0 , T1 && f1, F && func)
    {
        lcos::when_each(std::forward<T0>( f0 ) , std::forward<T1>( f1 ),
            std::forward<F>(func)).wait();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >
    >::type
    wait_each(T0 && f0 , T1 && f1 , T2 && f2, F && func)
    {
        lcos::when_each(std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ),
            std::forward<F>(func)).wait();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >
    >::type
    wait_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3, F && func)
    {
        lcos::when_each(std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ),
            std::forward<F>(func)).wait();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >
    >::type
    wait_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4, F && func)
    {
        lcos::when_each(std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ),
            std::forward<F>(func)).wait();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >
    >::type
    wait_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5, F && func)
    {
        lcos::when_each(std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ),
            std::forward<F>(func)).wait();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >
    >::type
    wait_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6, F && func)
    {
        lcos::when_each(std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ),
            std::forward<F>(func)).wait();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7, typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >
    >::type
    wait_each(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7, F && func)
    {
        lcos::when_each(std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ) , std::forward<T7>( f7 ),
            std::forward<F>(func)).wait();
    }
}}
