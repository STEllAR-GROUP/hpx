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
    void wait_all(T0 && f0, error_code& ec = throws)
    {
        return lcos::wait_some(1, std::forward<T0>( f0 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1>
    void wait_all(T0 && f0 , T1 && f1, error_code& ec = throws)
    {
        return lcos::wait_some(2, std::forward<T0>( f0 ) , std::forward<T1>( f1 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2, error_code& ec = throws)
    {
        return lcos::wait_some(3, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3, error_code& ec = throws)
    {
        return lcos::wait_some(4, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4, error_code& ec = throws)
    {
        return lcos::wait_some(5, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5, error_code& ec = throws)
    {
        return lcos::wait_some(6, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6, error_code& ec = throws)
    {
        return lcos::wait_some(7, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ), ec);
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7, error_code& ec = throws)
    {
        return lcos::wait_some(8, std::forward<T0>( f0 ) , std::forward<T1>( f1 ) , std::forward<T2>( f2 ) , std::forward<T3>( f3 ) , std::forward<T4>( f4 ) , std::forward<T5>( f5 ) , std::forward<T6>( f6 ) , std::forward<T7>( f7 ), ec);
    }
}}
