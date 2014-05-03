// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace lcos
{
    
    template <typename F, typename T0>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0>
    >::result_type> invoke_when_ready(F&& f, T0 && v0)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1 , typename T2>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1 , T2>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1 , T2 && v2)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1 , T2 , T3>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1 , T2 , T3 , T4>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1 , T2 , T3 , T4 , T5>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 )));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename F, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    future<typename detail::invoke_when_ready<
        typename util::decay<F>::type
      , util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12>
    >::result_type> invoke_when_ready(F&& f, T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12)
    {
        return invoke_fused_when_ready(std::forward<F>(f),
            util::forward_as_tuple(std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 )));
    }
}}
