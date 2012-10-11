// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2,
        typename TR0 , typename TR1 , typename TR2>
    inline HPX_STD_TUPLE<T0 , T1 , T2>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2)
    {
        f0.get(); f1.get(); f2.get();
    }
}}
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3,
        typename TR0 , typename TR1 , typename TR2 , typename TR3>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3)
    {
        f0.get(); f1.get(); f2.get(); f3.get();
    }
}}
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4,
        typename TR0 , typename TR1 , typename TR2 , typename TR3 , typename TR4>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3 , T4>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3 , lcos::future<T4, TR4> const& f4)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3 , lcos::future<void> const& f4)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get();
    }
}}
