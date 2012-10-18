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
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5,
        typename TR0 , typename TR1 , typename TR2 , typename TR3 , typename TR4 , typename TR5>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3 , T4 , T5>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3 , lcos::future<T4, TR4> const& f4 , lcos::future<T5, TR5> const& f5)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3 , lcos::future<void> const& f4 , lcos::future<void> const& f5)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get();
    }
}}
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6,
        typename TR0 , typename TR1 , typename TR2 , typename TR3 , typename TR4 , typename TR5 , typename TR6>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3 , T4 , T5 , T6>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3 , lcos::future<T4, TR4> const& f4 , lcos::future<T5, TR5> const& f5 , lcos::future<T6, TR6> const& f6)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3 , lcos::future<void> const& f4 , lcos::future<void> const& f5 , lcos::future<void> const& f6)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get();
    }
}}
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7,
        typename TR0 , typename TR1 , typename TR2 , typename TR3 , typename TR4 , typename TR5 , typename TR6 , typename TR7>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3 , lcos::future<T4, TR4> const& f4 , lcos::future<T5, TR5> const& f5 , lcos::future<T6, TR6> const& f6 , lcos::future<T7, TR7> const& f7)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3 , lcos::future<void> const& f4 , lcos::future<void> const& f5 , lcos::future<void> const& f6 , lcos::future<void> const& f7)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get();
    }
}}
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8,
        typename TR0 , typename TR1 , typename TR2 , typename TR3 , typename TR4 , typename TR5 , typename TR6 , typename TR7 , typename TR8>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3 , lcos::future<T4, TR4> const& f4 , lcos::future<T5, TR5> const& f5 , lcos::future<T6, TR6> const& f6 , lcos::future<T7, TR7> const& f7 , lcos::future<T8, TR8> const& f8)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3 , lcos::future<void> const& f4 , lcos::future<void> const& f5 , lcos::future<void> const& f6 , lcos::future<void> const& f7 , lcos::future<void> const& f8)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get();
    }
}}
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9,
        typename TR0 , typename TR1 , typename TR2 , typename TR3 , typename TR4 , typename TR5 , typename TR6 , typename TR7 , typename TR8 , typename TR9>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3 , lcos::future<T4, TR4> const& f4 , lcos::future<T5, TR5> const& f5 , lcos::future<T6, TR6> const& f6 , lcos::future<T7, TR7> const& f7 , lcos::future<T8, TR8> const& f8 , lcos::future<T9, TR9> const& f9)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get() , f9.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3 , lcos::future<void> const& f4 , lcos::future<void> const& f5 , lcos::future<void> const& f6 , lcos::future<void> const& f7 , lcos::future<void> const& f8 , lcos::future<void> const& f9)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get(); f9.get();
    }
}}
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10,
        typename TR0 , typename TR1 , typename TR2 , typename TR3 , typename TR4 , typename TR5 , typename TR6 , typename TR7 , typename TR8 , typename TR9 , typename TR10>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3 , lcos::future<T4, TR4> const& f4 , lcos::future<T5, TR5> const& f5 , lcos::future<T6, TR6> const& f6 , lcos::future<T7, TR7> const& f7 , lcos::future<T8, TR8> const& f8 , lcos::future<T9, TR9> const& f9 , lcos::future<T10, TR10> const& f10)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get() , f9.get() , f10.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3 , lcos::future<void> const& f4 , lcos::future<void> const& f5 , lcos::future<void> const& f6 , lcos::future<void> const& f7 , lcos::future<void> const& f8 , lcos::future<void> const& f9 , lcos::future<void> const& f10)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get(); f9.get(); f10.get();
    }
}}
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11,
        typename TR0 , typename TR1 , typename TR2 , typename TR3 , typename TR4 , typename TR5 , typename TR6 , typename TR7 , typename TR8 , typename TR9 , typename TR10 , typename TR11>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3 , lcos::future<T4, TR4> const& f4 , lcos::future<T5, TR5> const& f5 , lcos::future<T6, TR6> const& f6 , lcos::future<T7, TR7> const& f7 , lcos::future<T8, TR8> const& f8 , lcos::future<T9, TR9> const& f9 , lcos::future<T10, TR10> const& f10 , lcos::future<T11, TR11> const& f11)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get() , f9.get() , f10.get() , f11.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3 , lcos::future<void> const& f4 , lcos::future<void> const& f5 , lcos::future<void> const& f6 , lcos::future<void> const& f7 , lcos::future<void> const& f8 , lcos::future<void> const& f9 , lcos::future<void> const& f10 , lcos::future<void> const& f11)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get(); f9.get(); f10.get(); f11.get();
    }
}}
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12,
        typename TR0 , typename TR1 , typename TR2 , typename TR3 , typename TR4 , typename TR5 , typename TR6 , typename TR7 , typename TR8 , typename TR9 , typename TR10 , typename TR11 , typename TR12>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3 , lcos::future<T4, TR4> const& f4 , lcos::future<T5, TR5> const& f5 , lcos::future<T6, TR6> const& f6 , lcos::future<T7, TR7> const& f7 , lcos::future<T8, TR8> const& f8 , lcos::future<T9, TR9> const& f9 , lcos::future<T10, TR10> const& f10 , lcos::future<T11, TR11> const& f11 , lcos::future<T12, TR12> const& f12)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get() , f9.get() , f10.get() , f11.get() , f12.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3 , lcos::future<void> const& f4 , lcos::future<void> const& f5 , lcos::future<void> const& f6 , lcos::future<void> const& f7 , lcos::future<void> const& f8 , lcos::future<void> const& f9 , lcos::future<void> const& f10 , lcos::future<void> const& f11 , lcos::future<void> const& f12)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get(); f9.get(); f10.get(); f11.get(); f12.get();
    }
}}
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13,
        typename TR0 , typename TR1 , typename TR2 , typename TR3 , typename TR4 , typename TR5 , typename TR6 , typename TR7 , typename TR8 , typename TR9 , typename TR10 , typename TR11 , typename TR12 , typename TR13>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3 , lcos::future<T4, TR4> const& f4 , lcos::future<T5, TR5> const& f5 , lcos::future<T6, TR6> const& f6 , lcos::future<T7, TR7> const& f7 , lcos::future<T8, TR8> const& f8 , lcos::future<T9, TR9> const& f9 , lcos::future<T10, TR10> const& f10 , lcos::future<T11, TR11> const& f11 , lcos::future<T12, TR12> const& f12 , lcos::future<T13, TR13> const& f13)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get() , f9.get() , f10.get() , f11.get() , f12.get() , f13.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3 , lcos::future<void> const& f4 , lcos::future<void> const& f5 , lcos::future<void> const& f6 , lcos::future<void> const& f7 , lcos::future<void> const& f8 , lcos::future<void> const& f9 , lcos::future<void> const& f10 , lcos::future<void> const& f11 , lcos::future<void> const& f12 , lcos::future<void> const& f13)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get(); f9.get(); f10.get(); f11.get(); f12.get(); f13.get();
    }
}}
namespace hpx { namespace lcos
{
    template <
        typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14,
        typename TR0 , typename TR1 , typename TR2 , typename TR3 , typename TR4 , typename TR5 , typename TR6 , typename TR7 , typename TR8 , typename TR9 , typename TR10 , typename TR11 , typename TR12 , typename TR13 , typename TR14>
    inline HPX_STD_TUPLE<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14>
    wait (lcos::future<T0, TR0> const& f0 , lcos::future<T1, TR1> const& f1 , lcos::future<T2, TR2> const& f2 , lcos::future<T3, TR3> const& f3 , lcos::future<T4, TR4> const& f4 , lcos::future<T5, TR5> const& f5 , lcos::future<T6, TR6> const& f6 , lcos::future<T7, TR7> const& f7 , lcos::future<T8, TR8> const& f8 , lcos::future<T9, TR9> const& f9 , lcos::future<T10, TR10> const& f10 , lcos::future<T11, TR11> const& f11 , lcos::future<T12, TR12> const& f12 , lcos::future<T13, TR13> const& f13 , lcos::future<T14, TR14> const& f14)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get() , f9.get() , f10.get() , f11.get() , f12.get() , f13.get() , f14.get());
    }
    inline void
    wait (lcos::future<void> const& f0 , lcos::future<void> const& f1 , lcos::future<void> const& f2 , lcos::future<void> const& f3 , lcos::future<void> const& f4 , lcos::future<void> const& f5 , lcos::future<void> const& f6 , lcos::future<void> const& f7 , lcos::future<void> const& f8 , lcos::future<void> const& f9 , lcos::future<void> const& f10 , lcos::future<void> const& f11 , lcos::future<void> const& f12 , lcos::future<void> const& f13 , lcos::future<void> const& f14)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get(); f9.get(); f10.get(); f11.get(); f12.get(); f13.get(); f14.get();
    }
}}
