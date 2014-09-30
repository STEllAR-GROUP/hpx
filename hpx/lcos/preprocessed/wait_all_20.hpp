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
    void wait_all(T0 && f0)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1>
    void wait_all(T0 && f0 , T1 && f1)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12) , lcos::detail::get_shared_state(f13));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12) , lcos::detail::get_shared_state(f13) , lcos::detail::get_shared_state(f14));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type , typename lcos::detail::shared_state_ptr_for<T15>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12) , lcos::detail::get_shared_state(f13) , lcos::detail::get_shared_state(f14) , lcos::detail::get_shared_state(f15));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type , typename lcos::detail::shared_state_ptr_for<T15>::type , typename lcos::detail::shared_state_ptr_for<T16>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12) , lcos::detail::get_shared_state(f13) , lcos::detail::get_shared_state(f14) , lcos::detail::get_shared_state(f15) , lcos::detail::get_shared_state(f16));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type , typename lcos::detail::shared_state_ptr_for<T15>::type , typename lcos::detail::shared_state_ptr_for<T16>::type , typename lcos::detail::shared_state_ptr_for<T17>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12) , lcos::detail::get_shared_state(f13) , lcos::detail::get_shared_state(f14) , lcos::detail::get_shared_state(f15) , lcos::detail::get_shared_state(f16) , lcos::detail::get_shared_state(f17));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17 , T18 && f18)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type , typename lcos::detail::shared_state_ptr_for<T15>::type , typename lcos::detail::shared_state_ptr_for<T16>::type , typename lcos::detail::shared_state_ptr_for<T17>::type , typename lcos::detail::shared_state_ptr_for<T18>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12) , lcos::detail::get_shared_state(f13) , lcos::detail::get_shared_state(f14) , lcos::detail::get_shared_state(f15) , lcos::detail::get_shared_state(f16) , lcos::detail::get_shared_state(f17) , lcos::detail::get_shared_state(f18));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17 , T18 && f18 , T19 && f19)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type , typename lcos::detail::shared_state_ptr_for<T15>::type , typename lcos::detail::shared_state_ptr_for<T16>::type , typename lcos::detail::shared_state_ptr_for<T17>::type , typename lcos::detail::shared_state_ptr_for<T18>::type , typename lcos::detail::shared_state_ptr_for<T19>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12) , lcos::detail::get_shared_state(f13) , lcos::detail::get_shared_state(f14) , lcos::detail::get_shared_state(f15) , lcos::detail::get_shared_state(f16) , lcos::detail::get_shared_state(f17) , lcos::detail::get_shared_state(f18) , lcos::detail::get_shared_state(f19));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17 , T18 && f18 , T19 && f19 , T20 && f20)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type , typename lcos::detail::shared_state_ptr_for<T15>::type , typename lcos::detail::shared_state_ptr_for<T16>::type , typename lcos::detail::shared_state_ptr_for<T17>::type , typename lcos::detail::shared_state_ptr_for<T18>::type , typename lcos::detail::shared_state_ptr_for<T19>::type , typename lcos::detail::shared_state_ptr_for<T20>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12) , lcos::detail::get_shared_state(f13) , lcos::detail::get_shared_state(f14) , lcos::detail::get_shared_state(f15) , lcos::detail::get_shared_state(f16) , lcos::detail::get_shared_state(f17) , lcos::detail::get_shared_state(f18) , lcos::detail::get_shared_state(f19) , lcos::detail::get_shared_state(f20));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17 , T18 && f18 , T19 && f19 , T20 && f20 , T21 && f21)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type , typename lcos::detail::shared_state_ptr_for<T15>::type , typename lcos::detail::shared_state_ptr_for<T16>::type , typename lcos::detail::shared_state_ptr_for<T17>::type , typename lcos::detail::shared_state_ptr_for<T18>::type , typename lcos::detail::shared_state_ptr_for<T19>::type , typename lcos::detail::shared_state_ptr_for<T20>::type , typename lcos::detail::shared_state_ptr_for<T21>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12) , lcos::detail::get_shared_state(f13) , lcos::detail::get_shared_state(f14) , lcos::detail::get_shared_state(f15) , lcos::detail::get_shared_state(f16) , lcos::detail::get_shared_state(f17) , lcos::detail::get_shared_state(f18) , lcos::detail::get_shared_state(f19) , lcos::detail::get_shared_state(f20) , lcos::detail::get_shared_state(f21));
        frame_type frame(values);
        frame.wait_all();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17 , typename T18 , typename T19 , typename T20 , typename T21 , typename T22>
    void wait_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17 , T18 && f18 , T19 && f19 , T20 && f20 , T21 && f21 , T22 && f22)
    {
        typedef hpx::util::tuple<
                typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type , typename lcos::detail::shared_state_ptr_for<T13>::type , typename lcos::detail::shared_state_ptr_for<T14>::type , typename lcos::detail::shared_state_ptr_for<T15>::type , typename lcos::detail::shared_state_ptr_for<T16>::type , typename lcos::detail::shared_state_ptr_for<T17>::type , typename lcos::detail::shared_state_ptr_for<T18>::type , typename lcos::detail::shared_state_ptr_for<T19>::type , typename lcos::detail::shared_state_ptr_for<T20>::type , typename lcos::detail::shared_state_ptr_for<T21>::type , typename lcos::detail::shared_state_ptr_for<T22>::type
            > result_type;
        typedef detail::wait_all_frame<result_type> frame_type;
        result_type values(lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12) , lcos::detail::get_shared_state(f13) , lcos::detail::get_shared_state(f14) , lcos::detail::get_shared_state(f15) , lcos::detail::get_shared_state(f16) , lcos::detail::get_shared_state(f17) , lcos::detail::get_shared_state(f18) , lcos::detail::get_shared_state(f19) , lcos::detail::get_shared_state(f20) , lcos::detail::get_shared_state(f21) , lcos::detail::get_shared_state(f22));
        frame_type frame(values);
        frame.wait_all();
    }
}}
