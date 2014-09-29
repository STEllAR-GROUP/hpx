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
