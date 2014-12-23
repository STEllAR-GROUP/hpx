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
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type> >
    when_all(T0 && f0)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type> >
    when_all(T0 && f0 , T1 && f1)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12) , traits::acquire_future_disp()(f13));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12) , traits::acquire_future_disp()(f13) , traits::acquire_future_disp()(f14));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12) , traits::acquire_future_disp()(f13) , traits::acquire_future_disp()(f14) , traits::acquire_future_disp()(f15));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type , typename traits::acquire_future<T16>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type , typename traits::acquire_future<T16>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12) , traits::acquire_future_disp()(f13) , traits::acquire_future_disp()(f14) , traits::acquire_future_disp()(f15) , traits::acquire_future_disp()(f16));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    lcos::future<hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type , typename traits::acquire_future<T16>::type , typename traits::acquire_future<T17>::type> >
    when_all(T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12 , T13 && f13 , T14 && f14 , T15 && f15 , T16 && f16 , T17 && f17)
    {
        typedef hpx::util::tuple<typename traits::acquire_future<T0>::type , typename traits::acquire_future<T1>::type , typename traits::acquire_future<T2>::type , typename traits::acquire_future<T3>::type , typename traits::acquire_future<T4>::type , typename traits::acquire_future<T5>::type , typename traits::acquire_future<T6>::type , typename traits::acquire_future<T7>::type , typename traits::acquire_future<T8>::type , typename traits::acquire_future<T9>::type , typename traits::acquire_future<T10>::type , typename traits::acquire_future<T11>::type , typename traits::acquire_future<T12>::type , typename traits::acquire_future<T13>::type , typename traits::acquire_future<T14>::type , typename traits::acquire_future<T15>::type , typename traits::acquire_future<T16>::type , typename traits::acquire_future<T17>::type>
            result_type;
        typedef detail::when_all_frame<result_type> frame_type;
        result_type values(traits::acquire_future_disp()(f0) , traits::acquire_future_disp()(f1) , traits::acquire_future_disp()(f2) , traits::acquire_future_disp()(f3) , traits::acquire_future_disp()(f4) , traits::acquire_future_disp()(f5) , traits::acquire_future_disp()(f6) , traits::acquire_future_disp()(f7) , traits::acquire_future_disp()(f8) , traits::acquire_future_disp()(f9) , traits::acquire_future_disp()(f10) , traits::acquire_future_disp()(f11) , traits::acquire_future_disp()(f12) , traits::acquire_future_disp()(f13) , traits::acquire_future_disp()(f14) , traits::acquire_future_disp()(f15) , traits::acquire_future_disp()(f16) , traits::acquire_future_disp()(f17));
        boost::intrusive_ptr<frame_type> p(new frame_type(std::move(values)));
        p->await();
        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}
