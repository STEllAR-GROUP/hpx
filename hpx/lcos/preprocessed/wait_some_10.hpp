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
    void wait_some(std::size_t n, T0 && f0,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0));
        if (n == 0)
        {
            return;
        }
        if (n > 1)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1));
        if (n == 0)
        {
            return;
        }
        if (n > 2)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1 , T2 && f2,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2));
        if (n == 0)
        {
            return;
        }
        if (n > 3)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3));
        if (n == 0)
        {
            return;
        }
        if (n > 4)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4));
        if (n == 0)
        {
            return;
        }
        if (n > 5)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5));
        if (n == 0)
        {
            return;
        }
        if (n > 6)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6));
        if (n == 0)
        {
            return;
        }
        if (n > 7)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7));
        if (n == 0)
        {
            return;
        }
        if (n > 8)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8));
        if (n == 0)
        {
            return;
        }
        if (n > 9)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9));
        if (n == 0)
        {
            return;
        }
        if (n > 10)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10));
        if (n == 0)
        {
            return;
        }
        if (n > 11)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11));
        if (n == 0)
        {
            return;
        }
        if (n > 12)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
namespace hpx { namespace lcos
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    void wait_some(std::size_t n, T0 && f0 , T1 && f1 , T2 && f2 , T3 && f3 , T4 && f4 , T5 && f5 , T6 && f6 , T7 && f7 , T8 && f8 , T9 && f9 , T10 && f10 , T11 && f11 , T12 && f12,
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            typename lcos::detail::shared_state_ptr_for<T0>::type , typename lcos::detail::shared_state_ptr_for<T1>::type , typename lcos::detail::shared_state_ptr_for<T2>::type , typename lcos::detail::shared_state_ptr_for<T3>::type , typename lcos::detail::shared_state_ptr_for<T4>::type , typename lcos::detail::shared_state_ptr_for<T5>::type , typename lcos::detail::shared_state_ptr_for<T6>::type , typename lcos::detail::shared_state_ptr_for<T7>::type , typename lcos::detail::shared_state_ptr_for<T8>::type , typename lcos::detail::shared_state_ptr_for<T9>::type , typename lcos::detail::shared_state_ptr_for<T10>::type , typename lcos::detail::shared_state_ptr_for<T11>::type , typename lcos::detail::shared_state_ptr_for<T12>::type>
            result_type;
        result_type lazy_values_(
            lcos::detail::get_shared_state(f0) , lcos::detail::get_shared_state(f1) , lcos::detail::get_shared_state(f2) , lcos::detail::get_shared_state(f3) , lcos::detail::get_shared_state(f4) , lcos::detail::get_shared_state(f5) , lcos::detail::get_shared_state(f6) , lcos::detail::get_shared_state(f7) , lcos::detail::get_shared_state(f8) , lcos::detail::get_shared_state(f9) , lcos::detail::get_shared_state(f10) , lcos::detail::get_shared_state(f11) , lcos::detail::get_shared_state(f12));
        if (n == 0)
        {
            return;
        }
        if (n > 13)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }
        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);
        return (*f.get())();
    }
}}
