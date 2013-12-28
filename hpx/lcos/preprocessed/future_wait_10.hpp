// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace lcos
{
    template <typename F0 , typename F1 , typename F2>
    inline typename boost::enable_if_c<
        !(true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value)
      , HPX_STD_TUPLE<typename detail::future_traits<F0>::type , typename detail::future_traits<F1>::type , typename detail::future_traits<F2>::type>
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get());
    }
    
    template <typename F0 , typename F1 , typename F2>
    inline typename boost::enable_if_c<
        (true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value)
      , void
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2)
    {
        f0.get(); f1.get(); f2.get();
    }
}}
namespace hpx { namespace lcos
{
    template <typename F0 , typename F1 , typename F2 , typename F3>
    inline typename boost::enable_if_c<
        !(true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value)
      , HPX_STD_TUPLE<typename detail::future_traits<F0>::type , typename detail::future_traits<F1>::type , typename detail::future_traits<F2>::type , typename detail::future_traits<F3>::type>
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get());
    }
    
    template <typename F0 , typename F1 , typename F2 , typename F3>
    inline typename boost::enable_if_c<
        (true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value)
      , void
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3)
    {
        f0.get(); f1.get(); f2.get(); f3.get();
    }
}}
namespace hpx { namespace lcos
{
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
    inline typename boost::enable_if_c<
        !(true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value)
      , HPX_STD_TUPLE<typename detail::future_traits<F0>::type , typename detail::future_traits<F1>::type , typename detail::future_traits<F2>::type , typename detail::future_traits<F3>::type , typename detail::future_traits<F4>::type>
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get());
    }
    
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4>
    inline typename boost::enable_if_c<
        (true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value)
      , void
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get();
    }
}}
namespace hpx { namespace lcos
{
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5>
    inline typename boost::enable_if_c<
        !(true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value)
      , HPX_STD_TUPLE<typename detail::future_traits<F0>::type , typename detail::future_traits<F1>::type , typename detail::future_traits<F2>::type , typename detail::future_traits<F3>::type , typename detail::future_traits<F4>::type , typename detail::future_traits<F5>::type>
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get());
    }
    
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5>
    inline typename boost::enable_if_c<
        (true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value)
      , void
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get();
    }
}}
namespace hpx { namespace lcos
{
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6>
    inline typename boost::enable_if_c<
        !(true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value)
      , HPX_STD_TUPLE<typename detail::future_traits<F0>::type , typename detail::future_traits<F1>::type , typename detail::future_traits<F2>::type , typename detail::future_traits<F3>::type , typename detail::future_traits<F4>::type , typename detail::future_traits<F5>::type , typename detail::future_traits<F6>::type>
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get());
    }
    
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6>
    inline typename boost::enable_if_c<
        (true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value)
      , void
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get();
    }
}}
namespace hpx { namespace lcos
{
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7>
    inline typename boost::enable_if_c<
        !(true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value)
      , HPX_STD_TUPLE<typename detail::future_traits<F0>::type , typename detail::future_traits<F1>::type , typename detail::future_traits<F2>::type , typename detail::future_traits<F3>::type , typename detail::future_traits<F4>::type , typename detail::future_traits<F5>::type , typename detail::future_traits<F6>::type , typename detail::future_traits<F7>::type>
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get());
    }
    
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7>
    inline typename boost::enable_if_c<
        (true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value)
      , void
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get();
    }
}}
namespace hpx { namespace lcos
{
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8>
    inline typename boost::enable_if_c<
        !(true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value && boost::is_void< typename detail::future_traits<F8>::type>::value)
      , HPX_STD_TUPLE<typename detail::future_traits<F0>::type , typename detail::future_traits<F1>::type , typename detail::future_traits<F2>::type , typename detail::future_traits<F3>::type , typename detail::future_traits<F4>::type , typename detail::future_traits<F5>::type , typename detail::future_traits<F6>::type , typename detail::future_traits<F7>::type , typename detail::future_traits<F8>::type>
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get());
    }
    
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8>
    inline typename boost::enable_if_c<
        (true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value && boost::is_void< typename detail::future_traits<F8>::type>::value)
      , void
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get();
    }
}}
namespace hpx { namespace lcos
{
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9>
    inline typename boost::enable_if_c<
        !(true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value && boost::is_void< typename detail::future_traits<F8>::type>::value && boost::is_void< typename detail::future_traits<F9>::type>::value)
      , HPX_STD_TUPLE<typename detail::future_traits<F0>::type , typename detail::future_traits<F1>::type , typename detail::future_traits<F2>::type , typename detail::future_traits<F3>::type , typename detail::future_traits<F4>::type , typename detail::future_traits<F5>::type , typename detail::future_traits<F6>::type , typename detail::future_traits<F7>::type , typename detail::future_traits<F8>::type , typename detail::future_traits<F9>::type>
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get() , f9.get());
    }
    
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9>
    inline typename boost::enable_if_c<
        (true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value && boost::is_void< typename detail::future_traits<F8>::type>::value && boost::is_void< typename detail::future_traits<F9>::type>::value)
      , void
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get(); f9.get();
    }
}}
namespace hpx { namespace lcos
{
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10>
    inline typename boost::enable_if_c<
        !(true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value && boost::is_void< typename detail::future_traits<F8>::type>::value && boost::is_void< typename detail::future_traits<F9>::type>::value && boost::is_void< typename detail::future_traits<F10>::type>::value)
      , HPX_STD_TUPLE<typename detail::future_traits<F0>::type , typename detail::future_traits<F1>::type , typename detail::future_traits<F2>::type , typename detail::future_traits<F3>::type , typename detail::future_traits<F4>::type , typename detail::future_traits<F5>::type , typename detail::future_traits<F6>::type , typename detail::future_traits<F7>::type , typename detail::future_traits<F8>::type , typename detail::future_traits<F9>::type , typename detail::future_traits<F10>::type>
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get() , f9.get() , f10.get());
    }
    
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10>
    inline typename boost::enable_if_c<
        (true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value && boost::is_void< typename detail::future_traits<F8>::type>::value && boost::is_void< typename detail::future_traits<F9>::type>::value && boost::is_void< typename detail::future_traits<F10>::type>::value)
      , void
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get(); f9.get(); f10.get();
    }
}}
namespace hpx { namespace lcos
{
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11>
    inline typename boost::enable_if_c<
        !(true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value && boost::is_void< typename detail::future_traits<F8>::type>::value && boost::is_void< typename detail::future_traits<F9>::type>::value && boost::is_void< typename detail::future_traits<F10>::type>::value && boost::is_void< typename detail::future_traits<F11>::type>::value)
      , HPX_STD_TUPLE<typename detail::future_traits<F0>::type , typename detail::future_traits<F1>::type , typename detail::future_traits<F2>::type , typename detail::future_traits<F3>::type , typename detail::future_traits<F4>::type , typename detail::future_traits<F5>::type , typename detail::future_traits<F6>::type , typename detail::future_traits<F7>::type , typename detail::future_traits<F8>::type , typename detail::future_traits<F9>::type , typename detail::future_traits<F10>::type , typename detail::future_traits<F11>::type>
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get() , f9.get() , f10.get() , f11.get());
    }
    
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11>
    inline typename boost::enable_if_c<
        (true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value && boost::is_void< typename detail::future_traits<F8>::type>::value && boost::is_void< typename detail::future_traits<F9>::type>::value && boost::is_void< typename detail::future_traits<F10>::type>::value && boost::is_void< typename detail::future_traits<F11>::type>::value)
      , void
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get(); f9.get(); f10.get(); f11.get();
    }
}}
namespace hpx { namespace lcos
{
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12>
    inline typename boost::enable_if_c<
        !(true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value && boost::is_void< typename detail::future_traits<F8>::type>::value && boost::is_void< typename detail::future_traits<F9>::type>::value && boost::is_void< typename detail::future_traits<F10>::type>::value && boost::is_void< typename detail::future_traits<F11>::type>::value && boost::is_void< typename detail::future_traits<F12>::type>::value)
      , HPX_STD_TUPLE<typename detail::future_traits<F0>::type , typename detail::future_traits<F1>::type , typename detail::future_traits<F2>::type , typename detail::future_traits<F3>::type , typename detail::future_traits<F4>::type , typename detail::future_traits<F5>::type , typename detail::future_traits<F6>::type , typename detail::future_traits<F7>::type , typename detail::future_traits<F8>::type , typename detail::future_traits<F9>::type , typename detail::future_traits<F10>::type , typename detail::future_traits<F11>::type , typename detail::future_traits<F12>::type>
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12)
    {
        return HPX_STD_MAKE_TUPLE(f0.get() , f1.get() , f2.get() , f3.get() , f4.get() , f5.get() , f6.get() , f7.get() , f8.get() , f9.get() , f10.get() , f11.get() , f12.get());
    }
    
    template <typename F0 , typename F1 , typename F2 , typename F3 , typename F4 , typename F5 , typename F6 , typename F7 , typename F8 , typename F9 , typename F10 , typename F11 , typename F12>
    inline typename boost::enable_if_c<
        (true && boost::is_void< typename detail::future_traits<F0>::type>::value && boost::is_void< typename detail::future_traits<F1>::type>::value && boost::is_void< typename detail::future_traits<F2>::type>::value && boost::is_void< typename detail::future_traits<F3>::type>::value && boost::is_void< typename detail::future_traits<F4>::type>::value && boost::is_void< typename detail::future_traits<F5>::type>::value && boost::is_void< typename detail::future_traits<F6>::type>::value && boost::is_void< typename detail::future_traits<F7>::type>::value && boost::is_void< typename detail::future_traits<F8>::type>::value && boost::is_void< typename detail::future_traits<F9>::type>::value && boost::is_void< typename detail::future_traits<F10>::type>::value && boost::is_void< typename detail::future_traits<F11>::type>::value && boost::is_void< typename detail::future_traits<F12>::type>::value)
      , void
    >::type
    wait(BOOST_FWD_REF(F0) f0 , BOOST_FWD_REF(F1) f1 , BOOST_FWD_REF(F2) f2 , BOOST_FWD_REF(F3) f3 , BOOST_FWD_REF(F4) f4 , BOOST_FWD_REF(F5) f5 , BOOST_FWD_REF(F6) f6 , BOOST_FWD_REF(F7) f7 , BOOST_FWD_REF(F8) f8 , BOOST_FWD_REF(F9) f9 , BOOST_FWD_REF(F10) f10 , BOOST_FWD_REF(F11) f11 , BOOST_FWD_REF(F12) f12)
    {
        f0.get(); f1.get(); f2.get(); f3.get(); f4.get(); f5.get(); f6.get(); f7.get(); f8.get(); f9.get(); f10.get(); f11.get(); f12.get();
    }
}}
