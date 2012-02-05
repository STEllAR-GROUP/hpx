////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_16218775_E5D0_4B5A_BBA5_3C561B9E322F)
#define HPX_16218775_E5D0_4B5A_BBA5_3C561B9E322F

#include <boost/serialization/void_cast.hpp>

namespace hpx { namespace util
{

template <
    typename Derived
  , typename Base
>
inline boost::serialization::void_cast_detail::void_caster const&
void_cast_register_nonvirt(
    Derived const* = NULL 
  , Base const* = NULL 
    )
{
    using namespace boost::serialization::void_cast_detail;
    typedef void_caster_primitive<Derived, Base> type;
    return boost::serialization::singleton<type>::get_const_instance();
}

}}

#endif // HPX_16218775_E5D0_4B5A_BBA5_3C561B9E322F

