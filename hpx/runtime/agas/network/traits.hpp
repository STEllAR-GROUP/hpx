////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_BB272F46_646C_4DEC_A54A_B78181769456)
#define HPX_BB272F46_646C_4DEC_A54A_B78181769456

#include <hpx/runtime/agas/network/traits_fwd.hpp>

namespace hpx { namespace agas { namespace traits { namespace network 
{

template <typename Protocol>
inline typename name_hook<Protocol>::result_type name()
{ return name_hook<Protocol>::call(); }

}}}}

#endif // HPX_BB272F46_646C_4DEC_A54A_B78181769456

