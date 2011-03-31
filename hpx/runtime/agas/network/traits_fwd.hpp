////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_75B1EEED_DAC6_47D3_81CE_9A5BE97B3CB0)
#define HPX_75B1EEED_DAC6_47D3_81CE_9A5BE97B3CB0

namespace hpx { namespace agas { namespace traits { namespace network 
{

// MPL metafunction
template <typename Protocol, typename Enable = void>
struct endpoint_type;

// Spirit-style CP
template <typename Protocol, typename Enable = void>
struct name_hook;

template <typename Protocol>
inline typename name_hook<Protocol>::result_type name();

}}}}

#endif // HPX_75B1EEED_DAC6_47D3_81CE_9A5BE97B3CB0

