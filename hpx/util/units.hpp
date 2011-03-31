////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_C22F73E2_6809_4380_96B5_6B6FDF415584)
#define HPX_C22F73E2_6809_4380_96B5_6B6FDF415584

namespace hpx { namespace { util
{

enum byte_units {
  byte = 1,
  kilobyte = 1 << 10,
  megabyte = 1 << 20,
  gigabyte = 1 << 30,
  terabyte = 1 << 40
};

template <byte_units From, byte_units To, typename T>
inline double convert (T x)
{ return (double(x) * double(From)) / double(To); }

}}

#endif // HPX_C22F73E2_6809_4380_96B5_6B6FDF415584

