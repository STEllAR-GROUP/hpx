////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_C03F6E84_EC71_42EB_A136_DB07980B133B)
#define HPX_C03F6E84_EC71_42EB_A136_DB07980B133B

namespace hpx { namespace agas { namespace traits
{

// MPL integral constant wrapper
template <typename T, typename Enable = void> 
struct serialization_version;

// Spirit-style CP
template <typename Mutex, typename Enable = void> 
struct initialize_mutex_hook;

template <typename Mutex>
inline void initialize_mutex(Mutex&);

}}}

#endif // HPX_C03F6E84_EC71_42EB_A136_DB07980B133B

