////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_C03F6E84_EC71_42EB_A136_DB07980B133B)
#define HPX_C03F6E84_EC71_42EB_A136_DB07980B133B

#include <string>

namespace hpx { namespace agas { namespace magic
{

// MPL metafunction
template <typename Tag, typename Enable = void> 
struct mutex_type;

// Spirit-style CP
template <typename Mutex, typename Enable = void> 
struct initialize_mutex_hook;

template <typename Mutex>
inline void initialize_mutex(Mutex&);

// MPL metafunction
template <typename Tag, typename Enable = void> 
struct registry_type;

// MPL metafunction
template <typename Tag, typename Enable = void> 
struct key_type;

// MPL metafunction
template <typename Tag, typename Enable = void> 
struct mapped_type;

// MPL metafunction
template <typename Protocal, typename Enable = void> 
struct locality_type;

// Spirit-style CP
template <typename Protocal, typename Enable = void> 
struct protocal_name_hook;

template <typename Protocal>
inline std::string protocal_name();

///////////////////////////////////////////////////////////////////////////////
// basic_namespace implementation hooks (Spirit-stlye CPs)

// TODO: write prototypes for bind, unbind and resolve

}}}

#endif // HPX_C03F6E84_EC71_42EB_A136_DB07980B133B

