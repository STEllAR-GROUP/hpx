////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_C03F6E84_EC71_42EB_A136_DB07980B133B)
#define HPX_C03F6E84_EC71_42EB_A136_DB07980B133B

#include <string>

namespace hpx { namespace agas { namespace traits
{

// MPL integral constant wrapper
template <typename T, typename Enable = void> 
struct serialization_version;

// MPL metafunction
template <typename T, typename Enable = void> 
struct mutex_type;

// Spirit-style CP
template <typename Mutex, typename Enable = void> 
struct initialize_mutex_hook;

template <typename Mutex>
inline void initialize_mutex(Mutex&);

///////////////////////////////////////////////////////////////////////////////
// AGAS namespace traits. 

// MPL metafunction
template <typename Tag, typename Enable = void> 
struct registry_type;

// MPL metafunction
template <typename Tag, typename Enable = void> 
struct key_type;

// MPL metafunction
template <typename Tag, typename Enable = void> 
struct mapped_type;

///////////////////////////////////////////////////////////////////////////////
// Network protocol traits.

// MPL metafunction
template <typename Protocal, typename Enable = void> 
struct locality_type;

// Spirit-style CP
template <typename Protocal, typename Enable = void> 
struct protocal_name_hook;

template <typename Protocal>
inline std::string protocal_name();

///////////////////////////////////////////////////////////////////////////////
// basic_namespace implementation hooks (Spirit-style CPs).

template <typename Tag, typename Enable = void>
struct bind_hook;

template <typename Tag>
inline typename key_type<Tag>::type
bind(typename registry_type<Tag>::type&,
     typename key_type<Tag>::type const&,
     typename mapped_type<Tag>::type const&);

template <typename Tag, typename Enable = void>
struct update_hook;

template <typename Tag>
inline typename key_type<Tag>::type
update(typename registry_type<Tag>::type&,
       typename key_type<Tag>::type const&,
       typename mapped_type<Tag>::type const&);

template <typename Tag, typename Enable = void>
struct resolve_hook;

template <typename Tag>
inline typename mapped_type<Tag>::type
resolve(typename registry_type<Tag>::type&,
        typename key_type<Tag>::type const&);

template <typename Tag, typename Enable = void>
struct unbind_hook;

template <typename Tag>
inline bool 
unbind(typename registry_type<Tag>::type&,
       typename key_type<Tag>::type const&);

}}}

#endif // HPX_C03F6E84_EC71_42EB_A136_DB07980B133B

