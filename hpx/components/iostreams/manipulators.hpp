////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_9CF0FD9D_039C_4046_B244_C7FCC97D9945)
#define HPX_9CF0FD9D_039C_4046_B244_C7FCC97D9945

#include <ostream>

#include <hpx/config.hpp>

namespace hpx { namespace iostreams
{

struct sync_flush_type { };
struct sync_endl_type { };
struct flush_type : sync_flush_type { };
struct endl_type : sync_endl_type { };
struct async_flush_type { };
struct async_endl_type { };
struct local_flush_type { };
struct local_endl_type { };

// #if !defined(BOOST_WINDOWS)
HPX_COMPONENT_EXPORT extern sync_flush_type sync_flush; 
HPX_COMPONENT_EXPORT extern sync_endl_type sync_endl; 
HPX_COMPONENT_EXPORT extern flush_type flush; // alias for hpx::sync_flush 
HPX_COMPONENT_EXPORT extern endl_type endl; // alias for hpx::sync_endl
HPX_COMPONENT_EXPORT extern async_flush_type async_flush; 
HPX_COMPONENT_EXPORT extern async_endl_type async_endl; 
HPX_COMPONENT_EXPORT extern local_flush_type local_flush; 
HPX_COMPONENT_EXPORT extern local_endl_type local_endl; 
// #else
// sync_flush_type const sync_flush = {}; 
// sync_endl_type const sync_endl = {}; 
// flush_type const flush = flush_type(); // alias for hpx::sync_flush 
// endl_type const endl = endl_type(); // alias for hpx::sync_endl
// async_flush_type const async_flush = {}; 
// async_endl_type const async_endl= {}; 
// local_flush_type const local_flush = {}; 
// local_endl_type const local_endl = {}; 
// #endif

inline std::ostream& operator<< (std::ostream& os, sync_flush_type const&)
{ return os << std::flush; }

inline std::ostream& operator<< (std::ostream& os, sync_endl_type const&)
{ return os << std::endl; }

inline std::ostream& operator<< (std::ostream& os, async_flush_type const&)
{ return os << std::flush; }

inline std::ostream& operator<< (std::ostream& os, async_endl_type const&)
{ return os << std::endl; }

inline std::ostream& operator<< (std::ostream& os, local_flush_type const&)
{ return os << std::flush; }

inline std::ostream& operator<< (std::ostream& os, local_endl_type const&)
{ return os << std::endl; }

}

using iostreams::sync_flush;
using iostreams::sync_endl;
using iostreams::flush;
using iostreams::endl;
using iostreams::async_flush;
using iostreams::async_endl;
using iostreams::local_flush;
using iostreams::local_endl;

}

#endif // HPX_9CF0FD9D_039C_4046_B244_C7FCC97D9945

