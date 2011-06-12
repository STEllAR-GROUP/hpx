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

struct flush_type { };
struct endl_type { };
struct local_flush_type { };
struct local_endl_type { };

HPX_EXPORT extern flush_type flush; 
HPX_EXPORT extern endl_type endl; 
HPX_EXPORT extern local_flush_type local_flush; 
HPX_EXPORT extern local_endl_type local_endl; 

inline std::ostream& operator<< (std::ostream& os, flush_type const&)
{ return os << std::flush; }

inline std::ostream& operator<< (std::ostream& os, endl_type const&)
{ return os << std::endl; }

inline std::ostream& operator<< (std::ostream& os, local_flush_type const&)
{ return os << std::flush; }

inline std::ostream& operator<< (std::ostream& os, local_endl_type const&)
{ return os << std::endl; }

}

using iostreams::flush;
using iostreams::endl;
using iostreams::local_flush;
using iostreams::local_endl;

}

#endif // HPX_9CF0FD9D_039C_4046_B244_C7FCC97D9945

