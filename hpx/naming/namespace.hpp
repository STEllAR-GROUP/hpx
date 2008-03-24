//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_NAMESPACE_MAR_24_2008_1007AM)
#define HPX_NAMING_NAMESPACE_MAR_24_2008_1007AM

#include <iosfwd>

#include <boost/cstdint.hpp>

#include <hpx/util/util.hpp>
#include <hpx/naming/name.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    ///////////////////////////////////////////////////////////////////////////
    inline std::string 
    get_node_name(std::string const& parent_name, std::string const& node_name)
    {
        return parent_name + "/" + node_name;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline std::string 
    get_component_name(std::string const& node_name, naming::id_type id)
    {
        HPX_OSSTREAM s;
        s << node_name << "/" << std::hex << id;
        return HPX_OSSTREAM_GETSTRING(s);
    }

    ///////////////////////////////////////////////////////////////////////////
    //  The name of the locality is always '/localities/<prefix>', which makes 
    //  it easy to reference this locality in the global namespace
    inline std::string 
    get_locality_name(boost::uint64_t prefix)
    {
        return get_component_name("/localities", prefix);
    }
    
    ///////////////////////////////////////////////////////////////////////////
    //  The name of the factory component in a locality is always 
    //  '/localities/<prefix>/factory', which makes it easy to create a new 
    //  component on a specific locality
    inline std::string 
    get_factory_name(boost::uint64_t prefix)
    {
        return get_node_name(get_locality_name(prefix), "factory");
    }
    
    ///////////////////////////////////////////////////////////////////////////
    //  The name of a thread in a locality is always 
    //  '/localities/<prefix>/threads/<thread_id>', which makes it easy to 
    //  access a specific thread on a specific locality
    inline std::string 
    get_thread_name(boost::uint64_t prefix, naming::id_type id)
    {
        HPX_OSSTREAM s;
        s << "/localities/" << std::hex << prefix 
          << "/threads/" << std::hex << id;
        return HPX_OSSTREAM_GETSTRING(s);
    }
    
///////////////////////////////////////////////////////////////////////////////
}}

#endif 
