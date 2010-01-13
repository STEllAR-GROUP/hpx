//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_LOCAL_MAP_AUG_13_2009_1056PM)
#define HPX_COMPONENTS_LOCAL_MAP_AUG_13_2009_1056PM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/local_map.hpp"

namespace hpx { namespace components
{
    template <typename List>
    class local_map
      : public client_base<local_map<List>, stubs::local_map<List> >
    {
    private:
        typedef client_base<
                    local_map<List>, stubs::local_map<List>
                > base_type;

    public:
        local_map()
          : base_type(naming::invalid_id, true)
        {}

        local_map(naming::id_type gid, bool freeonexit = true)
          : base_type(gid, freeonexit)
        { }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        int append(List list)
        {
            return this->base_type::append(this->gid_, list);
        }

        List get(void)
        {
            return this->base_type::get(this->gid_);
        }

        naming::id_type value(naming::id_type key, components::component_type value_type)
        {
            return this->base_type::value(this->gid_, key, value_type);
        }
    };
    
}}

#endif
