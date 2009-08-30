//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_LOCAL_SET_AUG_13_2009_1056PM)
#define HPX_COMPONENTS_LOCAL_SET_AUG_13_2009_1056PM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/local_set.hpp"

namespace hpx { namespace components
{
    template <typename Item>
    class local_set
      : public client_base<local_set<Item>, stubs::local_set<Item> >
    {
    private:
        typedef client_base<
                    local_set<Item>, stubs::local_set<Item>
                > base_type;

    public:
        local_set(naming::id_type gid, bool freeonexit = true)
          : base_type(gid, freeonexit)
        {
            LAPP_(info) << " [LOCAL_SET] " << "Constructing local list" << std::endl;
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        typedef std::vector<naming::id_type> set_type;

        naming::id_type add_item(naming::id_type item=naming::invalid_id)
        {
            return this->base_type::add_item(this->gid_, item);
        }

        int append(set_type list)
        {
            return this->base_type::append(this->gid_, list);
        }

        set_type get(void)
        {
            return this->base_type::get(this->gid_);
        }
    };
    
}}

#endif
