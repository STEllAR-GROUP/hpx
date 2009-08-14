//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_LOCAL_LIST_AUG_13_2009_1056PM)
#define HPX_COMPONENTS_LOCAL_LIST_AUG_13_2009_1056PM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/local_list.hpp"

namespace hpx { namespace components
{
    template <typename List>
    class local_list
      : public client_base<local_list<List>, stubs::local_list<List> >
    {
    private:
        typedef client_base<
                    local_list<List>, stubs::local_list<List>
                > base_type;

    public:
        local_list(naming::id_type gid, bool freeonexit = true)
          : base_type(gid, freeonexit)
        {
            std::cout << "Constructing local list" << std::endl;
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        int append(List list)
        {
            return this->base_type::append(this->gid_, list);
        }
    };
    
}}

#endif
