//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DISTRIBUTED_MAP_MAY_18_2008_0822AM)
#define HPX_COMPONENTS_DISTRIBUTED_MAP_MAY_18_2008_0822AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/distributed_map.hpp"

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a distributed_map class is the client side representation of a
    /// specific \a server#distributed_map component
    ///
    /// The purpose of the distributed list component is simply to wrap local
    /// non-component list types, to give them GIDs
    ///
    /// Remember that List must be serializable

    template <typename List>
    class distributed_map
      : public client_base<
                   distributed_map<List>, stubs::distributed_map<List>
               >
    {
    private:
        typedef client_base<
                    distributed_map<List>, stubs::distributed_map<List>
                > base_type;

    public:
        distributed_map(naming::id_type gid, bool freeonexit = true)
          : base_type(gid, freeonexit)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        naming::id_type get_local(naming::id_type locale)
        {
            return this->base_type::get_local(this->gid_, locale);
        }

        std::vector<naming::id_type> locals(void)
        {
            return this->base_type::locals(this->gid_);
        }
    };
    
}}

#endif
