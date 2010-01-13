//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DISTRIBUTED_SET_MAY_18_2008_0822AM)
#define HPX_COMPONENTS_DISTRIBUTED_SET_MAY_18_2008_0822AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/distributed_set.hpp"

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a distributed_set class is the client side representation of a
    /// specific \a server#distributed_set component
    ///
    /// The purpose of the distributed list component is simply to wrap local
    /// non-component list types, to give them GIDs
    ///
    /// Remember that Item must be serializable

    template <typename Item>
    class distributed_set
      : public client_base<
                   distributed_set<Item>, stubs::distributed_set<Item>
               >
    {
    private:
        typedef client_base<
                    distributed_set<Item>, stubs::distributed_set<Item>
                > base_type;

    public:
        distributed_set()
          : base_type(naming::invalid_id, true)
        {}

        distributed_set(naming::id_type gid, bool freeonexit = true)
          : base_type(gid, freeonexit)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        int init(int num_items)
        {
            return this->base_type::init(this->gid_, num_items);
        }

        naming::id_type add_item(naming::id_type item=naming::invalid_id)
        {
            return this->base_type::add_item(this->gid_, item);
        }

        naming::id_type get_local(naming::id_type locale)
        {
            return this->base_type::get_local(this->gid_, locale);
        }

        std::vector<naming::id_type> locals(void)
        {
            return this->base_type::locals(this->gid_);
        }

        int size(void)
        {
            return this->base_type::size(this->gid_);
        }
    };
    
}}

#endif
