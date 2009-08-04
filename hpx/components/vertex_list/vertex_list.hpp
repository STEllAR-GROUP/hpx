//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_VERTEX_LIST_MAY_18_2008_0822AM)
#define HPX_COMPONENTS_VERTEX_LIST_MAY_18_2008_0822AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/vertex_list.hpp"

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a vertex_list class is the client side representation of a 
    /// specific \a server#vertex_list component
    class vertex_list 
      : public client_base<vertex_list, stubs::vertex_list>
    {
    private:
        typedef client_base<vertex_list, stubs::vertex_list> base_type;

    public:
        /// Create a client side representation for the existing
        /// \a server#vertex_list instance with the given global id \a gid.
        vertex_list(naming::id_type gid, bool freeonexit = true) 
          : base_type(gid, freeonexit)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the vertex_list
        int init(components::component_type item_type, std::size_t order)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init(gid_, item_type, order);
        }

        int size(void)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::size(gid_);
        }

        naming::id_type at_index(const int index)
        {
        	BOOST_ASSERT(gid_);
        	return this->base_type::at_index(gid_, index);
        }
    };
    
}}

#endif
