//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_VERTEX_MAY_18_2008_0822AM)
#define HPX_COMPONENTS_VERTEX_MAY_18_2008_0822AM

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <hpx/components/graph/stubs/vertex.hpp>

//typedef std::vector<std::pair<naming::id_type, int> > partial_edge_set_type;

namespace hpx { namespace components 
{
    typedef int count_t;
    
    ///////////////////////////////////////////////////////////////////////////
    /// The \a vertex class is the client side representation of a
    /// specific \a server#vertex component
    class vertex
      : public client_base<vertex, stubs::vertex>
    {
        typedef client_base<vertex, stubs::vertex> base_type;

    public:
        vertex()
        {}

        /// Create a client side representation for the existing
        /// \a server#vertex instance with the given global id \a gid.
        vertex(naming::id_type gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the vertex
        int init(count_t order) 
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init(gid_, order);
        }

        int label(void)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::label(gid_);
        }

        int add_edge(naming::id_type v_g, int label)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::add_edge(gid_, v_g, label);
        }

        partial_edge_set_type
        out_edges(void)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::out_edges(gid_);
        }

        int
        out_degree(void)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::out_degree(gid_);
        }
    };
    
}}

#endif
