//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BH_COMPONENTS_01192011_1200)
#define BH_COMPONENTS_0119201_1200

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/bhtree.hpp"


namespace hpx { namespace components 
{

    class IntrTreeNode 
      : public client_base<IntrTreeNode, stubs::IntrTreeNode>
    {
        typedef 
            client_base<IntrTreeNode, stubs::IntrTreeNode> 
        base_type;
    public:
        IntrTreeNode() 
        {}

        ~IntrTreeNode() 
        {}

        void newNode (double px, double py, double pz ) 
        {
            BOOST_ASSERT(gid_);
            this->base_type::newNode(gid_, px, py, pz);
        }
        /// Print the current value of the accumulator
        void print() 
        {
            BOOST_ASSERT(gid_);
            this->base_type::print(gid_);
        }

    };
}}
#endif
