#ifndef _BH_STUB_HPP_01182011
#define _BH_STUB_HPP_01182011


#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <examples/bh/bhtree/server/bhtree.hpp>


namespace hpx { namespace components { namespace stubs
{
    struct IntrTreeNode : stub_base<server::IntrTreeNode>
    {
        static void newNode(naming::id_type gid, double px, double py, double pz) 
        {
            applier::apply<server::IntrTreeNode::newNode_action>(gid, px, py, pz);
        }
        
        static void print(naming::id_type gid) 
        {
            applier::apply<server::IntrTreeNode::print_action>(gid);
        }
    };

}}}
#endif
