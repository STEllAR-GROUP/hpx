#ifndef _BH_STUB_HPP_01182011
#define _BH_STUB_HPP_01182011


#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <examples/bh/bh/server/bh.hpp>


namespace hpx { namespace components { namespace stubs
{
    struct IntrTreeNode : stub_base<server::IntrTreeNode>
    {
        static void newNode(naming::id_type gid, double px, double py, double pz) 
        {
            applier::apply<server::IntrTreeNode::newNode_action>(gid, px, py, pz);
        }

    };

}}}
#endif
