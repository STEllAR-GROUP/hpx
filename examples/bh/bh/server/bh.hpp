#ifndef _BH_SERVER_HPP_01182011
#define _BH_SERVER_HPP_01182011

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>


namespace hpx { namespace components { namespace server 
{
    class IntrTreeNode  
        : public components::detail::managed_component_base<IntrTreeNode>
    { 
    
    //private:
    //  IntrTreeNode *edge_link; 
    //  static IntrTreeNode *tree_head, *free_node_list;
    public:
        enum {
            newNode_func_code = 0,
        } func_codes;
        int node_type;
        double mass;
        double p[3];
        //static IntrTreeNode *newNode(double pos_buf[]); 
        
        IntrTreeNode()
        { 
            node_type = 0;
            mass = 0.0;
            p[0] = 0.0;
            p[1] = 0.0;
            p[2] = 0.0;
        }


        int newNode(double px, double py, double pz) 
        {

        //    IntrTreeNode temp_node; /* Declare the temporary node of type TreeNodeI */
            node_type=0;
            //temp_node= CELL;  /* Set the node type to CELL (since an intermediate node is being created */
            mass = 0.0;        /* intermediate node have no mass. Center of Mass is populated here later*/
            p[0] = px; /* Set the position of the cell to the position passed in*/
            p[1] = py;
            p[2] = pz;

            return 0;
        }

        typedef hpx::actions::result_action3<
            IntrTreeNode, int, newNode_func_code, double, double, double,&IntrTreeNode::newNode
        > newNode_action;

    };
}}}
#endif
