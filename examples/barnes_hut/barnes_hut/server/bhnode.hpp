////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _BARNES_HUT_OCTNODE_SERVER_HPP
#define _BARNES_HUT_OCTNODE_SERVER_HPP

/*This is the bhnode class implementation header file.
This is to store data semi-contiguously.
*/

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>                             
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/lcos/mutex.hpp>

#include <boost/serialization/vector.hpp>

using std::vector;
using hpx::naming::id_type;

namespace hpx { namespace components
{
    class bhnode;
}}

namespace hpx { namespace components { namespace server
{
    class HPX_COMPONENT_EXPORT bhnode :
          public simple_component_base<bhnode>
    {
    public:
        enum actions{
            hpl_constNode,
            hpl_cnstNode2,
            hpl_setBounds,
            hpl_findPoint,
            hpl_insrtNode,
            hpl_updateChd,
            hpl_simulate,
            hpl_printTree
        };
        //constructors and destructor
        bhnode(){}
        int construct_node(const id_type gid, const vector<double> dat,
            const bool root);
        int construct_node(const id_type gid, const id_type insertPoint,
            const vector<double> bounds);
        ~bhnode(){destruct_node();}
        void destruct_node();
        vector<double> find_insert_point(const vector<double> nodep,
            const double nodem, const id_type nodeGid);
        vector<double> insert_node(const vector<double> nodep,
            const double nodem, const id_type nodeGid, const int octant);
        int update_child(const int octant, const id_type newChild,
            const bool leaf, const vector<double> childPosition);
        int set_boundaries(const id_type parId, const vector<double> bounds);
        int run(const id_type controllerGid, const vector<double> info);

    private:
        int find_octant(const vector<double> nodep, const vector<double> center);
        vector<double> calculate_subboundary(const int octant,
            vector<double> subboundary);
        void update_com();

    public:
        //for debugging
        int print_tree(const int level, const int depth);

        //data members
        double boundary[6], pos[3], vel[3], mass, com[3];
        double childMasses[8], childPos[8][3];
        bool isLeaf, isRoot, hasChild[8], childIsLeaf[8];
        id_type _gid, parent, controller;
        id_type child[8];
        int numIters;
        double dtime, eps, tolerance;
        double halfdt, softening, invTolerance;
        hpx::lcos::mutex theMutex;

        //actions
        //first constructor action(particles and root)
        typedef hpx::actions::result_action3<bhnode, int, hpl_constNode,
            id_type, vector<double>, bool, &bhnode::construct_node>
            constNodeAction;
        //second constructor action(intermediate nodes)
        typedef hpx::actions::result_action3<bhnode, int, hpl_cnstNode2, id_type,
            id_type, vector<double>, &bhnode::construct_node> cnstNodeAction2;
        //setting bounds action
        typedef hpx::actions::result_action2<bhnode, int, hpl_setBounds, id_type,
            vector<double>, &bhnode::set_boundaries> setBoundsAction;
        //find insert point action
        typedef hpx::actions::result_action3<bhnode, vector<double>, hpl_findPoint,
            vector<double>, double, id_type, &bhnode::find_insert_point>
            findInPntAction;
        //inserting node action
        typedef hpx::actions::result_action4<bhnode, vector<double>, hpl_insrtNode,
            vector<double>, double, id_type, int, &bhnode::insert_node>
            insrtNodeAction;
        //update child action
        typedef hpx::actions::result_action4<bhnode, int, hpl_updateChd,
            int, id_type, bool, vector<double>, 
            &bhnode::update_child> updatChldAction;
        //run the simulation action
        typedef hpx::actions::result_action2<bhnode, int, hpl_simulate,
            id_type, vector<double>, &bhnode::run> runSimAction;

        //for debugging
        typedef hpx::actions::result_action2<bhnode, int, hpl_printTree,
            int, int, &bhnode::print_tree> printTreeAction;

        //futures
        typedef hpx::lcos::eager_future<server::bhnode::constNodeAction> constFuture;
        typedef hpx::lcos::eager_future<server::bhnode::cnstNodeAction2> cnstFuture2;
        typedef hpx::lcos::eager_future<server::bhnode::setBoundsAction> boundFuture;
        typedef hpx::lcos::eager_future<server::bhnode::insrtNodeAction> iNodeFuture;
        typedef hpx::lcos::eager_future<server::bhnode::findInPntAction> inPntFuture;
        typedef hpx::lcos::eager_future<server::bhnode::updatChldAction> childFuture;
        typedef hpx::lcos::eager_future<server::bhnode::runSimAction>    runFuture;
        //for debugging
        typedef hpx::lcos::eager_future<server::bhnode::printTreeAction> printFuture;

        iNodeFuture* insertFuture[8];
    };
////////////////////////////////////////////////////////////////////////////////
}}}
#endif
/*std::cout<<"adding "<<nodeGid<<" to octant "<<octant<<"\n";
std::cout<<_gid<<" is root? "<<isRoot<<" isLeaf? "<<isLeaf;
if(isRoot)std::cout<<"\n";else{std::cout<<" parent: "<<parent<<"\n";}
for(int i=0;i<8;i++){
if(hasChild[i]){std::cout<<child[i]<<" ";}
else{std::cout<<i<<" ";}}
for(int i=0;i<8;i++){std::cout<<childIsLeaf[i]<<"/t";}
std::cout<<"\n\n";*/
