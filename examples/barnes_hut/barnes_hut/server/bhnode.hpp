////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
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

#include <boost/serialization/vector.hpp>

using std::vector;
using hpx::naming::id_type;

struct region_path{
    bool isPath;
    id_type insertPoint;
    vector<id_type> child;
    vector<vector<double> > subboundaries, childData;
    vector<int> octants;
    template<typename Archive>
    void serialize(Archive& ar, unsigned int version){
        ar & isPath;
        ar & insertPoint;
        ar & child;
        ar & subboundaries;
        ar & childData;
        ar & octants;
    }
};

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
            hpl_insrtNode,
            hpl_updateChd
        };
        //constructors and destructor
        bhnode(){}
        int construct_node(const id_type gid, const vector<double> dat,
            const bool root);
        int construct_node(const id_type gid, const id_type insertPoint,
            const vector<double> bounds, const vector<id_type> child, 
            const vector<vector<double> > childData, const vector<int> octants);
        ~bhnode(){destruct_node();}
        void destruct_node();
        region_path insert_node(const vector<double> nodep, const double nodem,
            const id_type nodeGid);
        region_path build_subregions(int octant, vector<double> subboundary,
            vector<double> nodep, double nodem, const id_type node1Gid,
            const id_type node2Gid);
        int update_child(const int octant, const id_type newChild,
            const bool leaf);
        int set_boundaries(const id_type parId, const vector<double> bounds);
        int find_octant(const vector<double> nodep, const vector<double> center);
        vector<double> calculate_subboundary(const int octant,
            vector<double> subboundary);
        void update_com();

        //data members
        double boundary[6], pos[3], vel[3], mass, com[3];
        double childMasses[8], childPos[8][3];
        bool isLeaf, isRoot, hasChild[8], childIsLeaf[8];
        id_type _gid;
        id_type parent;
        id_type child[8];

        //actions
        typedef actions::result_action3<bhnode, int, hpl_constNode,
            id_type, vector<double>, bool, &bhnode::construct_node>
            constNodeAction;
        typedef actions::result_action6<bhnode, int, hpl_cnstNode2, id_type,
            id_type, vector<double>, vector<id_type>, vector<vector<double> >,
            vector<int>, &bhnode::construct_node> cnstNodeAction2;
        typedef actions::result_action2<bhnode, int, hpl_setBounds, id_type,
            vector<double>, &bhnode::set_boundaries> setBoundsAction;
        typedef actions::result_action3<bhnode, region_path, hpl_insrtNode,
            vector<double>, double, id_type, &bhnode::insert_node>
            insrtNodeAction;
        typedef actions::result_action3<bhnode, int, hpl_updateChd, int,
            id_type, bool, &bhnode::update_child> updatChldAction;

        //futures
        typedef lcos::eager_future<server::bhnode::constNodeAction> constFuture;
        typedef lcos::eager_future<server::bhnode::cnstNodeAction2> cnstFuture2;
        typedef lcos::eager_future<server::bhnode::setBoundsAction> boundFuture;
        typedef lcos::eager_future<server::bhnode::insrtNodeAction> iNodeFuture;
        typedef lcos::eager_future<server::bhnode::updatChldAction> childFuture;
    };
////////////////////////////////////////////////////////////////////////////////

/**********NEED TO ADD ISROOT TO FIRST CONSTRUCTOR****************/
    //the constructors
    //this first constructor is for all particles and the root node
    int bhnode::construct_node(const id_type gid, const vector<double> dat,
        const bool root){
        for(int i = 0; i < 3; i++){
            pos[i] = com[i] = dat[i+1];
            vel[i] = dat[i+4];
        }
        mass = dat[0];
        _gid = gid;
        for(int i = 0; i < 8; i++){hasChild[i] = childIsLeaf[i] = false;}
        isLeaf = !root;
        isRoot = root;
        return 0;
    }
    //this constructor is for all intermediate "region" nodes
    int bhnode::construct_node(const id_type gid, const id_type insertPoint,
        const vector<double> bounds, const vector<id_type> children, 
        const vector<vector<double> > data, const vector<int> octants){
        _gid = gid;
        parent = insertPoint;
        isLeaf = isRoot = false;
        for(int i = 0; i < 6; i++){boundary[i] = bounds[i];}
        for(int i = 0; i < 3; i++){
            pos[i] = (boundary[i] + boundary[i+3])*.5;
        }
        if(octants.size() == 1){
            for(int i = 0; i < 8; i++){
                if(i == octants[0]){
                    hasChild[i] = true;
                    child[i] = children[0];
                    for(int j = 0; j < 3; j++){childPos[i][j] = data[3][j];}
                    childMasses[i] = data[2][0] + data[2][1];
                }
                else{hasChild[i] = false;}
                childIsLeaf[i] = false;
            }
        }
        else{
            for(int i = 0; i < 8; i++){
                if(i == octants[octants.size()-2]){
                    hasChild[i] = false;
                    childIsLeaf[i] = false;
                }
            }
            insert_node(data[0], data[2][0], children[0]);
            insert_node(data[1], data[2][1], children[1]);
        }
        mass = data[2][0] + data[2][1];
        for(int i = 0; i < 3; i++){
            com[i] = (data[0][i]*data[2][0] + data[1][i]*data[2][1])*.5;
        }
        return 0;
    }

    //the destructor frees the memory
    void bhnode::destruct_node(){
    }

    region_path bhnode::insert_node(const vector<double> nodep,
        const double nodem, const id_type nodeGid){
        vector<double> tempPos(pos,pos+3);
        const int octant = find_octant(nodep, tempPos);
        region_path insertPath;
        insertPath.isPath = false;
/*std::cout<<"adding "<<nodeGid<<" to octant "<<octant<<"\n";
std::cout<<_gid<<" is root? "<<isRoot<<" isLeaf? "<<isLeaf;
if(isRoot)std::cout<<"\n";else{std::cout<<" parent: "<<parent<<"\n";}
for(int i=0;i<8;i++){
if(hasChild[i]){std::cout<<child[i]<<" ";}
else{std::cout<<i<<" ";}}
for(int i=0;i<8;i++){std::cout<<childIsLeaf[i]<<"/t";}
std::cout<<"\n\n";*/
        if(hasChild[octant] && !childIsLeaf[octant]){
            iNodeFuture waitFuture(child[octant], nodep, nodem, nodeGid);
            insertPath = waitFuture.get();
        }
        else if(hasChild[octant]){
            vector<double> tempBoundary(boundary,boundary+6);
            vector<double> tP(childPos[octant],childPos[octant]+3);
            vector<double> cM;
            vector<double> subboundary =
                                    calculate_subboundary(octant, tempBoundary);
            insertPath = build_subregions(octant, subboundary, nodep, nodem,
                                          nodeGid, child[octant]);
            insertPath.isPath = true;
            insertPath.insertPoint = _gid;
            insertPath.child.push_back(nodeGid);
            insertPath.child.push_back(child[octant]);
            insertPath.childData.push_back(nodep);
            insertPath.childData.push_back(tP);
            cM.push_back(nodem);
            cM.push_back(childMasses[octant]);
            insertPath.childData.push_back(cM);
            childIsLeaf[octant] = false;
        }
        else{
            vector<double> tempBoundary(boundary,boundary+6);
            vector<double> subboundary = 
                                    calculate_subboundary(octant, tempBoundary);
            boundFuture waitFuture(nodeGid, _gid, subboundary);
            child[octant] = nodeGid;
            hasChild[octant] = true;
            childIsLeaf[octant] = true;
            for(int i = 0; i < 3; i++){childPos[octant][i] = nodep[i];}
            childMasses[octant] = nodem;
            waitFuture.get();
        }
        update_com();
        return insertPath; 
    }

    //can't actually create new nodes from here, so we send back the regions
    //for the new nodes as well as the id of the region node from which to
    //create tne new region nodes and then create them with the controller.
    //the below function recursively calculates the boundaries for all
    //region nodes that need to be created
    region_path bhnode::build_subregions(int octant,
        vector<double> subboundary, vector<double> nodep, double nodem,
        const id_type node1Gid, const id_type node2Gid){
        int octant1, octant2;
        region_path path;
        vector<double> centerPos;
        vector<double> chPos(childPos[octant],childPos[octant]+3);

        path.subboundaries.push_back(subboundary);
        path.octants.push_back(octant);
        for(int i = 0; i < 3; i++){
            centerPos.push_back((subboundary[i]+subboundary[i+3])*.5);
        }

        octant1 = find_octant(nodep,centerPos);
        octant2 = find_octant(chPos,centerPos);
        if(octant1 == octant2){
            vector<double> tempBoundary = 
                calculate_subboundary(octant1, subboundary);
            region_path tempPath = build_subregions(octant1, tempBoundary, nodep,
                                                    nodem, node1Gid, node2Gid);
            for(int i = 0; i < (int)tempPath.subboundaries.size(); i++){
                path.subboundaries.push_back(tempPath.subboundaries[i]);
                path.octants.push_back(tempPath.octants[i]);
            }
        }
        else{
            path.octants.push_back(octant1);
            path.octants.push_back(octant2);
        }
        return path;
    }

    int bhnode::find_octant(const vector<double> nodep,
        const vector<double> center){
        if(nodep[0] > center[0]){
            if(nodep[1] > center[1]){
                if(nodep[2] > center[2]){return 0;}
                else{return 4;}
            }
            else{
                if(nodep[2] > center[2]){return 3;}
                else{return 7;}
        }   }
        else{
            if(nodep[1] > center[1]){
                if(nodep[2] > center[2]){return 1;}
                else{return 5;}
            }
            else{
                if(nodep[2] > center[2]){return 2;}
                else{return 6;}
        }   }
    }

    vector<double> bhnode::calculate_subboundary(const int octant,
        vector<double> subboundary){
        if(octant == 0 || octant == 3 || octant == 4 || octant == 7){
            subboundary[3] = (subboundary[0]+subboundary[3])*.5;
            subboundary[0] = subboundary[0];
        }
        else{
            subboundary[0] = (subboundary[0]+subboundary[3])*.5;
            subboundary[3] = subboundary[3];
        }            
        if(octant == 0 || octant == 1 || octant == 4 || octant == 5){
            subboundary[4] = (subboundary[1]+subboundary[4])*.5;
            subboundary[1] = subboundary[1];
        }
        else{
            subboundary[1] = (subboundary[1]+subboundary[4])*.5;
            subboundary[4] = subboundary[4];
        }            
        if(octant == 0 || octant == 1 || octant == 2 || octant == 3){
            subboundary[5] = (subboundary[2]+subboundary[5])*.5;
            subboundary[2] = subboundary[2];
        }
        else{
            subboundary[2] = (subboundary[2]+subboundary[5])*.5;
            subboundary[5] = subboundary[5];
        }
        return subboundary;
    }


    int bhnode::set_boundaries(const id_type parId, const vector<double> bounds){
        if(parId == _gid){isRoot = true;}
        else{parent = parId;}
        for(int i = 0; i < 6; i++){
            boundary[i] = bounds[i];
        }
        if(!isLeaf){
            pos[0] = (boundary[0]+boundary[3])*.5;
            pos[1] = (boundary[1]+boundary[4])*.5;
            pos[2] = (boundary[2]+boundary[5])*.5;
        }
        return 0;
    }

    int bhnode::update_child(const int octant, const id_type newChild,
        const bool leaf){
        child[octant] = newChild;
        hasChild[octant] = true;
        childIsLeaf[octant] = leaf;
        update_com();
        return 0;
    }

    void bhnode::update_com(){}

}}}
#endif
