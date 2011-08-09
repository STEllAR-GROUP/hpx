//////////////////////////////////////////////////////////////////////////////// 
//  Copyright (C) 2011 Daniel Kogler                                                
//                                                                               
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)        
////////////////////////////////////////////////////////////////////////////////
using hpx::applier::get_applier;
namespace hpx { namespace components { namespace server
{
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
        const vector<double> bounds){
        _gid = gid;
        parent = insertPoint;
        isLeaf = isRoot = false;
        for(int i = 0; i < 8; i++){
            hasChild[i] = false;
            childIsLeaf[i] = false;
        }
        for(int i = 0; i < 6; i++){boundary[i] = bounds[i];}
        for(int i = 0; i < 3; i++){pos[i] = (bounds[i]+bounds[i+3])*.5;}
        return 0;
    }

    //the destructor frees the memory
    void bhnode::destruct_node(){
    }

    int bhnode::run(const id_type controllerGid, const vector<double> info){
        controller = controllerGid;
        numIters = info[0];
        dtime = info[1];
        eps = info[2];
        tolerance = info[3];
        halfdt = .5 * dtime;
        softening = eps * eps;
        invTolerance = 1 / (tolerance * tolerance);
        return 0;
    }

    vector<double> bhnode::find_insert_point(const vector<double> nodep,
        const double nodem, const id_type nodeGid){
        vector<double> tempPos(pos,pos+3);
        vector<double> returnValues;
        const int octant = find_octant(nodep, tempPos);
        int i;
        inPntFuture* waitFuture = 0;
        hpx::lcos::mutex::scoped_lock l(theMutex);
        if(hasChild[octant] && !childIsLeaf[octant]){
            waitFuture = new inPntFuture(child[octant],nodep,nodem,nodeGid);
            returnValues = waitFuture->get();
            childMasses[octant] = returnValues[0];
            for(i=0;i<3;i++){childPos[octant][i] = returnValues[i+1];}
        }
        else{
            if(hasChild[octant]){
                insertFuture[octant]->get();
            }
            insertFuture[octant] = new iNodeFuture(_gid, nodep,
                                                   nodem, nodeGid, octant);
            returnValues = insertFuture[octant]->get();
            hasChild[octant] = true;
        }
        update_com();
        returnValues.clear();
        returnValues.push_back(mass);
        for(i = 0; i < 3; i++){returnValues.push_back(com[i]);}
        return returnValues;
    }

    vector<double> bhnode::insert_node(const vector<double> nodep,
        const double nodem, const id_type nodeGid, const int octant){
        int i;
        vector<double> returnValues;
        const vector<double> tempBoundary(boundary,boundary+6);
        const vector<double> subboundary =
                                    calculate_subboundary(octant, tempBoundary);
        if(hasChild[octant]){
            vector<double> tP;
            components::bhnode* newNode = new components::bhnode;
            newNode->create(get_applier().get_runtime_support_gid());
            cnstFuture2 makeFuture = newNode->construct_node(_gid, subboundary);
            makeFuture.get();
            inPntFuture waitFuture1(newNode->get_gid(), nodep, nodem, nodeGid);
            for(i = 0; i < 3; i++){tP.push_back(childPos[octant][i]);}
            waitFuture1.get();
            inPntFuture waitFuture2(newNode->get_gid(), tP,
                childMasses[octant], child[octant]);
            childIsLeaf[octant] = false;
            child[octant] = newNode->get_gid();
            returnValues = waitFuture2.get();
            for(i = 0; i < 3; i++){
                childPos[octant][i] = returnValues[i+1];
            }
            childMasses[octant] = returnValues[0];
        }
        else{
            boundFuture waitFuture(nodeGid, _gid, subboundary);
            child[octant] = nodeGid;
            hasChild[octant] = true;
            childIsLeaf[octant] = true;
            for(i = 0; i < 3; i++){childPos[octant][i] = nodep[i];}
            childMasses[octant] = nodem;
            waitFuture.get();
        }
        returnValues.clear();
        returnValues.push_back(mass);
        for(i = 0; i < 3; i++){returnValues.push_back(com[i]);}
        return returnValues; 
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
        const bool leaf, const vector<double> childPosition){
        child[octant] = newChild;
        hasChild[octant] = true;
        childIsLeaf[octant] = leaf;
        for(int i = 0; i < 3; i++){childPos[octant][i] = childPosition[i];}
        update_com();
        return 0;
    }

    void bhnode::update_com(){
        mass = com[0] = com[1] = com[2] = 0;
        for(int i = 0; i < 8; i++){
            if(hasChild[i]){
                mass += childMasses[i];
                com[0] += childMasses[i]*childPos[i][0];
                com[1] += childMasses[i]*childPos[i][1];
                com[2] += childMasses[i]*childPos[i][2];
            }
        }
        com[0] /= mass;
        com[1] /= mass;
        com[2] /= mass;
    }

    int bhnode::print_tree(const int level, const int depth){
        int i;
        if(level > 0){std::cout<<"  ";}
        for(i = 1; i < level; i++){std::cout<<"| ";}
        if(level > 0){std::cout<<"|_";}
        std::cout<<"m: "<<mass<<"  p:("<<pos[0]<<", "<<pos[1]<<", "<<pos[2]
                 <<")  com:("<<com[0]<<", "<<com[1]<<", "<<com[2]<<")\n";
        for(i = 0; i < 8; i++){
            if(hasChild[i]){
                printFuture future(child[i],level+1, depth);
                future.get();
            }
        }
        return 0;
    }
}}}
