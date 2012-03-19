////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _LUBLOCK_SERVER_HPP
#define _LUBLOCK_SERVER_HPP

/*This is the lublock class implementation header file.
This is to store data semi-contiguously.
*/

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

using std::vector;
using hpx::naming::id_type;

namespace hpx { namespace components { namespace server
{
    class HPX_COMPONENT_EXPORT lublock :
          public simple_component_base<lublock>
    {
    public:
        enum actions{
            hpl_constructBlock,
            hpl_gcorner,
            hpl_gtop,
            hpl_gleft,
            hpl_gtrail,
            hpl_getRows,
            hpl_getColumns,
            hpl_getData,
            hpl_getFuture
        };
        //constructors and destructor
        lublock(){}
        int construct_block(const int h, const int w, const int px,
            const int py, const int size, vector<vector<id_type> > gids,
            vector<vector<double> > theData);
        ~lublock(){destruct_block();}
        void destruct_block();
        int lu_gauss_corner(const int iter);
        int lu_gauss_top(const int iter);
        int lu_gauss_left(const int iter);
        int lu_gauss_trail(const int iter);
        int get_rows(){return rows;}
        int get_columns(){return columns;}
        vector<vector<double> > get_data(){return data;};
        int get_needed_future(const int type, const int iter);

        //data members
        int rows, columns, posX, posY, gridSize;
        vector<vector<double> > data;
        vector<vector<id_type> > gidList;

        //actions
        typedef hpx::actions::result_action7<lublock, int, hpl_constructBlock, int,
            int, int, int, int,vector<vector<id_type> >,vector<vector<double> >,
            &lublock::construct_block> constructBlock_action;

        typedef hpx::actions::result_action1<lublock, int, hpl_gcorner, int,
            &lublock::lu_gauss_corner> gcorner_action;

        typedef hpx::actions::result_action1<lublock, int, hpl_gtop, int,
            &lublock::lu_gauss_top> gtop_action;

        typedef hpx::actions::result_action1<lublock, int, hpl_gleft, int,
            &lublock::lu_gauss_left> gleft_action;

        typedef hpx::actions::result_action1<lublock, int, hpl_gtrail, int,
            &lublock::lu_gauss_trail> gtrail_action;

        typedef hpx::actions::result_action0<lublock, int, hpl_getRows,
            &lublock::get_rows> getRows_action;

        typedef hpx::actions::result_action0<lublock, int, hpl_getColumns,
            &lublock::get_columns> getColumns_action;

        typedef hpx::actions::result_action0<lublock, vector<vector<double> >,
            hpl_getData, &lublock::get_data> getData_action;

        typedef hpx::actions::result_action2<lublock, int, hpl_getFuture, int, int,
            &lublock::get_needed_future> getFuture_action;

        //futures
        typedef hpx::lcos::packaged_task<server::lublock::getData_action> dataFuture;
        typedef hpx::lcos::packaged_task<server::lublock::getRows_action> rowFuture;
        typedef
            hpx::lcos::packaged_task<server::lublock::getColumns_action> columnFuture;
        typedef hpx::lcos::packaged_task<server::lublock::getFuture_action> getFuture;
        typedef hpx::lcos::packaged_task<server::lublock::gcorner_action> gcFuture;
        typedef hpx::lcos::packaged_task<server::lublock::gleft_action> glFuture;
        typedef hpx::lcos::packaged_task<server::lublock::gtop_action> gtoFuture;
        typedef hpx::lcos::packaged_task<server::lublock::gtrail_action> gtrFuture;

        //the following variables require the above typedefs
        glFuture*  nextLeft;
        gtoFuture* nextTop;
        gtrFuture** nextRight;
        gtrFuture** nextBelow;
        gtrFuture** nextDagnl;

/*        //the below functions require futures and actions declared above
        int create_left_futures(const int row, const int iter,
            int brows,const vector<vector<id_type> > gidList);
        typedef actions::result_action4<lublock, int, hpl_createLeft, int, int,
            int, vector<vector<id_type> >,
            &lublock::create_left_futures> createLeftFuture_action;
        typedef lcos::packaged_task<server::lublock::createLeftFuture_action>
            createLeftFuture;
*/
    };
////////////////////////////////////////////////////////////////////////////////

    //the constructor initializes the matrix
    int lublock::construct_block(const int h, const int w, const int px,
        const int py, const int size, vector<vector<id_type> > gids,
        vector<vector<double> > theData){
        gidList = gids;
        posX = px;
        posY = py;
        gridSize = size;
        nextRight = new gtrFuture*[size];
        nextBelow = new gtrFuture*[size];
        nextDagnl = new gtrFuture*[size];
        for(int i = 0; i < h; i++){
            vector<double> row;
            data.push_back(row);
            for(int j = 0; j < w; j++){
                data[i].push_back(theData[i][j]);
        }   }
        rows = h;
        columns = w;
        return 0;
    }

    //the destructor frees the memory
    void lublock::destruct_block(){
        data.clear();
        for(int i = 0; i < gridSize; i++){
            if(nextRight[i] != 0){delete nextRight[i];}
            if(nextBelow[i] != 0){delete nextBelow[i];}
            if(nextDagnl[i] != 0){delete nextDagnl[i];}
        }
        if(nextLeft != 0){delete nextLeft;}
        if(nextTop != 0){delete nextTop;}
        free(nextRight);
        free(nextBelow);
        free(nextDagnl);
    }

    //get_needed_future gets the specified future from the component. It allows
    //a different component to retreive the value from this component's futures.
    int lublock::get_needed_future(const int type, const int iter){
        switch(type){
            case 1: return nextLeft->get_future().get();
            case 2: return nextTop->get_future().get();
            case 3: return nextBelow[iter]->get_future().get();
            case 4: return nextRight[iter]->get_future().get();
            case 5: return nextDagnl[iter]->get_future().get();
        }
        return 1;
    }

    //lugausscorner peforms gaussian elimination on the topleft corner block
    //of data that has not yet completed all of it's gaussian elimination
    //computations. Once complete, this block will need no further computations
    int lublock::lu_gauss_corner(const int iter){
    getFuture* neededPrev = 0;
    if(iter > 0){neededPrev = new getFuture(gidList[iter-1][iter-1],5, iter-1);}
    int i, j, k;
    double fFactor, factor;

    if(iter > 0){
        neededPrev->get_future().get();
        delete neededPrev;
    }
    for(i=0;i<rows;i++){
        if(data[i][i] == 0){std::cerr<<"Warning: divided by zero\n";}
        fFactor = 1/data[i][i];
        for(j=i+1;j<rows;j++){
            factor = fFactor*data[j][i];
            for(k=i+1;k<columns;k++){
                data[j][k] -= factor*data[i][k];
    }   }   }

    if(posX < gridSize-1){
        nextLeft = new glFuture(gidList[posX+1][posX], iter);
        nextTop = new gtoFuture(gidList[posX][posX+1], iter);
        nextDagnl[iter] = new gtrFuture(gidList[posX+1][posX+1], iter);
    }
    return 0;
    }

    //lugausstop performs gaussian elimination on the topmost row of blocks
    //that have not yet finished all gaussian elimination computation.
    //Once complete, these blocks will no longer need further computations
    int lublock::lu_gauss_top(const int iter){
    getFuture* neededPrev = 0;
    if(iter > 0){
        neededPrev = new getFuture(gidList[iter][posX-1], 4, iter-1);
    }
    int i,j,k;
    double fFactor, factor;
    vector<vector<double> > cornerData = dataFuture(gidList[iter][iter]).get_future().get();

    if(iter > 0){
        neededPrev->get_future().get();
        delete neededPrev;
    }
    if(posX < gridSize - 1){
        nextTop = new gtoFuture(gidList[posY][posX+1], iter);
    }
    for(i=0;i<rows;i++){
        fFactor = 1/cornerData[i][i];
        for(j=i+1;j<rows;j++){
            factor = fFactor*cornerData[j][i];
            for(k=0;k<columns;k++){
                data[j][k] -= factor*data[i][k];
    }   }   }

    return 0;
    }

    //lugaussleft performs gaussian elimination on the leftmost column of
    //blocks that have not yet finished all gaussian elimination computation.
    //Upon completion, no further computations need be done on these blocks.
    int lublock::lu_gauss_left(const int iter){
    getFuture* neededPrev = 0;
    if(iter > 0){
        neededPrev = new getFuture(gidList[posY-1][iter], 3, iter-1);
    }
    int i,j,k;
    double factor;
    double* fFactor = new double[columns];
    vector<vector<double> > cornerData = dataFuture(gidList[iter][iter]).get_future().get();

    if(iter > 0){
        neededPrev->get_future().get();
        delete neededPrev;
    }
    if(posY < gridSize - 1){
        nextLeft = new glFuture(gidList[posY+1][posX], iter);
    }
    //this first block of code finds all necessary factors early on
    //and allows for more efficient cache accesses for the second
    //block, which is where the majority of work is performed
    for(i=0;i<columns;i++){
        fFactor[i] = 1/cornerData[i][i];
        factor = fFactor[i]*data[0][i];
        for(k=i+1;k<columns;k++){
            data[0][k] -= factor*cornerData[i][k];
    }   }
    for(j=1;j<rows;j++){
        for(i=0;i<columns;i++){
            factor = fFactor[i]*data[j][i];
            for(k=i+1;k<columns;k++){
                data[j][k] -= factor*cornerData[i][k];
    }   }   }

    return 0;
    }

    //lugausstrail performs gaussian elimination on the trailing submatrix of
    //the blocks operated on during the current iteration of the Gaussian
    //elimination computations. These blocks will still require further
    //computations to be performed in future iterations.
    int lublock::lu_gauss_trail(const int iter){
    getFuture neededLeft(gidList[posY-1][iter],1,iter);
    getFuture neededTop(gidList[iter][posX-1],2,iter);
    if(iter > 0){
        getFuture* neededPrev;
        if(posX < posY){
            neededPrev = new getFuture(gidList[posY-1][posX],3,iter-1);
        }
        else if(posX > posY){
            neededPrev = new getFuture(gidList[posY][posX-1],4,iter-1);
        }
        else{neededPrev = new getFuture(gidList[posY-1][posX-1],5,iter-1);}
        neededPrev->get_future().get();
        delete neededPrev;
    }
    neededLeft.get_future().get();
    neededTop.get_future().get();

    if(posX <= posY && posY < gridSize - 1){
        nextBelow[iter] = new gtrFuture(gidList[posY+1][posX], iter);
    }
    if(posX == posY && posY < gridSize - 1){
        nextDagnl[iter] = new gtrFuture(gidList[posY+1][posX+1], iter);
    }
    if(posX >= posY && posX < gridSize - 1){
        nextRight[iter] = new gtrFuture(gidList[posY][posX+1], iter);
    }

    int i,j,k;
    double fFactor, factor;
    vector<vector<double> > cornerData = dataFuture(gidList[iter][iter]).get_future().get();
    vector<vector<double> > leftData = dataFuture(gidList[posY][iter]).get_future().get();
    vector<vector<double> > topData = dataFuture(gidList[iter][posX]).get_future().get();
    const int size = cornerData.size();

    for(i=0;i<size;i++){
        fFactor = 1/cornerData[i][i];
        for(j=0;j<rows;j++){
            factor = fFactor*leftData[j][i];
            for(k=0;k<columns;k++){
                data[j][k] -= factor*topData[i][k];
    }   }   }

    return 0;
    }
}}}

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<
        std::vector<std::vector<double> >
    >::set_result_action,
    base_lco_with_value_set_result_vector_vector_double);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::lublock::constructBlock_action,HPLconstructBlock_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::lublock::gcorner_action,HPLgcorner_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::lublock::gtop_action,HPLgtop_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::lublock::gleft_action,HPLgleft_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::lublock::gtrail_action,HPLgtrail_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::lublock::getRows_action,HPLgetRows_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::lublock::getColumns_action,HPLgetColumns_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::lublock::getData_action,HPLgetData_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::lublock::getFuture_action,HPLgetFuture_action);
#endif
