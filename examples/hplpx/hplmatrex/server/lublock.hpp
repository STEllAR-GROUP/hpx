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
            hpl_getData
        };
        //constructors and destructor
        lublock(){}
        int construct_block(const int h, const int w, const naming::id_type ctrl,
            std::vector<std::vector<double> > theData);
        ~lublock(){destruct_block();}
        void destruct_block();
        int lu_gauss_corner();
        int lu_gauss_top(const naming::id_type iterCorner);
        int lu_gauss_left(const naming::id_type iterCorner);
        int lu_gauss_trail(const int size, const naming::id_type iterCorner,
            const naming::id_type iterLeft, const naming::id_type iterTop);
        int get_rows(){return rows;}
        int get_columns(){return columns;}
        std::vector<std::vector<double> > get_data(){return data;};

        //data members
        int rows;
        int columns;
        double* workSpace;
        naming::id_type controller;
        std::vector<std::vector<double> > data;

        //actions
        typedef actions::result_action4<lublock, int, hpl_constructBlock,
            int, int, naming::id_type, std::vector<std::vector<double> >,
            &lublock::construct_block> constructBlock_action;
        typedef actions::result_action0<lublock, int, hpl_gcorner,
            &lublock::lu_gauss_corner> gcorner_action;
        typedef actions::result_action1<lublock, int, hpl_gtop,
            naming::id_type, &lublock::lu_gauss_top> gtop_action;
        typedef actions::result_action1<lublock, int, hpl_gleft,
            naming::id_type, &lublock::lu_gauss_left> gleft_action;
        typedef actions::result_action4<lublock, int, hpl_gtrail, int,
            naming::id_type, naming::id_type, naming::id_type, 
            &lublock::lu_gauss_trail> gtrail_action;
        typedef actions::result_action0<lublock, int, hpl_getRows,
            &lublock::get_rows> getRows_action;
        typedef actions::result_action0<lublock, int, hpl_getColumns,
            &lublock::get_columns> getColumns_action;
        typedef actions::result_action0<lublock, std::vector<std::vector<double> >,
            hpl_getData, &lublock::get_data> getData_action;

        //futures
        typedef lcos::eager_future<server::lublock::getData_action> dataFuture;
        typedef lcos::eager_future<server::lublock::getRows_action> rowFuture;
        typedef lcos::eager_future<server::lublock::getColumns_action> columnFuture;
        typedef lcos::eager_future<server::lublock::gcorner_action> gcFuture;
        typedef lcos::eager_future<server::lublock::gtop_action> gtopFuture;
        typedef lcos::eager_future<server::lublock::gleft_action> glFuture;
        typedef lcos::eager_future<server::lublock::gtrail_action> gtrFuture;
    };
////////////////////////////////////////////////////////////////////////////////

    //the constructor initializes the matrix
    int lublock::construct_block(const int h, const int w,
        const naming::id_type ctrl, std::vector<std::vector<double> > theData){
        controller = ctrl;
//        workSpace = (double*) std::malloc((8+h*w)*sizeof(double));
//        data = (double**) std::malloc(h*sizeof(double*));
        for(int i = 0; i < h; i++){
            std::vector<double> row;
            data.push_back(row);
            for(int j = 0; j < w; j++){
                data[i].push_back(theData[i][j]);
        }   }
        rows = h;
        columns = w;
//        for(int i = 0; i < h; i++){
//            data[i] = workSpace + 8 + i*w;
//        }
        return 1;
    }

    //the destructor frees the memory
    void lublock::destruct_block(){
        free(workSpace);
        data.clear();
    }

    //lugausscorner peforms gaussian elimination on the topleft corner block
    //of data that has not yet completed all of it's gaussian elimination
    //computations. Once complete, this block will need no further computations
    int lublock::lu_gauss_corner(){
    int i, j, k;
    double fFactor, factor;

    for(i=0;i<rows;i++){
        if(data[i][i] == 0){std::cerr<<"Warning: divided by zero\n";}
        fFactor = 1/data[i][i];
        for(j=i+1;j<rows;j++){
            factor = fFactor*data[j][i];
            for(k=i+1;k<columns;k++){
                data[j][k] -= factor*data[i][k];
    }   }   }
    return 0;
    }

    //lugausstop performs gaussian elimination on the topmost row of blocks
    //that have not yet finished all gaussian elimination computation.
    //Once complete, these blocks will no longer need further computations
    int lublock::lu_gauss_top(const naming::id_type iterCorner){
    int i,j,k;
    double fFactor, factor;
    std::vector<std::vector<double> > cornerData = dataFuture(iterCorner).get();

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
    int lublock::lu_gauss_left(const naming::id_type iterCorner){
    int i,j,k;
    double factor;
    double* fFactor = (double*) std::malloc(columns*sizeof(double));
    std::vector<std::vector<double> > cornerData = dataFuture(iterCorner).get();

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
    int lublock::lu_gauss_trail(const int size,const naming::id_type iterCorner,
        const naming::id_type iterLeft, const naming::id_type iterTop){
    int i,j,k;
    double fFactor, factor;
    std::vector<std::vector<double> > cornerData = dataFuture(iterCorner).get();
    std::vector<std::vector<double> > leftData = dataFuture(iterLeft).get();
    std::vector<std::vector<double> > topData = dataFuture(iterTop).get();

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
#endif
