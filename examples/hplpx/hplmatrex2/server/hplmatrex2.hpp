////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _HPLMATREX2_SERVER_HPP
#define _HPLMATREX2_SERVER_HPP

/*This is the HPLMatreX2 class implementation header file.
In order to keep things simple, only operations necessary
to to perform LUP decomposition are declared, which is
basically just constructors, assignment operators,
a destructor, and access operators.
*/

#include <time.h>

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/mutex.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include <boost/foreach.hpp>
#include <boost/random/linear_congruential.hpp>
#include "LUblock.hpp"

namespace hpx { namespace components { namespace server
{
    class HPX_COMPONENT_EXPORT HPLMatreX2 : public simple_component_base<HPLMatreX2>
    {
    public:
    //enumerate all of the actions that will(or can) be employed
        enum actions{
                hpl_construct,
                hpl_destruct,
                hpl_assign,
                hpl_get,
                hpl_set,
                hpl_solve,
        hpl_swap,        
        hpl_gmain,
        hpl_mintrail,
        hpl_bsubst,
        hpl_check
        };

    //constructors and destructor
    HPLMatreX2(){}
    int construct(naming::id_type gid, unsigned int h, unsigned int w,
        unsigned int ab, unsigned int bs);
    ~HPLMatreX2(){destruct();}
    void destruct();

    //operators for assignment and leftdata access
    double get(unsigned int row, unsigned int col);
    void set(unsigned int row, unsigned int col, double val);

    //functions for manipulating the matrix
    double LUsolve();

    private:
    void allocate();
    int assign(unsigned int row, unsigned int offset, bool complete, unsigned int seed);
    void pivot();
    int swap(unsigned int brow, unsigned int bcol);
    void LUgausscorner(unsigned int iter);
    void LUgausstop(unsigned int iter, unsigned int bcol);
    void LUgaussleft(unsigned int brow, unsigned int iter);
    void LUgausstrail(unsigned int brow, unsigned int bcol, unsigned int iter);
    int minitrail(unsigned int brow, unsigned int bcol, unsigned int iter, unsigned int i);
    int LUgaussmain(unsigned int brow, unsigned int bcol, int iter, unsigned int type);
    int LUbacksubst();
    double checksolve(unsigned int row, unsigned int offset, bool complete);
    void print();
    void print2();

    int rows;            //number of rows in the matrix
    int brows;            //number of rows of blocks in the matrix
    int columns;            //number of columns in the matrix
    int bcolumns;            //number of columns of blocks in the matrix
    int allocblock;            //reflects amount of allocation done per thread
    int blocksize;            //reflects amount of computation per thread
    naming::id_type _gid;        //the instances gid
    LUblock*** datablock;        //stores the data being operated on
    double** factordata;        //stores factors for computations
    double** truedata;        //the original unaltered data
        double* solution;        //for storing the solution
    int* pivotarr;            //array for storing pivot elements
                    //(maps original rows to pivoted/reordered rows)
    lcos::mutex mtex;        //mutex

    public:
    //here we define the actions that will be used
    //the construct function
    typedef actions::result_action5<HPLMatreX2, int, hpl_construct, naming::id_type,
        unsigned int, unsigned int, unsigned int, unsigned int,
        &HPLMatreX2::construct> construct_action;
    //the destruct function
    typedef actions::action0<HPLMatreX2, hpl_destruct,
        &HPLMatreX2::destruct> destruct_action;
     //the assign function
    typedef actions::result_action4<HPLMatreX2, int, hpl_assign, unsigned int,
        unsigned int, bool, unsigned int, &HPLMatreX2::assign> assign_action;
    //the get function
    typedef actions::result_action2<HPLMatreX2, double, hpl_get, unsigned int,
            unsigned int, &HPLMatreX2::get> get_action;
    //the set function
    typedef actions::action3<HPLMatreX2, hpl_set, unsigned int,
            unsigned int, double, &HPLMatreX2::set> set_action;
    //the solve function
    typedef actions::result_action0<HPLMatreX2, double, hpl_solve,
        &HPLMatreX2::LUsolve> solve_action;
     //the swap function
    typedef actions::result_action2<HPLMatreX2, int, hpl_swap, unsigned int,
        unsigned int, &HPLMatreX2::swap> swap_action;
    //the main gaussian function
    typedef actions::result_action4<HPLMatreX2, int, hpl_gmain, unsigned int, unsigned
        int, int, unsigned int, &HPLMatreX2::LUgaussmain> gmain_action;
    //the mini gauss trail function
    typedef actions::result_action4<HPLMatreX2, int, hpl_mintrail, unsigned int, unsigned
        int, unsigned int, unsigned int, &HPLMatreX2::minitrail> mintrail_action;
    //backsubstitution function
    typedef actions::result_action0<HPLMatreX2, int, hpl_bsubst,
        &HPLMatreX2::LUbacksubst> bsubst_action;
    //checksolve function
    typedef actions::result_action3<HPLMatreX2, double, hpl_check, unsigned int,
        unsigned int, bool, &HPLMatreX2::checksolve> check_action;

    //here begins the definitions of most of the future types that will be used
    //the first of which is for assign action
    typedef lcos::eager_future<server::HPLMatreX2::assign_action> assign_future;

    //Here is the swap future, which works the same way as the assign future
    typedef lcos::eager_future<server::HPLMatreX2::swap_action> swap_future;

    //This future corresponds to the Gaussian elimination functions
    typedef lcos::eager_future<server::HPLMatreX2::gmain_action> gmain_future;

    //This future corresponds to the minigausstrail function
    typedef lcos::eager_future<server::HPLMatreX2::mintrail_action> mintrail_future;
    //the backsubst future is used to make sure all computations are complete before
    //returning from LUsolve, to avoid killing processes and erasing the leftdata while
    //it is still being worked on

    typedef lcos::eager_future<server::HPLMatreX2::bsubst_action> bsubst_future;

    //the final future type for the class is used for checking the accuracy of
    //the results of the LU decomposition
    typedef lcos::eager_future<server::HPLMatreX2::check_action> check_future;

//***//
    //central_futures is an array of pointers to future which needs to be kept global
    //to this class and is used to represent LUgausscorner inducing gmain_futures
    gmain_future** central_futures;

    //top_futures and left_futures are 2d arrays of pointers to futures similar to
    //central_futures
    gmain_future*** top_futures;
    gmain_future*** left_futures;
    };
//////////////////////////////////////////////////////////////////////////////////////

    //the constructor initializes the matrix
    int HPLMatreX2::construct(naming::id_type gid, unsigned int h, unsigned int w,
        unsigned int ab, unsigned int bs){
// / / /initialize class variables/ / / / / / / / / / / /
    if(ab > std::ceil(((float)h)*.5)){
        allocblock = (int)std::ceil(((float)h)*.5);
    }
    else{ allocblock = ab;}
    if(bs > h){
        blocksize = h;
    }
    else{ blocksize = bs;}
    rows = h;
    brows = (int)std::floor((float)h/blocksize);
    columns = w;
    _gid = gid;

// / / / / / / / / / / / / / / / / / / / / / / / / / / /

    int i,j;          //just counting variables
    unsigned int offset = 1; //the initial offset used for the memory handling algorithm
    boost::rand48 gen;     //random generator used for seeding other generators
    gen.seed(time(NULL));

    /*allocate memory for the various arrays that will be used*/
    central_futures = (gmain_future**) std::malloc(brows*sizeof(gmain_future*));
    left_futures = (gmain_future***) std::malloc(brows*sizeof(gmain_future**));
    top_futures = (gmain_future***) std::malloc(brows*sizeof(gmain_future**));
    datablock = (LUblock***) std::malloc(brows*sizeof(LUblock**));
    factordata = (double**) std::malloc(h*sizeof(double*));
    truedata = (double**) std::malloc(h*sizeof(double*));
    pivotarr = (int*) std::malloc(h*sizeof(int));

    allocate();    //allocate memory for the elements of the array

    //By making offset a power of two, the assign functions
    //are much simpler than they would be otherwise.
    //The result of the below computation is that offset will be
    //the greatest power of two less than or equal to the number
    //of rows in the matrix
    h = (unsigned int)std::ceil(((float)h)*.5);
    while(offset < h){
        offset *= 2;
    }

    //here we initialize the the matrix
    lcos::eager_future<server::HPLMatreX2::assign_action>
        assign_future(gid,(unsigned int)0,offset,false,gen());

    //initialize the pivot array
    for(i=0;i<rows;i++){pivotarr[i]=i;}

    //initialize the arrays of futures
    for(i=0;i<brows;i++){
        central_futures[i] = NULL;
        for(j=0;j<i;j++){
            left_futures[i][j] = NULL;
        }
        for(j=0;j<brows-i-1;j++){
            top_futures[i][j] = NULL;
    }    }

    //make sure that everything has been allocated their memory
    //and initialized
    return assign_future.get();
    }

    //allocate() allocates memory space for the matrix
    void HPLMatreX2::allocate(){
    for(unsigned int i = 0;i < rows;i++){
        truedata[i] = (double*) std::malloc(columns*sizeof(double));
        factordata[i] = (double*) std::malloc(i*sizeof(double));
    }
    for(unsigned int i = 0;i < brows;i++){
        datablock[i] = (LUblock**)std::malloc(brows*sizeof(LUblock*));
        top_futures[i] = (gmain_future**)std::malloc((brows-i-1)*sizeof(gmain_future*));
        left_futures[i] = (gmain_future**)std::malloc(i*sizeof(gmain_future*));
    }
    }

    /*assign gives values to the empty elements of the array.
    The idea behind this algorithm is that the initial thread produces threads with offsets
    of each power of 2 less than (or equal to) the number of rows in the matrix.  Each of the
    generated threads repeats the process using its assigned row as the base and the new set
    of offsets is each power of two that will not overlap with other threads' assigned rows.
    After each thread has produced all of its child threads, that thread initializes the data
    of its assigned row and waits for the child threads to complete before returning.*/
    int HPLMatreX2::assign(unsigned int row, unsigned int offset, bool complete, unsigned int seed){
        //futures is used to allow this thread to continue spinning off new threads
        //while other threads work, and is checked at the end to make certain all
        //threads are completed before returning.
        std::vector<assign_future> futures;
    boost::rand48 gen;
    gen.seed(seed);

    //create multiple futures which in turn create more futures
        while(!complete){
        //there is only one more child thread left to produce
            if(offset <= allocblock){
                if(row + offset < rows){
                    futures.push_back(assign_future(_gid,row+offset,offset,true,gen()));
                }
                complete = true;
            }
        //there are at least two more child threads to produce
            else{
                if(row + offset < rows){
                    futures.push_back(assign_future(_gid,row+offset,
            (unsigned int)(offset*.5),false,gen()));
                }
                offset = (unsigned int)(offset*.5);
            }
        }
    //initialize the assigned row
    unsigned int temp = std::min((int)offset, (int)(rows - row));
    for(unsigned int i=0;i<temp;i++){
        for(unsigned int j=0;j<columns;j++){
        truedata[row+i][j] = (double) (gen() % 1000);
        }
    }

    //once all spun off futures are complete we are done
        BOOST_FOREACH(assign_future af, futures){
                af.get();
        }
    return 1;
    }

    //the destructor frees the memory
    void HPLMatreX2::destruct(){
    unsigned int i;
    for(i=0;i<rows;i++){
        free(truedata[i]);
    }
    for(i=0;i<brows;i++){
        for(int j=0;j<brows;j++){
            delete datablock[i][j];
        }
        free(datablock[i]);
    }
    free(datablock);
    free(factordata);
    free(truedata);
    free(pivotarr);
    free(solution);
    }

//DEBUGGING FUNCTIONS/////////////////////////////////////////////////
    //get() gives back an element in the original matrix
    double HPLMatreX2::get(unsigned int row, unsigned int col){
    return truedata[row][col];
    }

    //set() assigns a value to an element in all matrices
    void HPLMatreX2::set(unsigned int row, unsigned int col, double val){
    truedata[row][col] = val;
    }

    //print out the matrix
    void HPLMatreX2::print(){
    for(int i = 0; i < brows; i++){
        for(int j = 0; j < datablock[i][0]->getrows(); j++){
        for(int k = 0; k < brows; k++){
            for(int l = 0; l < datablock[i][k]->getcolumns(); l++){
            std::cout<<datablock[i][k]->get(j,l)<<" ";
            }
            std::cout<<" >> ";
        }
        std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }   
    std::cout<<std::endl;
    }

    void HPLMatreX2::print2(){
    for(int x = 0; x < brows; x++){
        for(int y = 0; y < brows; y++){
        std::cout<<datablock[x][y]->getcolumns()<<" ";

        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    }
//END DEBUGGING FUNCTIONS/////////////////////////////////////////////

    //LUsolve is simply a wrapper function for LUfactor and LUbacksubst
    double HPLMatreX2::LUsolve(){
    //first perform partial pivoting
    pivot();

//print2();
//print();
    //to initiate the Gaussian elimination, create a future to obtain the final
    //set of computations we will need
    gmain_future main_future(_gid,brows-1,brows-1,brows-1,1);
    main_future.get();

    //allocate memory space to store the solution
    solution = (double*) std::malloc(rows*sizeof(double));

    //create a future to perform back substitution
    bsubst_future bsub(_gid);

    unsigned int h = (unsigned int)std::ceil(((float)rows)*.5);
    unsigned int offset = 1;
    while(offset < h){offset *= 2;}

    //confirm back substitution is completed before continuing
    bsub.get();

    //create a future to check the accuracy of the generated solution
    check_future chk(_gid,0,offset,false);
    return chk.get();
//    return 1;
    }

    //pivot() finds the pivot element of each column and stores it. All
    //pivot elements are found before any swapping takes place so that
    //the swapping is simplified
    void HPLMatreX2::pivot(){
    unsigned int max, max_row;
    unsigned int temp_piv;
    double temp;
    //unsigned int h = (unsigned int)std::ceil(((float)rows)*.5);

    //This first section of the pivot() function finds all of the pivot
    //values to compute the final pivot array
    for(unsigned int i=0;i<rows-1;i++){
        max_row = i;
        max = (unsigned int)fabs(truedata[pivotarr[i]][i]);
        temp_piv = pivotarr[i];
        for(unsigned int j=i+1;j<rows;j++){
        temp = (unsigned int)fabs(truedata[pivotarr[j]][i]);
        if(temp > max){
            max = (unsigned int)temp;
            max_row = j;
        }    }
        pivotarr[i] = pivotarr[max_row];
        pivotarr[max_row] = temp_piv;
    }

    //call the swap function for each datablock in the matrix
    for(int i=0;i<brows;i++){
        for(int j=0;j<brows;j++){
            swap(i,j);
    }    }
    }

    //swap() creates the datablocks and reorders the original
    //truedata matrix when assigning the initial values to the datablocks
    //according to the pivotarr data.  truedata itself remains unchanged
    int HPLMatreX2::swap(unsigned int brow, unsigned int bcol){
    unsigned int temp = rows/blocksize;
    unsigned int numrows = blocksize, numcols = blocksize;
    if(brow == brows-1){numrows = rows - (temp-1)*blocksize;}
    if(bcol == brows-1){numcols = columns - (temp-1)*blocksize;}

    datablock[brow][bcol] = new LUblock(numrows,numcols);
    for(unsigned int i=0;i<numrows;i++){
        for(unsigned int j=0;j<numcols;j++){
        datablock[brow][bcol]->set(i, j,
            truedata[pivotarr[brow*blocksize+i]][bcol*blocksize+j]);
    }   }
    }

    //LUgaussmain is a wrapper function which is used so that only one type of action
    //is needed instead of four types of actions
    int HPLMatreX2::LUgaussmain(unsigned int brow,unsigned int bcol,int iter,
        unsigned int type){
    //if the following conditional is true, then there is nothing to do
    if(type == 0 && (brow == 0 || bcol == 0) || iter < 0){return 1;}

    //we will need to create a future to perform LUgausstrail regardless
    //of what this current function instance will compute
    gmain_future trail_future(_gid,brow,bcol,iter-1,0);

    //below decides what other futures need to be created in order to complete all
    //necessary computations before we can return
    if(type == 1){
        trail_future.get();
        LUgausscorner(iter);
    }
    else if(type == 2){
        {
            lcos::mutex::scoped_lock l(mtex);
            if(central_futures[iter] == NULL){
            central_futures[iter] = new gmain_future(_gid,iter,iter,iter,1);
            }
        }
        trail_future.get();
        central_futures[iter]->get();
        LUgausstop(iter,bcol);
    }
    else if(type == 3){
        {
            lcos::mutex::scoped_lock l(mtex);
            if(central_futures[iter] == NULL){
            central_futures[iter] = new gmain_future(_gid,iter,iter,iter,1);
            }
        }
        trail_future.get();
        central_futures[iter]->get();
        LUgaussleft(brow,iter);
    }
    else{
        {
            lcos::mutex::scoped_lock l(mtex);
            if(top_futures[iter][bcol-iter-1] == NULL){
            top_futures[iter][bcol-iter-1] = new gmain_future(_gid,iter,bcol,iter,2);
            }
            if(left_futures[brow][iter] == NULL){
            left_futures[brow][iter] = new gmain_future(_gid,brow,iter,iter,3);
            }
        }
        trail_future.get();
        top_futures[iter][bcol-iter-1]->get();
        left_futures[brow][iter]->get();
        LUgausstrail(brow,bcol,iter);
    }
    return 1;
    }

    //LUgausscorner peforms gaussian elimination on the topleft corner block
    //of data that has not yet completed all of it's gaussian elimination
    //computations. Once complete, this block will need no further computations
    void HPLMatreX2::LUgausscorner(unsigned int iter){
    unsigned int i, j, k;
    unsigned int offset = iter*blocksize;        //factordata index offset
    double f_factor;

    for(i=0;i<datablock[iter][iter]->getrows();i++){
        if(datablock[iter][iter]->get(i,i) == 0){
            std::cerr<<"Warning: divided by zero\n";
        }
        f_factor = 1/datablock[iter][iter]->get(i,i);
        for(j=i+1;j<datablock[iter][iter]->getrows();j++){
            factordata[j+offset][i+offset] = f_factor*datablock[iter][iter]->get(j,i);
            for(k=i+1;k<datablock[iter][iter]->getcolumns();k++){
            datablock[iter][iter]->set(j,k,datablock[iter][iter]->get(j,k) -
                factordata[j+offset][i+offset]*datablock[iter][iter]->get(i,k));
    }    }   }
    }

    //LUgausstop performs gaussian elimination on the topmost row of blocks
    //that have not yet finished all gaussian elimination computation.
    //Once complete, these blocks will no longer need further computations
    void HPLMatreX2::LUgausstop(unsigned int iter, unsigned int bcol){
    unsigned int i,j,k;
    unsigned int offset = iter*blocksize;        //factordata index offset

    for(i=0;i<datablock[iter][bcol]->getrows();i++){
        for(j=i+1;j<datablock[iter][bcol]->getrows();j++){
            for(k=0;k<datablock[iter][bcol]->getcolumns();k++){
            datablock[iter][bcol]->set(j,k,datablock[iter][bcol]->get(j,k) -
                factordata[j+offset][i+offset]*datablock[iter][bcol]->get(i,k));
    }    }   }
    }

    //LUgaussleft performs gaussian elimination on the leftmost column of blocks
    //that have not yet finished all gaussian elimination computation.
    //Upon completion, no further computations need be done on these blocks.
    void HPLMatreX2::LUgaussleft(unsigned int brow, unsigned int iter){
    unsigned int i,j,k;
    unsigned int offset = brow*blocksize;
    unsigned int offset_col = iter*blocksize;    //factordata offset
    double f_factor;

    for(i=0;i<datablock[brow][iter]->getcolumns();i++){
        f_factor = 1/datablock[iter][iter]->get(i,i);
        for(j=0;j<datablock[brow][iter]->getrows();j++){
            factordata[j+offset][i+offset_col] = f_factor*datablock[brow][iter]->get(j,i);
            for(k=i+1;k<datablock[brow][iter]->getcolumns();k++){
            datablock[brow][iter]->set(j,k,datablock[brow][iter]->get(j,k) -
                factordata[j+offset][i+offset_col]*datablock[iter][iter]->get(i,k));
    }    }   }
    }

    //LUgausstrail performs gaussian elimination on the trailing submatrix of
    //the blocks operated on during the current iteration of the Gaussian elimination
    //computations. These blocks will still require further computations to be
    //performed in future iterations.
    void HPLMatreX2::LUgausstrail(unsigned int brow, unsigned int bcol, unsigned int iter){
    unsigned int i;
    std::vector<mintrail_future> g_futures;

    //outermost loop: iterates over the f_factors of the most recent corner block
    //    (f_factors are used indirectly through factordata)
    //middle loop: iterates over the rows of the current block
    //inner loop: iterates across the columns of the current block

    for(i=0;i<datablock[iter][iter]->getrows();i++){
        g_futures.push_back(mintrail_future(_gid,brow,bcol,iter,i));   
        }    
    BOOST_FOREACH(mintrail_future mf, g_futures){
                   mf.get();
    }
    }

    //This function was made to allow the gaussian elimination step to proceed in parallel for LUgausstrail
    int HPLMatreX2::minitrail(unsigned int brow, unsigned int bcol, unsigned int iter, unsigned int i){
    unsigned int j,k;
    unsigned int offset = brow*blocksize;
    unsigned int offset_col;
    
    offset_col = iter*blocksize;
        
    for(j=0;j<datablock[brow][bcol]->getrows();j++){
        for(k=0;k<datablock[brow][bcol]->getcolumns();k++){
        //lcos::mutex::scoped_lock l(mtex);
        datablock[brow][bcol]->set(j,k,datablock[brow][bcol]->get(j,k) -
            factordata[j+offset][i+offset_col]*datablock[iter][bcol]->get(i,k));
        }
    }
    return 1;
    }

    //this is an implementation of back substitution modified for use on
    //multiple datablocks instead of a single large data structure
    int HPLMatreX2::LUbacksubst(){
    //i,j,k,l standard loop variables, row and col keep
    //track of where the loops would be in terms of a single
    //large data structure; using it allows addition
    //where multiplication would need to be used without it
        int i,j,k,l,row,col;
    //unsigned int temp = datablock[0][bcolumns-1]->getcolumns()-1;

    for(i=0;i<brows;i++){
        for(j=0;j<datablock[i][0]->getrows();j++){
        solution[i*blocksize+j] = datablock[i][brows-1]->get(j,
            datablock[i][brows-1]->getcolumns()-1);
    }   }

        for(i=brows-1;i>=0;i--){
        row = i*blocksize;
        for(j=brows-1;j>=i;j--){
        col = j*blocksize;
        //the block of code following the if statement handles all data blocks
        //that to not include elements on the diagonal
        if(i!=j){
            for(k=datablock[i][j]->getcolumns()-((j>=brows-1)?(2):(1));k>=0;k--){
            for(l=datablock[i][j]->getrows()-1;l>=0;l--){
                        solution[row+l]-=datablock[i][j]->get(l,k)*solution[col+k];
            datablock[i][j]->set(l,k,3333);
                }   }   }
        //this block of code following the else statement handles all data blocks
        //that do include elements on the diagonal
        else{
            for(k=datablock[i][i]->getcolumns()-((i==brows-1)?(2):(1));k>=0;k--){
            solution[row+k]/=datablock[i][i]->get(k,k);
            datablock[i][i]->set(k,k,9999);
//            std::cout<<solution[row+k]<<std::endl;
            for(l=k-1;l>=0;l--){
                        solution[row+l]-=datablock[i][i]->get(l,k)*solution[col+k];
            datablock[i][i]->set(l,k,3333);
        }   }    }   }    }

    return 1;
    }

    //finally, this function checks the accuracy of the LU computation a few rows at
    //a time
    double HPLMatreX2::checksolve(unsigned int row, unsigned int offset, bool complete){
    double toterror = 0;    //total error from all checks

        //futures is used to allow this thread to continue spinning off new threads
        //while other threads work, and is checked at the end to make certain all
        //threads are completed before returning.
        std::vector<check_future> futures;

        //start spinning off work to do
        while(!complete){
            if(offset <= allocblock){
                if(row + offset < rows){
                    futures.push_back(check_future(_gid,row+offset,offset,true));
                }
                complete = true;
            }
            else{
                if(row + offset < rows){
            futures.push_back(check_future(_gid,row+offset,
                (unsigned int)(offset*.5),false));
                }
                offset = (unsigned int)(offset*.5);
            }
        }

    //accumulate the total error for a subset of the solutions
        unsigned int temp = std::min((int)offset, (int)(rows - row));
        for(unsigned int i=0;i<temp;i++){
        double sum = 0;
            for(unsigned int j=0;j<rows;j++){
                sum += truedata[pivotarr[row+i]][j] * solution[j];
            }
        toterror += std::fabs(sum-truedata[pivotarr[row+i]][rows]);
        }

        //collect the results and add them up
        BOOST_FOREACH(check_future cf, futures){
                toterror += cf.get();
        }
        return toterror;
    }
}}}

#endif
