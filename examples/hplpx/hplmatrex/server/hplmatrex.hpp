///////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0.(See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////
#ifndef _HPLMATREX_SERVER_HPP
#define _HPLMATREX_SERVER_HPP

/*This is the HPLMatreX class implementation header file.
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

#include <boost/date_time/posix_time/posix_time.hpp>
using namespace boost::posix_time;

namespace hpx { namespace components { namespace server
{
    class HPX_COMPONENT_EXPORT HPLMatreX : 
          public simple_component_base<HPLMatreX>
    {
    public:
    //enumerate all of the actions that will(or can) be employed
        enum actions{
            hpl_construct=0,
            hpl_destruct=1,
            hpl_assign=2,
            hpl_trans=3,
            hpl_set=4,
            hpl_solve=5,
            hpl_swap=6,
            hpl_gmain=7,
            hpl_bsubst=8,
            hpl_check=9
        };

    //constructors and destructor
    HPLMatreX(){}
    int construct(naming::id_type gid, int h, int l, int ab, int bs);
    ~HPLMatreX(){destruct();}
    void destruct();

    //operators for assignment and leftdata access
    double get(int row, int col);
    void set(int row, int col, double val);

    //functions for manipulating the matrix
    double LUsolve();

    private:
    void allocate();
    int assign(int row, int offset, bool complete, int seed);
    int transpose(int offset_row, int offset_col);
    void pivot();
    int swap(int brow, int bcol);
    void LUgausscorner(int iter);
    void LUgausstop(int iter, int bcol);
    void LUgaussleft(int brow, int iter);
    void LUgausstrail(int brow, int bcol, int iter);
    int LUgaussmain(int brow, int bcol, int iter, int type);
    int LUbacksubst();
    double checksolve(int row, int offset, bool complete);
    void print();
    void print2();

    int rows;             //number of rows in the matrix
    int brows;            //number of rows of blocks in the matrix
    int columns;          //number of columns in the matrix
    int bcolumns;         //number of columns of blocks in the matrix
    int allocblock;       //reflects amount of allocation done per thread
    int blocksize;        //reflects amount of computation per thread
    int lit_block;        //size of inner loop blocks
    naming::id_type _gid; //the instances gid
    LUblock*** datablock; //stores the data being operated on
    double** factordata;  //stores factors for computations
    double** truedata;    //the original unaltered data
    double** transdata;   //transpose of the original data(speeds up pivoting)
    double* solution;     //for storing the solution
    int* pivotarr;        //array for storing pivot elements
                          //(maps original rows to pivoted/reordered rows)
    lcos::mutex mtex;     //mutex

    ptime starttime;      //used for measurements
    ptime rectime;        //used for measurements
    public:
    //here we define the actions that will be used
    //the construct function
    typedef actions::result_action5<HPLMatreX, int, hpl_construct, 
        naming::id_type, int, int, int, int, &HPLMatreX::construct> 
        construct_action;
    //the destruct function
    typedef actions::action0<HPLMatreX, hpl_destruct,
        &HPLMatreX::destruct> destruct_action;
     //the assign function
    typedef actions::result_action4<HPLMatreX, int, hpl_assign, int,
        int, bool, int, &HPLMatreX::assign> assign_action;
    //the transpose function
    typedef actions::result_action2<HPLMatreX, int, hpl_trans, int,
            int, &HPLMatreX::transpose> transpose_action;
    //the set function
    typedef actions::action3<HPLMatreX, hpl_set, int,
            int, double, &HPLMatreX::set> set_action;
    //the solve function
    typedef actions::result_action0<HPLMatreX, double, hpl_solve,
        &HPLMatreX::LUsolve> solve_action;
     //the swap function
    typedef actions::result_action2<HPLMatreX, int, hpl_swap, int,
        int, &HPLMatreX::swap> swap_action;
    //the main gaussian function
    typedef actions::result_action4<HPLMatreX, int, hpl_gmain, int,
        int, int, int, &HPLMatreX::LUgaussmain> gmain_action;
    //backsubstitution function
    typedef actions::result_action0<HPLMatreX, int, hpl_bsubst,
        &HPLMatreX::LUbacksubst> bsubst_action;
    //checksolve function
    typedef actions::result_action3<HPLMatreX, double, hpl_check, int,
        int, bool, &HPLMatreX::checksolve> check_action;

    //here begins the definitions of most of the future types that will be used
    //the first of which is for assign action
    typedef lcos::eager_future<server::HPLMatreX::assign_action> assign_future;

    //This is the future for the transpose function
    typedef lcos::eager_future<server::HPLMatreX::transpose_action> transpose_future;

    //Here is the swap future, which works the same way as the assign future
    typedef lcos::eager_future<server::HPLMatreX::swap_action> swap_future;

    //This future corresponds to the Gaussian elimination functions
    typedef lcos::eager_future<server::HPLMatreX::gmain_action> gmain_future;

    //the backsubst future is used to make sure all computations are complete
    //before returning from LUsolve, to avoid killing processes and erasing the
    //leftdata while it is still being worked on
    typedef lcos::eager_future<server::HPLMatreX::bsubst_action> bsubst_future;

    //the final future type for the class is used for checking the accuracy of
    //the results of the LU decomposition
    typedef lcos::eager_future<server::HPLMatreX::check_action> check_future;

//***//
    //central_futures is an array of pointers to future which needs to be 
    //kept global to this class and is used to represent LUgausscorner 
    //inducing gmain_futures
    gmain_future** central_futures;

    //top_futures and left_futures are 2d arrays of pointers to futures
    //similar to central_futures
    gmain_future*** top_futures;
    gmain_future*** left_futures;
    };
///////////////////////////////////////////////////////////////////////////////

    //the constructor initializes the matrix
    int HPLMatreX::construct(naming::id_type gid, int h, int l,
        int ab, int bs){
    starttime = ptime(microsec_clock::local_time());
// / / /initialize class variables/ / / / / / / / / / / /
    if(ab > std::ceil(((float)h)*.5)){
        allocblock = (int)std::ceil(((float)h)*.5);
    }
    else{ allocblock = ab;}
    if(bs > h){
        blocksize = h;
    }
    else{ blocksize = bs;}
    if(l > blocksize){
        lit_block = blocksize;
    }
    else{ lit_block = l;}
    rows = h;
    brows = (int)std::floor((float)h/blocksize);
    columns = h+1;
    _gid = gid;

// / / / / / / / / / / / / / / / / / / / / / / / / / / /

    int i,j;          //just counting variables
    int offset = 1; //the initial offset used for the memory handling algorithm
    std::vector<transpose_future> futures; //used for transposing the matrix
    boost::rand48 gen;     //random generator used for seeding other generators
    gen.seed(time(NULL));

    /*allocate memory for the various arrays that will be used*/
    allocate();    //allocate memory for the elements of the array

    //By making offset a power of two, the assign functions
    //are much simpler than they would be otherwise.
    //The result of the below computation is that offset will be
    //the greatest power of two less than or equal to the number
    //of rows in the matrix
    h = (int)std::ceil(((float)h)*.5);
    while(offset < h){
        offset *= 2;
    }

    //here we initialize the the matrix
    lcos::eager_future<server::HPLMatreX::assign_action>
        assign_future(gid,(int)0,offset,false,gen()); 

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
    assign_future.get();

    return 1;
    }

    //allocate() allocates memory space for the matrix
    void HPLMatreX::allocate(){
    central_futures = (gmain_future**) std::malloc(brows*sizeof(gmain_future*));
    left_futures = (gmain_future***) std::malloc(brows*sizeof(gmain_future**));
    top_futures = (gmain_future***) std::malloc(brows*sizeof(gmain_future**));
    datablock = (LUblock***) std::malloc(brows*sizeof(LUblock**));
    factordata = (double**) std::malloc(rows*sizeof(double*));
    transdata = (double**) std::malloc(columns*sizeof(double*));
    truedata = (double**) std::malloc(rows*sizeof(double*));
    pivotarr = (int*) std::malloc(rows*sizeof(int));
    for(int i = 0;i < rows;i++){
        truedata[i] = (double*) std::malloc(columns*sizeof(double));
        transdata[i] = (double*) std::malloc(rows*sizeof(double));
    }
    transdata[rows] = (double*) std::malloc(rows*sizeof(double));
    for(int i = 0;i < brows;i++){
        datablock[i] = (LUblock**)std::malloc(brows*sizeof(LUblock*));
        top_futures[i] = (gmain_future**)std::malloc((brows-i-1)
                            *sizeof(gmain_future*));
        left_futures[i] = (gmain_future**)std::malloc(i*sizeof(gmain_future*));
    }
    }

    /*assign gives values to the empty elements of the array.
    The idea behind this algorithm is that the initial thread produces
    threads with offsets of each power of 2 less than (or equal to) the
    number of rows in the matrix.  Each of the generated threads repeats
    the process using its assigned row as the base and the new set of 
    offsets is each power of two that will not overlap with other threads'
    assigned rows. After each thread has produced all of its child threads,
    that thread initializes the data of its assigned row and waits for the
    child threads to complete before returning.*/
    int HPLMatreX::assign(int row, int offset, bool complete, int seed){
        //futures is used to allow this thread to continue spinning off new
        //threads while other threads work, and is checked at the end to make
        //certain all threads are completed before returning.
        std::vector<assign_future> futures;
        boost::rand48 gen;
        gen.seed(seed);

        //create multiple futures which in turn create more futures
        while(!complete){
        //there is only one more child thread left to produce
            if(offset <= allocblock){
                if(row + offset < rows){
                    futures.push_back(assign_future(_gid,
                       row+offset,offset,true,gen()));
                }
                complete = true;
            }
            //there are at least two more child threads to produce
            else{
                if(row + offset < rows){
                    futures.push_back(assign_future(_gid,row+offset,
            (int)(offset*.5),false,gen()));
                }
                offset = (int)(offset*.5);
            }
        }
    //initialize the assigned row
    int temp = std::min((int)offset, (int)(rows - row));
    for(int i=0;i<temp;i++){
        for(int j=0;j<columns;j++){
            transdata[j][row+i] = truedata[row+i][j] = (double) (gen() % 1000);
        }
    }

    //once all spun off futures are complete we are done
    BOOST_FOREACH(assign_future af, futures){
        af.get();
    }
    return 1;
    }

    //transpose transposes a block of the truedata array into the
    //transdata array. we do this because even with the additional
    //initialization work the speedup of the partial_pivoting more
    //than compensates
    int HPLMatreX::transpose(int offset_row,int offset_col){
    int i,j,ii,jj;
    int bsize = 64;
    int endrow = std::min(offset_row+blocksize,rows);
    int endcolumn = std::min(offset_col+blocksize,columns);

    if(offset_row/blocksize == brows-1){endrow = rows;}
    if(offset_col/blocksize == brows-1){endcolumn = columns;}
    //initialize the transdata matrix
    for(ii = offset_row; ii < endrow; ii+=bsize){
    for(jj = offset_col; jj < endcolumn; jj+=bsize){
    for(i = ii; i < std::min(endrow,ii+bsize); i++){
    for(j = jj; j < std::min(endcolumn,jj+bsize); j++){
        transdata[j][i] = truedata[i][j];
    }}}}
    return 1;
    }

    //the destructor frees the memory
    void HPLMatreX::destruct(){
    int i;
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
    double HPLMatreX::get(int row, int col){
    return truedata[row][col];
    }

    //set() assigns a value to an element in all matrices
    void HPLMatreX::set(int row, int col, double val){
    truedata[row][col] = val;
    }

    //print out the matrix
    void HPLMatreX::print(){
    for(int i = 0; i < brows; i++){
        for(int j = 0; j < datablock[i][0]->getrows(); j++){
        for(int k = 0; k < brows; k++){
            for(int l = 0; l < datablock[i][k]->getcolumns(); l++){
            std::cout<<datablock[i][k]->get(j,l)<<" ";
        }   }
        std::cout<<std::endl;
    }   }
    std::cout<<std::endl;
    }
    void HPLMatreX::print2(){
    for(int i = 0;i < rows; i++){
        for(int j = 0;j < columns; j++){
        std::cout<<truedata[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    for(int i = 0;i < columns; i++){
        for(int j = 0;j < rows; j++){
        std::cout<<transdata[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    }
//END DEBUGGING FUNCTIONS/////////////////////////////////////////////

    //LUsolve is simply a wrapper function for LUfactor and LUbacksubst
    double HPLMatreX::LUsolve(){
    //first perform partial pivoting
    ptime temp = ptime(microsec_clock::local_time());
    std::cout<<"alloc time "<<temp-starttime<<std::endl;
    pivot();

    ptime temp2 = ptime(microsec_clock::local_time());
    std::cout<<"pivoting over "<<temp2-temp<<std::endl;

    //to initiate the Gaussian elimination, create a future to obtain the final
    //set of computations we will need
    gmain_future main_future(_gid,brows-1,brows-1,brows-1,1);
    main_future.get();

    ptime temp3 = ptime(microsec_clock::local_time());
    std::cout<<"finished gaussian "<<temp3 - temp2<<std::endl;
    //allocate memory space to store the solution
    solution = (double*) std::malloc(rows*sizeof(double));

    //perform back substitution
    LUbacksubst();

    int h = (int)std::ceil(((float)rows)*.5);
    int offset = 1;
    while(offset < h){offset *= 2;}

    ptime temp4 = ptime(microsec_clock::local_time());
    std::cout<<"bsub done "<<temp4-temp3<<std::endl;
    //create a future to check the accuracy of the generated solution
    check_future chk(_gid,0,offset,false);
    return chk.get();
//    return 1;
    }


    //pivot() finds the pivot element of each column and stores it. All
    //pivot elements are found before any swapping takes place so that
    //the swapping is simplified
    void HPLMatreX::pivot(){
    double max;
    int max_row;
    int temp_piv;
    double temp;
    int outer;
    int i=0,j;
    std::vector<swap_future> futures;

    for(outer=0;outer<=brows;outer++){
        //This first section of the pivot() function finds all of the pivot
        //values to compute the final pivot array
        for(i=i;i<std::min((outer+1)*blocksize,rows-1);i++){
            max_row = i;
            max = fabs(transdata[i][pivotarr[i]]);
            temp_piv = pivotarr[i];
            for(j=i+1;j<rows;j++){
            temp = fabs(transdata[i][pivotarr[j]]);
            if(temp > max){
                max = temp;
                max_row = j;
            }   }
            pivotarr[i] = pivotarr[max_row];
            pivotarr[max_row] = temp_piv;
        }
        //here we begin swapping portions of the matrix we have finished
        //finding the pivot values for
        if(outer<brows){
            futures.push_back(swap_future(_gid,outer,0));
        }
    }
    //transdata is no longer needed so free the memory and allocate
    //space for factordata
    for(i=0;i<rows;i++){
        free(transdata[i]);
        factordata[i] = (double*) std::malloc(i*sizeof(double));
    }
    free(transdata[rows]);
    free(transdata);

    //ensure that all pivoting is complete
    BOOST_FOREACH(swap_future sf, futures){
        sf.get();
    }

    }


    //swap() creates the datablocks and reorders the original
    //truedata matrix when assigning the initial values to the datablocks
    //according to the pivotarr data.  truedata itself remains unchanged
    int HPLMatreX::swap(int brow, int bcol){
    int temp = rows/blocksize;
    int numrows = blocksize, numcols = blocksize;

    for(int k=0;k<brows;k++){
    if(brow == brows-1){numrows = rows - (temp-1)*blocksize;}
    if(k == brows-1){numcols = columns - (temp-1)*blocksize;}
    datablock[brow][k] = new LUblock(numrows,numcols);
    for(int i=0;i<numrows;i++){
        for(int j=0;j<numcols;j++){
        datablock[brow][k]->data[i][j] =
            truedata[pivotarr[brow*blocksize+i]][k*blocksize+j];
    }   }
    }
    return 1;
    }

    //LUgaussmain is a wrapper function which is used so that only one type of
    //action is needed instead of four types of actions
    int HPLMatreX::LUgaussmain(int brow,int bcol,int iter,
        int type){
    //if the following conditional is true, then there is nothing to do
    if(type == 0 && (brow == 0 || bcol == 0) || iter < 0){return 1;}

    //we will need to create a future to perform LUgausstrail regardless
    //of what this current function instance will compute
    gmain_future trail_future(_gid,brow,bcol,iter-1,0);
    for(int i = 0; i<2000000;i++){i++;}
    //below decides what other futures need to be created in order to complete
    //all necessary computations before we can return
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
            top_futures[iter][bcol-iter-1] = 
                new gmain_future(_gid,iter,bcol,iter,2);
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
    void HPLMatreX::LUgausscorner(int iter){
    int i, j, k;
    int offset = iter*blocksize;        //factordata index offset
    double f_factor;
    double factor;

    for(i=0;i<datablock[iter][iter]->getrows();i++){
        if(datablock[iter][iter]->get(i,i) == 0){
            std::cerr<<"Warning: divided by zero\n";
        }
        f_factor = 1/datablock[iter][iter]->get(i,i);
        for(j=i+1;j<datablock[iter][iter]->rows;j++){
            factor = f_factor*datablock[iter][iter]->data[j][i];
            factordata[j+offset][i+offset] = factor;
            for(k=i+1;k<datablock[iter][iter]->columns;k++){
            datablock[iter][iter]->data[j][k] -= 
                factor*datablock[iter][iter]->data[i][k];
    }   }   }
    }

    //LUgausstop performs gaussian elimination on the topmost row of blocks
    //that have not yet finished all gaussian elimination computation.
    //Once complete, these blocks will no longer need further computations
    void HPLMatreX::LUgausstop(int iter, int bcol){
    int i,j,k;
    int offset = iter*blocksize;        //factordata index offset
    double factor;

    for(i=0;i<datablock[iter][bcol]->getrows();i++){
        for(j=i+1;j<datablock[iter][bcol]->rows;j++){
            factor = factordata[j+offset][i+offset];
            for(k=0;k<datablock[iter][bcol]->columns;k++){
            datablock[iter][bcol]->data[j][k] -= 
                factor*datablock[iter][bcol]->data[i][k];
    }   }   }
    }

    //LUgaussleft performs gaussian elimination on the leftmost column of
    //blocks that have not yet finished all gaussian elimination computation.
    //Upon completion, no further computations need be done on these blocks.
    void HPLMatreX::LUgaussleft(int brow, int iter){
    int i,j,k;
    int offset = brow*blocksize;
    int offset_col = iter*blocksize;    //factordata offset
    double f_factor;
    double factor;

    for(i=0;i<datablock[brow][iter]->getcolumns();i++){
        f_factor = 1/datablock[iter][iter]->get(i,i);
        for(j=0;j<datablock[brow][iter]->rows;j++){
            factor = f_factor*datablock[brow][iter]->data[j][i];
            factordata[j+offset][i+offset_col] = factor;
            for(k=i+1;k<datablock[brow][iter]->columns;k++){
            datablock[brow][iter]->data[j][k] -= 
                factor*datablock[iter][iter]->data[i][k];
    }    }   }
    }

    //LUgausstrail performs gaussian elimination on the trailing submatrix of
    //the blocks operated on during the current iteration of the Gaussian
    //elimination computations. These blocks will still require further 
    //computations to be performed in future iterations.
    void HPLMatreX::LUgausstrail(int brow, int bcol, int iter){
    int i,j,k,ii,jj,kk;
    int offset = brow*blocksize;    //factordata row offset
    int offset_col = iter*blocksize;    //factordata column offset
    double factor;

    //outermost loop: iterates over the f_factors of the most recent corner 
    //block (f_factors are used indirectly through factordata)
    //middle loop: iterates over the rows of the current block
    //inner loop: iterates across the columns of the current block
    for(jj = 0;jj<datablock[brow][bcol]->rows;jj+=lit_block){
    for(ii = 0;ii<datablock[iter][iter]->rows;ii+=lit_block){
//    for(kk = 0;kk<datablock[brow][bcol]->columns;kk+=.5*blocksize){
//    for(j=0;j<datablock[brow][bcol]->rows;j++){
    for(j=jj;j<std::min(jj+lit_block,datablock[brow][bcol]->rows);j++){
//        for(i=0;i<datablock[iter][iter]->rows;i++){
        for(i=ii;i<std::min(ii+lit_block,datablock[iter][iter]->rows);i++){
            factor = factordata[j+offset][i+offset_col];
            for(k=0;k<datablock[brow][bcol]->columns;k++){
//            for(k=kk;k<std::min(kk+(int)(.5*blocksize),datablock[brow][bcol]->columns);k++){
            datablock[brow][bcol]->data[j][k] -= 
                factor*datablock[iter][bcol]->data[i][k];
    }   }   }
    }
    }
    }
//    }

    //this is an implementation of back substitution modified for use on
    //multiple datablocks instead of a single large data structure
    int HPLMatreX::LUbacksubst(){
    //i,j,k,l standard loop variables, row and col keep
    //track of where the loops would be in terms of a single
    //large data structure; using it allows addition
    //where multiplication would need to be used without it
    int i,j,k,l,row,col;

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
            for(k=datablock[i][j]->getcolumns()-
                ((j>=brows-1)?(2):(1));k>=0;k--){
                for(l=datablock[i][j]->getrows()-1;l>=0;l--){
                    solution[row+l] -=
                        datablock[i][j]->data[l][k]*solution[col+k];
        }   }   }
        //this block of code following the else statement handles all data 
        //blocks that do include elements on the diagonal
        else{
            for(k=datablock[i][i]->getcolumns()-
                ((i==brows-1)?(2):(1));k>=0;k--){
                solution[row+k]/=datablock[i][i]->data[k][k];
                for(l=k-1;l>=0;l--){
                    solution[row+l] -=
                        datablock[i][i]->data[l][k]*solution[col+k];
        }   }    }   }    }

    return 1;
    }

    //finally, this function checks the accuracy of the LU computation a few
    //rows at a time
    double HPLMatreX::checksolve(int row, int offset, bool complete){
    double toterror = 0;    //total error from all checks

        //futures is used to allow this thread to continue spinning off new 
        //thread while other threads work, and is checked at the end to make
        // certain all threads are completed before returning.
        std::vector<check_future> futures;

        //start spinning off work to do
        while(!complete){
            if(offset <= allocblock){
                if(row + offset < rows){
                    futures.push_back(check_future(_gid,
                        row+offset,offset,true));
                }
                complete = true;
            }
            else{
                if(row + offset < rows){
            futures.push_back(check_future(_gid,row+offset,
                (int)(offset*.5),false));
                }
                offset = (int)(offset*.5);
            }
        }

    //accumulate the total error for a subset of the solutions
        int temp = std::min((int)offset, (int)(rows - row));
        for(int i=0;i<temp;i++){
        double sum = 0;
            for(int j=0;j<rows;j++){
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
