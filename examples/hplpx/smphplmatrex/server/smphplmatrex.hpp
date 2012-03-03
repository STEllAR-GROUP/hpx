///////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0.(See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////
#ifndef _SMPHPLMATREX_SERVER_HPP
#define _SMPHPLMATREX_SERVER_HPP

/*This is the smphplmatrex class implementation header file.
In order to keep things simple, only operations necessary
to to perform LUP decomposition are declared, which is
basically just constructors, assignment operators,
a destructor, and access operators.
*/

#include <time.h>

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include <boost/foreach.hpp>
#include <boost/random/linear_congruential.hpp>
#include "lublock.hpp"

#include <boost/date_time/posix_time/posix_time.hpp>
using namespace boost::posix_time;

namespace hpx { namespace components { namespace server
{
    class HPX_COMPONENT_EXPORT smphplmatrex :
          public simple_component_base<smphplmatrex>
    {
    public:
    //enumerate all of the actions that will(or can) be employed
        enum actions{
            hpl_construct=0,
            hpl_destruct=1,
            hpl_assign=2,
            hpl_partbsub=3,
            hpl_solve=5,
            hpl_swap=6,
            hpl_gmain=7,
            hpl_search=8,
            hpl_check=9
        };

    //constructors and destructor
    smphplmatrex(){}
    int construct(naming::id_type gid, int h, int ab, int bs);
    ~smphplmatrex(){destruct();}
    void destruct();

    double LUsolve();

    private:
    void allocate();
    int assign(int row, int offset, bool complete, int seed);
    void pivot();
    int search_pivots(const int row);
    int swap(const int brow, const int bcol);
    void LU_gauss_manager();
    void LU_gauss_corner(const int iter);
    void LU_gauss_top(const int iter, const int bcol);
    void LU_gauss_left(const int brow, const int iter);
    void LU_gauss_trail(const int brow, const int bcol, const int iter);
    int LU_gauss_main(const int brow, const int bcol,
                      const int iter, const int type);
    int LUbacksubst();
    int part_bsub(const int brow, const int bcol);
    double checksolve(int row, int offset, bool complete);

    int rows;             //number of rows in the matrix
    int brows;            //number of rows of blocks in the matrix
    int columns;          //number of columns in the matrix
    int bcolumns;         //number of columns of blocks in the matrix
    int allocblock;       //reflects amount of allocation done per thread
    int blocksize;        //reflects amount of computation per thread
    int litBlock;         //size of inner loop blocks
    naming::id_type _gid; //the instances gid
    lublock*** datablock; //stores the data being operated on
    double** factorData;  //stores factors for computations
    double** trueData;    //the original unaltered data
    double** transData;   //transpose of the original data(speeds up pivoting)
    double** solution;     //for storing the solution
    int* pivotarr;        //array for storing pivot elements
                          //(maps original rows to pivoted/reordered rows)
    int* tempivotarr;     //temporary copy of pivotarr
    lcos::local::mutex mtex;     //mutex

    public:
    //here we define the actions that will be used
    //the construct function
    typedef hpx::actions::result_action4<smphplmatrex, int, hpl_construct,
        naming::id_type, int, int, int, &smphplmatrex::construct>
        construct_action;
    //the destruct function
    typedef hpx::actions::action0<smphplmatrex, hpl_destruct,
        &smphplmatrex::destruct> destruct_action;
    //the assign function
    typedef hpx::actions::result_action4<smphplmatrex, int, hpl_assign, int,
        int, bool, int, &smphplmatrex::assign> assign_action;
    //the solve function
    typedef hpx::actions::result_action0<smphplmatrex, double, hpl_solve,
        &smphplmatrex::LUsolve> solve_action;
    //the search_pivots function
    typedef hpx::actions::result_action1<smphplmatrex, int, hpl_search, int,
        &smphplmatrex::search_pivots> search_action;
    //the swap function
    typedef hpx::actions::result_action2<smphplmatrex, int, hpl_swap, int,
        int, &smphplmatrex::swap> swap_action;
    //the main gaussian function
    typedef hpx::actions::result_action4<smphplmatrex, int, hpl_gmain, int,
        int, int, int, &smphplmatrex::LU_gauss_main> gmain_action;
    //part_bsub function
    typedef hpx::actions::result_action2<smphplmatrex, int, hpl_partbsub, int,
        int, &smphplmatrex::part_bsub> partbsub_action;
    //checksolve function
    typedef hpx::actions::result_action3<smphplmatrex, double, hpl_check, int,
        int, bool, &smphplmatrex::checksolve> check_action;

    //here begins the definitions of most of the future types that will be used
    //the first of which is for assign action
    typedef hpx::lcos::eager_future<server::smphplmatrex::assign_action> assign_future;
    //the search pivots future
    typedef hpx::lcos::eager_future<server::smphplmatrex::search_action> search_future;
    //Here is the swap future, which works the same way as the assign future
    typedef hpx::lcos::eager_future<server::smphplmatrex::swap_action> swap_future;
    //This future corresponds to the Gaussian elimination functions
    typedef hpx::lcos::eager_future<server::smphplmatrex::gmain_action> gmain_future;
    //the backsubst future is used to make sure all computations are complete
    //before returning from LUsolve, to avoid killing processes and erasing the
    //leftdata while it is still being worked on
    typedef
        hpx::lcos::eager_future<server::smphplmatrex::partbsub_action> partbsub_future;
    //the final future type for the class is used for checking the accuracy of
    //the results of the LU decomposition
    typedef hpx::lcos::eager_future<server::smphplmatrex::check_action> check_future;

    //right here are special arrays of futures
    gmain_future*** leftFutures, topFutures;
    gmain_future**** rightFutures, belowFutures, dagnlFutures;
    };
///////////////////////////////////////////////////////////////////////////////

    //the constructor initializes the matrix
    int smphplmatrex::construct(naming::id_type gid, int h, int ab, int bs){
// / / /initialize class variables/ / / / / / / / / / / /
    if(ab > std::ceil(((float)h)*.5)){
        allocblock = (int)std::ceil(((float)h)*.5);
    }
    else{ allocblock = ab;}
    if(bs > h){
        blocksize = h;
    }
    else{ blocksize = bs;}
    litBlock = 32;
    rows = h;
    brows = (int)std::floor((float)h/blocksize);
    columns = h+1;
    _gid = gid;
// / / / / / / / / / / / / / / / / / / / / / / / / / / /

    int i;          //just counting variables
    int offset = 1; //the initial offset used for the memory handling algorithm
    boost::rand48 gen;     //random generator used for seeding other generators
    gen.seed(time(NULL));
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
    assign_future future(gid,(int)0,offset,false,gen());

    //initialize the pivot array
    for(i=0;i<rows;i++){pivotarr[i]=tempivotarr[i]=i;}
    future.get();
    return 1;
    }

    //allocate() allocates memory space for the matrix
    void smphplmatrex::allocate(){
    datablock = new lublock**[brows];
    factorData = new double*[rows];
    transData = new double*[columns];
    trueData = new double*[rows];
    pivotarr = new int[rows];
    tempivotarr = new int[rows];
    for(int i = 0;i < rows;i++){
        trueData[i] = new double[columns];
        transData[i] = new double[rows];
    }
    transData[rows] = new double[rows];
    for(int i = 0;i < brows;i++){
        datablock[i] = new lublock*[brows];
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
    int smphplmatrex::assign(int row, int offset, bool complete, int seed){
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
    int temp = (std::min)((int)offset, (int)(rows - row));
    int location;
    for(int i=0;i<temp;i++){
        location = row+i;
        for(int j=0;j<columns;j++){
            transData[j][location] = trueData[location][j]
                                   = (double) (gen() % 1000);
        }
    }
    //once all spun off futures are complete we are done
    BOOST_FOREACH(assign_future af, futures){
        af.get();
    }
    return 1;
    }

    //the destructor frees the memory
    void smphplmatrex::destruct(){
    int i;
    for(i=0;i<rows;i++){
        free(trueData[i]);
        free(solution[i]);
    }
    for(i=0;i<brows;i++){
        for(int j=0;j<brows;j++){
            delete datablock[i][j];
        }
        free(datablock[i]);
    }
    free(datablock);
    free(pivotarr);
    free(trueData);
    free(solution);
    }

    //LUsolve is simply a wrapper function for LUfactor and LUbacksubst
    double smphplmatrex::LUsolve(){
    //first perform partial pivoting
    ptime starttime = ptime(microsec_clock::local_time());
    pivot();
    ptime temp = ptime(microsec_clock::local_time());
    std::cout<<"pivoting over "<<temp-starttime<<std::endl;

    //to initiate the Gaussian elimination
    LU_gauss_manager();
    ptime temp2 = ptime(microsec_clock::local_time());
    std::cout<<"finished gaussian "<<temp2 - temp<<std::endl;

    //allocate memory space to store the solution
    solution = new double*[rows];
    for(int i = 0; i < rows; i++){
        free(factorData[i]);
        solution[i] = new double[brows+1];
    }
    free(factorData);

    //perform back substitution
    LUbacksubst();
    ptime endtime = ptime(microsec_clock::local_time());
    std::cout<<"bsub done "<<endtime-temp2<<std::endl;
    std::cout<<"total LU time: "<<endtime-starttime<<std::endl;

    int h = (int)std::ceil(((float)rows)*.5);
    int offset = 1;
    while(offset < h){offset *= 2;}
    check_future chk(_gid,0,offset,false);
    return chk.get();
    }

    //pivot() finds the pivot element of each column and stores it. All
    //pivot elements for a number of rows equal to blocksize are found
    //before swapping takes place for those rows.  The swapping is done in
    //parallel, and while one group of columns are being searched for pivot
    //points other likely pivot points are identified in parallel as well.
    void smphplmatrex::pivot(){
    double max, temp;
    int maxRow, temp_piv, outer;
    int i=0,j;
    bool good;
    int* guessedPivots = new int[rows];
    std::vector<swap_future> futures;
    std::vector<search_future> searches;

    //each iteration finds pivot points for a blocksize number of rows.
    //The prediction of future pivot points only is performed for the first
    //half of the columns because later searches take less time than earlier
    //searches and eventually overhead takes up more time than is saved
    //from correct predictions.
    for(outer=0;outer<=brows;outer++){
//******involved in predictions**********************************************//
        if(outer < (brows/2)){
            searches.push_back(search_future(_gid,(outer+1)*blocksize));
        }
        if(outer > 0 && outer <= brows/2){
            searches[outer-1].get();
            temp = (outer-1)*blocksize;
            for(j=(int)temp;j<rows;j++){
                guessedPivots[j] = tempivotarr[j];
                tempivotarr[j] = pivotarr[j];
        }   }
//***************************************************************************//
        for(i=i;i<(std::min)((outer+1)*blocksize,rows-1);i++){
//**********involved in predictions******************************************//
            good = true;
            if(outer > 0 && outer <= brows/2){
                for(j=i-blocksize;j<i;j++){
                    if(pivotarr[j] == guessedPivots[i]){
                        good = false;
                        break;
            }   }   }
            else{good = false;}
            if(good){
                temp_piv = pivotarr[i];
                pivotarr[i] = guessedPivots[i];
                for(j=i-blocksize+1;j<rows;j++){
                    if(pivotarr[j] == pivotarr[i]){
                        pivotarr[j] = temp_piv;
                        break;
            }   }   }
//***************************************************************************//
            //This else statement is where the code for a normal pivot search
            //is contained
            else{
                maxRow = i;
                max = fabs(transData[i][pivotarr[i]]);
                temp_piv = pivotarr[i];
                for(j=i+1;j<rows;j++){
                    temp = fabs(transData[i][pivotarr[j]]);
                    if(temp > max){
                        max = temp;
                        maxRow = j;
                }   }
                pivotarr[i] = pivotarr[maxRow];
                pivotarr[maxRow] = temp_piv;
        }   }
        //here we begin swapping portions of the matrix we have finished
        //finding the pivot values for.  Due to how the lublocks are used
        //to represent the entire dataset, the second to last iteration
        //does not create a new swap future.
        if(outer<brows-1){futures.push_back(swap_future(_gid,outer,0));}
        else if(outer==brows){futures.push_back(swap_future(_gid,outer-1,0));}
    }
    //transData is no longer needed so free the memory and allocate
    //space for factorData
    for(i=0;i<rows;i++){
        free(transData[i]);
        factorData[i] = new double[i];
    }
    free(transData[rows]);
    free(transData);

    //ensure that all pivoting is complete
    BOOST_FOREACH(swap_future sf, futures){
        sf.get();
    }
    free(tempivotarr);
    }

    //search_pivots guesses where pivots will be to speed up the average
    //search time in the pivot() function
    int smphplmatrex::search_pivots(const int row){
    int i, j, maxRow, temp_piv;
    double max, temp;

    for(i=row;i<(std::min)(row+blocksize,rows);i++){
        maxRow = i-blocksize;
        max = fabs(transData[i][tempivotarr[i-blocksize]]);
        temp_piv = tempivotarr[i];
        for(j=i-blocksize+1;j<rows;j++){
            temp = fabs(transData[i][tempivotarr[j]]);
            if(temp > max){
                max = temp;
                maxRow = j;
        }   }
        tempivotarr[i] = tempivotarr[maxRow];
        tempivotarr[maxRow] = temp_piv;
    }
    return 1;
    }

    //swap() creates the datablocks and reorders the original
    //trueData matrix when assigning the initial values to the datablocks
    //according to the pivotarr data.  trueData itself remains unchanged
    int smphplmatrex::swap(const int brow, const int bcol){
    const int temp = rows/blocksize;
    int numrows = blocksize, numcols = blocksize;
    int i,j,k;
    for(k=0;k<brows;k++){
    if(brow == brows-1){numrows = rows - (temp-1)*blocksize;}
    if(k == brows-1){numcols = columns - (temp-1)*blocksize;}
    datablock[brow][k] = new lublock(numrows,numcols);
    for(i=0;i<numrows;i++){
        for(j=0;j<numcols;j++){
            datablock[brow][k]->data[i][j] =
            trueData[pivotarr[brow*blocksize+i]][k*blocksize+j];
    }   }
    }
    return 1;
    }

    //LU_gauss_manager creates futures as old futures finish their computations
    //Though not the perfect way of generating futures(a small amount of
    //starvation occurs), the manager ensures that the computation is done in
    //order with as many datablocks being operated on simultaneously as possible
    void smphplmatrex::LU_gauss_manager(){
/*    int iter, i,j,k;

    for(i = 0; i < brows-1; i++){
        leftFutures = new gmain_future
        for(j = 0; j < brows-1; j++){

    }
*/
    int iter = 0, i, j;
    int startElement, beginElement, endElement, nextElement;
    std::vector<gmain_future> futures;

    //the first iteration works different because we do not need to wait for
    //any futures to complete before creating new futures
    LU_gauss_corner(0);
    for(i = 1; i < brows; i++){
        futures.push_back(gmain_future(_gid,i,0,0,3));
    }
    endElement = futures.size()-1;
    for(i = 1; i < brows; i++){
        futures.push_back(gmain_future(_gid,0,i,0,2));
    }
    beginElement = futures.size();
    for(i = 0; i <= endElement; i++){
        futures[i].get();
    }
    for(i = 1; i < brows; i++){
        futures[endElement+i].get();
        for(j = 1; j < brows; j++){
            futures.push_back(gmain_future(_gid,j,i,0,1));
    }   }
    //from here on we need to wait for the previous iteration to partially
    //complete before launching new futures
    for(iter = 1; iter < brows; iter++){
        startElement = futures.size();
        futures[beginElement].get();
        LU_gauss_corner(iter);
        for(i = iter+1; i < brows; i++){
            futures[beginElement+i-iter].get();
            futures.push_back(gmain_future(_gid,i,iter,iter,3));
        }
        endElement = futures.size()-1;
        for(i = iter+1; i < brows; i++){
            futures[beginElement+(brows-iter)*(i-iter)].get();
            futures.push_back(gmain_future(_gid,iter,i,iter,2));
        }
        nextElement = futures.size();
        for(i = startElement; i <= endElement; i++){
            futures[i].get();
        }
        for(i = iter+1; i < brows; i++){
            futures[endElement+i-iter].get();
            for(j = iter+1; j < brows; j++){
                futures[beginElement+(brows-iter)*(i-iter)+j-iter].get();
                futures.push_back(gmain_future(_gid,j,i,iter,1));
        }   }
        beginElement = nextElement;
    }
    }

    //LUgaussmain is a wrapper function which is used so that only one type of
    //action is needed instead of three types of actions
    int smphplmatrex::LU_gauss_main(const int brow,const int bcol,const int iter,
        const int type){
    if(type == 1){
        LU_gauss_trail(brow,bcol,iter);
    }
    else if(type == 2){
        LU_gauss_top(iter,bcol);
    }
    else{
        LU_gauss_left(brow,iter);
    }

    return 1;
    }

    //LUgausscorner peforms gaussian elimination on the topleft corner block
    //of data that has not yet completed all of it's gaussian elimination
    //computations. Once complete, this block will need no further computations
    void smphplmatrex::LU_gauss_corner(const int iter){
    int i, j, k;
    const int offset = iter*blocksize;
    double fFactor;
    double factor;

    for(i=0;i<datablock[iter][iter]->rows;i++){
        if(datablock[iter][iter]->data[i][i] == 0){
            std::cerr<<"Warning: divided by zero\n";
        }
        fFactor = 1/datablock[iter][iter]->data[i][i];
        for(j=i+1;j<datablock[iter][iter]->rows;j++){
            factor = fFactor*datablock[iter][iter]->data[j][i];
            factorData[j+offset][i+offset] = factor;
            for(k=i+1;k<datablock[iter][iter]->columns;k++){
            datablock[iter][iter]->data[j][k] -=
                factor*datablock[iter][iter]->data[i][k];
    }   }   }
    }

    //LUgausstop performs gaussian elimination on the topmost row of blocks
    //that have not yet finished all gaussian elimination computation.
    //Once complete, these blocks will no longer need further computations
    void smphplmatrex::LU_gauss_top(const int iter, const int bcol){
    int i,j,k;
    const int offset = iter*blocksize;
    double factor;

    for(i=0;i<datablock[iter][bcol]->rows;i++){
        for(j=i+1;j<datablock[iter][bcol]->rows;j++){
            factor = factorData[j+offset][i+offset];
            for(k=0;k<datablock[iter][bcol]->columns;k++){
            datablock[iter][bcol]->data[j][k] -=
                factor*datablock[iter][bcol]->data[i][k];
    }   }   }
    }

    //LUgaussleft performs gaussian elimination on the leftmost column of
    //blocks that have not yet finished all gaussian elimination computation.
    //Upon completion, no further computations need be done on these blocks.
    void smphplmatrex::LU_gauss_left(const int brow, const int iter){
    int i,j,k;
    const int offset = brow*blocksize;
    const int offsetCol = iter*blocksize;
    double* fFactor = new double[datablock[brow][iter]->columns];
    double factor;

    //this first block of code finds all necessary factors early on
    //and allows for more efficient cache accesses for the second
    //block, which is where the majority of work is performed
    for(i=0;i<datablock[brow][iter]->columns;i++){
        fFactor[i] = 1/datablock[iter][iter]->data[i][i];
        factor = fFactor[i]*datablock[brow][iter]->data[0][i];
        factorData[offset][i+offsetCol] = factor;
        for(k=i+1;k<datablock[brow][iter]->columns;k++){
            datablock[brow][iter]->data[0][k] -=
                factor*datablock[iter][iter]->data[i][k];
    }   }
    for(j=1;j<datablock[brow][iter]->rows;j++){
        for(i=0;i<datablock[brow][iter]->columns;i++){
            factor = fFactor[i]*datablock[brow][iter]->data[j][i];
            factorData[j+offset][i+offsetCol] = factor;
            for(k=i+1;k<datablock[brow][iter]->columns;k++){
            datablock[brow][iter]->data[j][k] -=
                factor*datablock[iter][iter]->data[i][k];
    }    }   }
    }

    //LUgausstrail performs gaussian elimination on the trailing submatrix of
    //the blocks operated on during the current iteration of the Gaussian
    //elimination computations. These blocks will still require further
    //computations to be performed in future iterations.
    void smphplmatrex::LU_gauss_trail(const int brow, const int bcol,
        const int iter){
    int i,j,k,jj;
    const int offset = brow*blocksize;
    const int offsetCol = iter*blocksize;
    double factor,temp;

    //outermost loop: iterates over the fFactors of the most recent corner
    //block (fFactors are used indirectly through factorData)
    //middle loop: iterates over the rows of the current block
    //inner loop: iterates across the columns of the current block
    for(jj=0;jj<datablock[brow][bcol]->rows;jj+=litBlock){
    for(j=jj;j<(std::min)(jj+litBlock,datablock[brow][bcol]->rows);j++){
        for(i=0;i<datablock[iter][iter]->rows;i++){
            factor = factorData[j+offset][i+offsetCol];
            for(k=0;k<datablock[brow][bcol]->columns;k++){
            temp = factor*datablock[iter][bcol]->data[i][k];
            datablock[brow][bcol]->data[j][k] -= temp;
    }   }   }
    }}

    //this is an implementation of back substitution modified for use on
    //multiple datablocks instead of a single large data structure.
    //Additionally, a large amount of the work is performed in parallel. This
    //requires a significant amount of overhead, but the speedup is well worth
    //the additional work.
    int smphplmatrex::LUbacksubst(){
    int i,k,l,row,temp;
    int* neededFuture = new int[brows-1];
    std::vector<partbsub_future> futures;

    //first the solution values are initialized
    for(i=0;i<brows;i++){
        temp = i*blocksize;
        for(k=0;k<datablock[i][0]->rows;k++){
            for(l=0;l<brows;l++){solution[temp+k][l] = 0;}
            solution[temp+k][brows] =
            datablock[i][brows-1]->data[k][datablock[i][brows-1]->columns-1];
    }   }

    //next the first iteration is completed(we don't wait for futures here)
    i = brows-1;
    row = i*blocksize;
    for(k=datablock[i][i]->columns-2;k>=0;k--){
        temp = row+k;
        solution[temp][brows]/=datablock[i][i]->data[k][k];
        for(l=k-1;l>=0;l--){
            solution[row+l][brows] -=
                datablock[i][i]->data[l][k]*solution[temp][brows];
    }   }
    neededFuture[0] = futures.size();
    for(k=brows-2;k>=0;k--){futures.push_back(partbsub_future(_gid,k,i));}

    //the remaining iterations are performed in this block of code
    for(i=brows-2;i>=0;i--){
        row = i*blocksize;
        for(k=0;k<brows-i-1;k++){
            futures[neededFuture[k]].get();
            neededFuture[k]+=1;
        }
        for(k=row;k<row+blocksize;k++){
            for(l=i;l<brows;l++){
                solution[k][brows] += solution[k][l];
        }   }
        for(k=blocksize-1;k>=0;k--){
            temp = row+k;
            solution[temp][brows]/=datablock[i][i]->data[k][k];
            for(l=k-1;l>=0;l--){
                solution[row+l][brows] -=
                    datablock[i][i]->data[l][k]*solution[temp][brows];
        }   }
        neededFuture[brows-i-1] = futures.size();
        for(k=i-1;k>=0;k--){futures.push_back(partbsub_future(_gid,k,i));}
    }
    return 1;
    }

    //part_bsub performs backsubstitution on a single block of data
    //the function is designed to both take advantage of cache locality
    //and to allow fine grained parallelism during back substitution
    int smphplmatrex::part_bsub(const int brow, const int bcol){
        const int row = brow*blocksize, col = bcol*blocksize;
        int cols, i, j;
        if(bcol == brows-1){cols = datablock[bcol][bcol]->columns-1;}
        else{cols = blocksize;}

        for(i=0;i<blocksize;i++){
            for(j=0;j<cols;j++){
                solution[row+i][bcol] -=
                    datablock[brow][bcol]->data[i][j]*solution[col+j][brows];
            }
        }
        return bcol;
    }

    //finally, this function checks the accuracy of the LU computation a few
    //rows at a time
    double smphplmatrex::checksolve(int row, int offset, bool complete){
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
        int temp = (std::min)((int)offset, (int)(rows - row));
        int i,j;
        double sum;
        for(i=0;i<temp;i++){
            sum = 0;
            for(j=0;j<rows;j++){
                sum += trueData[pivotarr[row+i]][j] * solution[j][brows];
            }
            toterror += std::fabs(sum-trueData[pivotarr[row+i]][rows]);
        }

        //collect the results and add them up
        BOOST_FOREACH(check_future cf, futures){
            toterror += cf.get();
        }
        return toterror;
    }
}}}

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::smphplmatrex::construct_action,HPLconstruct_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::smphplmatrex::destruct_action,HPLdestruct_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::smphplmatrex::assign_action,HPLassign_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::smphplmatrex::solve_action,HPLsolve_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::smphplmatrex::search_action,HPLsearch_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::smphplmatrex::swap_action,HPLswap_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::smphplmatrex::gmain_action,HPLgmain_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::smphplmatrex::partbsub_action,HPLpartbsub_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::smphplmatrex::check_action,HPLcheck_action);

#endif
