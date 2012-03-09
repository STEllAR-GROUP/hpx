///////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0.(See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////
#ifndef _HPLMATREX_SERVER_HPP
#define _HPLMATREX_SERVER_HPP

/*This is the hplmatrex class implementation header file.
In order to keep things simple, only operations necessary
to to perform luP decomposition are declared, which is
basically just constructors, assignment operators,
a destructor, and access operators.
*/

#include <time.h>

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include <boost/foreach.hpp>
#include <boost/random/linear_congruential.hpp>
#include "../lublock.hpp"

#include <boost/date_time/posix_time/posix_time.hpp>
using namespace boost::posix_time;

namespace hpx { namespace components { namespace server
{
    class HPX_COMPONENT_EXPORT hplmatrex :
          public simple_component_base<hplmatrex>
    {
    public:
    //enumerate all of the actions that will(or can) be employed
    enum actions{
        hpl_construct,
        hpl_assign,
        hpl_partbsub,
        hpl_solve,
        hpl_swap,
        hpl_search,
        hpl_check
    };

    //constructors and destructor
    hplmatrex(){}
    int construct(naming::id_type gid, int h, int ab, int bs);
    ~hplmatrex(){destruct();}
    void destruct();

    double lusolve();

    private:
    void allocate();
    int assign(int row, int offset, bool complete, int seed);
    void pivot();
    int search_pivots(const int row);
    int swap(const int brow, const int bcol);
    int lubacksubst();
    int part_bsub(const int brow, const int bcol);
    double checksolve(int row, int offset, bool complete);
//    void print();
//    void print2();

    int rows;             //number of rows in the matrix
    int brows;            //number of rows of blocks in the matrix
    int columns;          //number of columns in the matrix
    int bcolumns;         //number of columns of blocks in the matrix
    int allocblock;       //reflects amount of allocation done per thread
    int blocksize;        //reflects amount of computation per thread
    naming::id_type _gid; //the instance's gid
    components::lublock** datablock;  //stores pointers to data components
    std::vector<std::vector<naming::id_type> > gidList;
                          //the gids of the array of lublock components
    double** trueData;    //the original unaltered data
    double** transData;   //transpose of the original data(speeds up pivoting)
    double** solution;    //for storing the solution
    int* pivotarr;        //array for storing pivot elements
                          //(maps original rows to pivoted/reordered rows)
    int* tempivotarr;     //temporary copy of pivotarr
    lcos::local::mutex mtex;     //mutex

    public:
    //here we define the actions that will be used
    //the construct function
    typedef hpx::actions::result_action4<hplmatrex, int, hpl_construct,
        naming::id_type, int, int, int, &hplmatrex::construct>
        construct_action;
    //the assign function
    typedef hpx::actions::result_action4<hplmatrex, int, hpl_assign, int,
        int, bool, int, &hplmatrex::assign> assign_action;
    //the solve function
    typedef hpx::actions::result_action0<hplmatrex, double, hpl_solve,
        &hplmatrex::lusolve> solve_action;
    //the search_pivots function
    typedef hpx::actions::result_action1<hplmatrex, int, hpl_search, int,
        &hplmatrex::search_pivots> search_action;
    //the swap function
    typedef hpx::actions::result_action2<hplmatrex, int, hpl_swap, int,
        int, &hplmatrex::swap> swap_action;
    //part_bsub function
    typedef hpx::actions::result_action2<hplmatrex, int, hpl_partbsub, int,
        int, &hplmatrex::part_bsub> partbsub_action;
    //checksolve function
    typedef hpx::actions::result_action3<hplmatrex, double, hpl_check, int,
        int, bool, &hplmatrex::checksolve> check_action;

    //here begins the definitions of most of the future types that will be used
    //the first of which is for assign action
    typedef hpx::lcos::future<int> assign_future;
    //the search pivots future
    typedef hpx::lcos::future<int> search_future;
    //Here is the swap future
    typedef hpx::lcos::future<int> swap_future;
    //the backsubst future is used to make sure all computations are complete
    //before returning from lusolve, to avoid killing processes and erasing the
    //leftdata while it is still being worked on
    typedef hpx::lcos::future<int> partbsub_future;
    //the final future type for the class is used for checking the accuracy of
    //the results of the lu decomposition
    typedef hpx::lcos::future<double> check_future;
    };
///////////////////////////////////////////////////////////////////////////////

    //the constructor initializes the matrix
    int hplmatrex::construct(naming::id_type gid, int h, int ab, int bs){
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
    columns = h+1;
    _gid = gid;
// / / / / / / / / / / / / / / / / / / / / / / / / / / /

    int i,j;        //just counting variables
    int offset = 1; //the initial offset used for the memory handling algorithm
    boost::rand48 gen;     //random generator used for seeding other generators
    gen.seed(time(NULL));
    allocate();     //allocate memory for the elements of the array

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
    assign_future future =
        hpx::lcos::async<server::hplmatrex::assign_action>(gid,(int)0,offset,false,gen());

    //initialize the pivot array
    for(i=0;i<rows;i++){pivotarr[i]=tempivotarr[i]=i;}
    for(i=0;i<brows;i++){
        std::vector<naming::id_type> vectorRow;
        gidList.push_back(vectorRow);
        for(j=0;j<brows;j++){
            naming::id_type prefix =
                hpx::applier::get_applier().get_runtime_support_gid();
            datablock[i][j].create(prefix);
            gidList[i].push_back(datablock[i][j].get_gid());
    }   }
    future.get();
    return 1;
    }

    //allocate() allocates memory space for the matrix
    void hplmatrex::allocate(){
    datablock = new components::lublock*[brows];
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
        datablock[i] = new components::lublock[brows];
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
    int hplmatrex::assign(int row, int offset, bool complete, int seed){
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
        }   }
        //initialize the assigned row
        int temp = (std::min)((int)offset, (int)(rows - row));
        int location;
        for(int i=0;i<temp;i++){
            location = row+i;
            for(int j=0;j<columns;j++){
                transData[j][location] = trueData[location][j]
                                   = (double) (gen() % 1000);
        }   }
        //once all spun off futures are complete we are done
        BOOST_FOREACH(assign_future af, futures){
            af.get();
        }
        return 1;
    }

    //the destructor frees the memory
    void hplmatrex::destruct(){
    int i;
    for(i=0;i<rows;i++){
        free(trueData[i]);
        free(solution[i]);
    }
    for(i=0;i<brows;i++){
        for(int j=0;j<brows;j++){
            datablock[i][j].free();
        }
        free(datablock[i]);
    }
    free(datablock);
    free(pivotarr);
    free(trueData);
    free(solution);
    }

    //lusolve is simply a wrapper function for all subfunctions
    double hplmatrex::lusolve(){
    //first perform partial pivoting
    ptime starttime = ptime(microsec_clock::local_time());
    pivot();
    ptime temp = ptime(microsec_clock::local_time());
    std::cout<<"pivoting over "<<temp-starttime<<std::endl;

    //to initiate the Gaussian elimination
    for(int iter = 0; iter < brows; iter++){
        lublock::gcFuture(
            datablock[iter][iter].gauss_corner(iter)).get();
    }
    ptime temp2 = ptime(microsec_clock::local_time());
    std::cout<<"finished gaussian "<<temp2 - temp<<std::endl;

    //allocate memory space to store the solution
    solution = new double*[rows];
    for(int i = 0; i < rows; i++){solution[i] = new double[brows+1];}

    //perform back substitution
    lubacksubst();
    ptime endtime = ptime(microsec_clock::local_time());
    std::cout<<"bsub done "<<endtime-temp2<<std::endl;
    std::cout<<"total lu time: "<<endtime-starttime<<std::endl;

    int h = (int)std::ceil(((float)rows)*.5);
    int offset = 1;
    while(offset < h){offset *= 2;}
    return async<server::hplmatrex::check_action>(_gid,0,offset,false).get();
    }

    //pivot() finds the pivot element of each column and stores it. All
    //pivot elements for a number of rows equal to blocksize are found
    //before swapping takes place for those rows.  The swapping is done in
    //parallel, and while one group of columns are being searched for pivot
    //points other likely pivot points are identified in parallel as well.
    void hplmatrex::pivot(){
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
            searches.push_back(
                hpx::lcos::async<server::hplmatrex::search_action>(
                    _gid,(outer+1)*blocksize));
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
        if(outer<brows-1){
            futures.push_back(async<server::hplmatrex::swap_action>(_gid,outer,0));
        }
        else if(outer==brows){
            futures.push_back(async<server::hplmatrex::swap_action>(_gid,outer-1,0));
        }
    }
    //transData is no longer needed so free the memory and allocate
    //space for factorData
    for(i=0;i<rows;i++){
        free(transData[i]);
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
    int hplmatrex::search_pivots(const int row){
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
    int hplmatrex::swap(const int brow, const int bcol){
    const int temp = rows/blocksize;
    int numrows = blocksize, numcols = blocksize;
    int i,j,k;
    std::vector<std::vector<double> > tempData;

    if(brow == brows-1){numrows = rows - (temp-1)*blocksize;}
    for(k=0;k<brows;k++){
        if(k == brows-1){numcols = columns - (temp-1)*blocksize;}
        for(i=0;i<numrows;i++){
            std::vector<double> row;
            tempData.push_back(row);
            int off1 = brow*blocksize, off2 = k*blocksize;
            for(j=0;j<numcols;j++){
                tempData[i].push_back(trueData[pivotarr[off1+i]][off2+j]);
        }   }
        datablock[brow][k].construct_block(
            numrows,numcols,k,brow,brows,gidList,tempData);
        tempData.clear();
    }
    return 1;
    }

    //this is an implementation of back substitution modified for use on
    //multiple datablocks instead of a single large data structure.
    //Additionally, a large amount of the work is performed in parallel. This
    //requires a significant amount of overhead, but the speedup is well worth
    //the additional work.
    int hplmatrex::lubacksubst(){
    int i,k,l,row,temp;
    int* neededFuture = new int[brows-1];
    std::vector<partbsub_future> futures;
    std::vector<std::vector<double> > tempData;

    //first the solution values are initialized
    for(i=0;i<brows;i++){
        temp = i*blocksize;
        tempData.clear();
        tempData = datablock[i][brows-1].get_data();
        int haveColumns = datablock[i][brows-1].get_columns()-1;
        for(k=0;k<datablock[i][0].get_rows();k++){
            for(l=0;l<brows;l++){solution[temp+k][l] = 0;}
            solution[temp+k][brows] = tempData[k][haveColumns];
    }   }

    //next the first iteration is completed(we don't wait for futures here)
    i = brows-1;
    row = i*blocksize;
    tempData.clear();
    tempData = datablock[i][i].get_data();
    for(k=datablock[i][i].get_columns()-2;k>=0;k--){
        temp = row+k;
        solution[temp][brows]/=tempData[k][k];
        for(l=k-1;l>=0;l--){
            solution[row+l][brows] -=
                tempData[l][k]*solution[temp][brows];
    }   }
    neededFuture[0] = futures.size();
    for(k=brows-2;k>=0;k--){futures.push_back(async<server::hplmatrex::partbsub_action>(_gid,k,i));}

    //the remaining iterations are performed in this block of code
    for(i=brows-2;i>=0;i--){
        row = i*blocksize;
        tempData.clear();
        tempData = datablock[i][i].get_data();
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
            solution[temp][brows]/=tempData[k][k];
            for(l=k-1;l>=0;l--){
                solution[row+l][brows] -= tempData[l][k]*solution[temp][brows];
        }   }
        neededFuture[brows-i-1] = futures.size();
        for(k=i-1;k>=0;k--){futures.push_back(async<server::hplmatrex::partbsub_action>(_gid,k,i));}
    }
    return 1;
    }

    //part_bsub performs backsubstitution on a single block of data
    //the function is designed to both take advantage of cache locality
    //and to allow fine grained parallelism during back substitution
    int hplmatrex::part_bsub(const int brow, const int bcol){
        const int row = brow*blocksize, col = bcol*blocksize;
        int cols, i, j;
        std::vector<std::vector<double> > tempData;
        tempData = datablock[brow][bcol].get_data();
        if(bcol == brows-1){cols = datablock[bcol][bcol].get_columns()-1;}
        else{cols = blocksize;}

        for(i=0;i<blocksize;i++){
            for(j=0;j<cols;j++){
                solution[row+i][bcol] -= tempData[i][j]*solution[col+j][brows];
            }
        }
        return bcol;
    }

    //finally, this function checks the accuracy of the lu computation a few
    //rows at a time
    double hplmatrex::checksolve(int row, int offset, bool complete){
    double toterror = 0;    //total error from all checks
        //futures is used to allow this thread to continue spinning off new
        //thread while other threads work, and is checked at the end to make
        // certain all threads are completed before returning.
        std::vector<check_future> futures;

        //start spinning off work to do
        while(!complete){
            if(offset <= allocblock){
                if(row + offset < rows){
                    futures.push_back(async<server::hplmatrex::check_action>(_gid,
                        row+offset,offset,true));
                }
                complete = true;
            }
            else{
                if(row + offset < rows){
                    futures.push_back(async<server::hplmatrex::check_action>(_gid,row+offset,
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
    hpx::components::server::hplmatrex::construct_action,HPLconstruct_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::hplmatrex::assign_action,HPLassign_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::hplmatrex::solve_action,HPLsolve_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::hplmatrex::search_action,HPLsearch_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::hplmatrex::swap_action,HPLswap_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::hplmatrex::partbsub_action,HPLpartbsub_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::hplmatrex::check_action,HPLcheck_action);

#endif
