//  Copyright (c) 2013 Jeremy Kemp
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Static blocked LU Decomposition

#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>

#include <hpx/dataflow.hpp>
#include <hpx/util/unwrapped.hpp>
#include <vector>

#include <hpx/util/lightweight_test.hpp>

#include <boost/assign.hpp>

#include <cstdio>
#include <cstdlib>

using std::vector;
using hpx::lcos::shared_future;
using hpx::wait_all;
using hpx::async;
using hpx::dataflow;
using hpx::util::unwrapped;

struct block {
    int size;
    int start;
    int height;
    block(int size, int startAddress, int H) : size(size),
        start(startAddress), height(H){}
    block() : size(0), start(0), height(0){}
};
void LU(int numBlocks);
void checkResult( vector<double> &originalA);

block ProcessDiagonalBlock( block B);
block ProcessBlockOnColumn( block B1, block B2);
block ProcessBlockOnRow( block B1, block B2);
block ProcessInnerBlock( block B1, block B2, block B3);

void getBlockList(vector<vector<block> > &blocks, int numBlocks);

void Print_Matrix(vector<double> &v);
void InitMatrix3();
void initLoop(int i);


vector<double> A;
vector<double> L;
vector<double> U;
int size = 200;

boost::uint64_t get_tick_count()
{
    return hpx::util::high_resolution_clock::now() / 1000;
}

int hpx_main (int argc, char *argv[])
{
    boost::uint64_t t1, t2;
    vector<double> originalA;
    int numBlocks = 20;

    if( argc > 1 )
        size = atoi(argv[1]);
    if( argc > 2 )
        numBlocks = atoi(argv[2]);
    printf("size = %d, numBlocks = %d\n", size, numBlocks);

    A.resize(size*size, 0);
    L.resize(size*size, 0);
    U.resize(size*size, 0);
    t1 = get_tick_count();
    InitMatrix3();
    t2 = get_tick_count();
    originalA.resize(size*size);
    for(int i = 0; i < size * size; i++)
        originalA[i] = A[i];
    printf("init done, time = %f\n", (t2-t1)/1000000.0);

    t1 = get_tick_count();
    if(numBlocks == 1)
        ProcessDiagonalBlock(block(size, 0, size));
    else if( numBlocks > 1)
        LU(numBlocks);
    else
        printf("Error: numBlocks must be greater than 0.\n");
    t2 = get_tick_count();
    printf("Time for LU-decomposition in secs: %f \n", (t2-t1)/1000000.0);

    checkResult( originalA );

    hpx::finalize();
    return hpx::util::report_errors();
}

int main(int argc, char *argv[])
{
    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(argc, argv, cfg);
}

void LU( int numBlocks)
{
    printf("LU\n");
    hpx::naming::id_type here = hpx::find_here();
    vector<vector<block> > blockList;
    getBlockList(blockList, numBlocks);
    vector<vector<vector<shared_future<block> > > > dfArray(numBlocks);
    shared_future<block> *diag_block, *first_col;

    for(int i = 0; i < numBlocks; i++){
        dfArray[i].resize(numBlocks);
        for(int j = 0; j < numBlocks; j++){
            dfArray[i][j].resize(numBlocks, hpx::make_ready_future(block()));
        }
    }
    //first iteration through matrix, initialized vector of futures
    dfArray[0][0][0] = async( ProcessDiagonalBlock, blockList[0][0] );
    diag_block = &dfArray[0][0][0];
    for(int i = 1; i < numBlocks; i++) {
        dfArray[0][0][i] = dataflow( unwrapped( &ProcessBlockOnRow ),
            hpx::make_ready_future( blockList[0][i] ), *diag_block);
    }
    for(int i = 1; i < numBlocks; i++) {
        dfArray[0][i][0] = dataflow( unwrapped( &ProcessBlockOnColumn ),
            hpx::make_ready_future( blockList[i][0] ), *diag_block);
        first_col = &dfArray[0][i][0];
        for(int j = 1; j < numBlocks; j++) {
            dfArray[0][i][j] = dataflow( unwrapped( &ProcessInnerBlock ),
                hpx::make_ready_future( blockList[i][j]), dfArray[0][0][j], *first_col );
        }
    }
    //all calculation after initialization. Each iteration,
    //the number of tasks/blocks spawned is decreased.
    for(int i = 1; i < numBlocks; i++) {
        dfArray[i][i][i] = dataflow( unwrapped( &ProcessDiagonalBlock ),
            dfArray[i-1][i][i]);
        diag_block = &dfArray[i][i][i];
        for(int j = i + 1; j < numBlocks; j++){
            dfArray[i][i][j] = dataflow( unwrapped(&ProcessBlockOnRow),
                dfArray[i-1][i][j], *diag_block);
        }
        for(int j = i + 1; j < numBlocks; j++){
            dfArray[i][j][i] = dataflow( unwrapped( &ProcessBlockOnColumn ),
                dfArray[i-1][j][i], *diag_block);
            first_col = &dfArray[i][j][i];
            for(int k = i + 1; k < numBlocks; k++) {
                dfArray[i][j][k] = dataflow( unwrapped( &ProcessInnerBlock ),
                    dfArray[i-1][j][k], dfArray[i][i][k], *first_col );
            }
        }
    }
    wait_all(dfArray[numBlocks-1][numBlocks-1][numBlocks-1]);
}

void getBlockList(vector<vector<block> > &blockList, int numBlocks)
{
    int blockSize, start, height;
    for(int i=0; i < numBlocks; i++)
        blockList.push_back(vector<block>());

    height = size/numBlocks;
    if(size%numBlocks > 0)
        height += 1;
    for(int i=0; i < numBlocks; i++) {
        if(i < size % numBlocks) {
            blockSize = size/numBlocks+1;
            start = (size/numBlocks+1)*i;
        } else {
            blockSize = size/numBlocks;
            start = (size/numBlocks+1)*(size%numBlocks)
                + (size/numBlocks)*(i-size%numBlocks);
        }
        blockList[0].push_back( block( blockSize, start, height));
    }
    for(int i = 1; i < numBlocks; i++) {
        height = blockList[0][i].size;
        for(int j = 0; j < numBlocks; j++) {
            blockSize = blockList[0][j].size;
            start = blockList[i-1][j].start + blockList[i-1][0].height * size;
            blockList[i].push_back( block( blockSize, start, height));
        }
    }
}

block ProcessDiagonalBlock( block B)
{
    for(int i = 0; i < B.size; i++) {
        for(int j = i+1; j < B.size; j++){
            A[B.start+j*size+i] /= A[B.start+i*size+i];
            for(int k = i+1; k < B.size; k++) {
                A[B.start+j*size+k] -= A[B.start+j*size+i] * A[B.start+i*size+k];
            }
        }
    }
    return  B;
}

block ProcessBlockOnColumn( block B1, block B2)
{
    for(int i=0; i < B2.size; i++) {
        for(int j=0; j < B1.height; j++) {
            A[B1.start+j*size+i] /= A[B2.start+i*size+i];
            for(int k = i+1; k < B2.size; k++) {
                A[B1.start+j*size+k] += -A[B1.start+j*size+i] * A[B2.start+i*size+k];
            }
        }
    }
    return B1;
}

block ProcessBlockOnRow( block B1, block B2)
{
    for(int i=0; i < B2.size; i++)
        for(int j=i+1; j < B2.size; j++)
            for(int k=0; k < B1.size; k++)
                A[B1.start+j*size+k] += -A[B2.start+j*size+i] * A[B1.start+i*size+k];
    return B1;
}

block ProcessInnerBlock( block B1, block B2, block B3)
{
    for(int i=0; i < B3.size; i++)
        for(int j=0; j < B1.height; j++)
            for(int k=0; k < B2.size; k++)
                A[B1.start+j*size+k] += -A[B3.start+j*size+i] * A[B2.start+i*size+k];
    return B1;
}

void checkResult( vector<double> &originalA )
{
    int errors = 0;
    double temp2;
    vector<double> L(size*size, 0);
    vector<double> U(size*size, 0);
    for(int i=0;i<size;i++)
        for(int j=0;j<size;j++)
            if (i>j)
                L[i*size+j] = A[i*size+j];
            else
                U[i*size+j] = A[i*size+j];
    for(int i=0;i<size;i++)
        L[i*size+i] = 1;

    for(int i=0;i<size;i++) {
        for(int j=0;j<size;j++){
            temp2=0;
            for(int k=0;k<size;k++)
                temp2+=L[i*size+k]*U[k*size+j];
            if( (originalA[i*size+j]-temp2) / originalA[i*size+j] > 0.1
                || (originalA[i*size+j]-temp2) / originalA[i*size+j] < -0.1 ){
                printf("error:[%d][%d] ", i, j);
                errors++;
            }
        }
    }
    HPX_TEST_EQ(errors, 0);
}

void Print_Matrix(vector<double> &v)
{
    printf( "\n" );
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++)
            printf( "%5.1f, ", v[i*size + j] );
        printf( "\n" );
    }
    printf( "\n" );
}

void InitMatrix3()
{
    vector<shared_future<void> > futures;
    futures.reserve(size);
    for(int i = 0; i < size; i++)
        for(int j = 0; j < size; j++){
            if(i >= j)
                L[i*size + j] = i-j+1;
            else
                L[i*size + j] = 0;
            if(i <= j)
                U[i*size + j] = j-i+1;
            else
                U[i*size + j] = 0;
        }
    for(int i = 0; i < size; i++) {
        futures.push_back( async( initLoop, i));
    }
    wait_all(futures);
}

void initLoop(int i) {
    for(int j = 0; j < size; j++)
        for(int k = 0; k < size; k++)
            A[i*size + j] += L[i*size + k] * U[k*size + j];
}

