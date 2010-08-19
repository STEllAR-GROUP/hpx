#ifndef _HPLMATREX3_SERVER_HPP
#define _HPLMATREX3_SERVER_HPP

/*This is the HPLMatreX3 class implementation header file.
This includes constructors, assignment operators,
a destructor, and access operators, as well as many
computational functions.
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
#include "LUblock.hpp"

namespace hpx { namespace components { namespace server
{
    class HPX_COMPONENT_EXPORT HPLMatreX3 : public simple_component_base<HPLMatreX3>
    {
    public:
	//enumerate all of the actions that will(or can) be employed
        enum actions{
                hpl_construct=0,
                hpl_destruct=1,
                hpl_get=2,
                hpl_set=3,
                hpl_solve=4,
		hpl_ghub=5,
		hpl_bsubst=6,
		hpl_check=7
        };

	//constructors and destructor
	HPLMatreX3(){}
	int construct(naming::id_type gid, unsigned int h, unsigned int w,
		unsigned int ab, unsigned int bs);
	~HPLMatreX3(){destruct();}
	void destruct();

	//functions for assignment and leftdata access
	double get(unsigned int row, unsigned int col);
	void set(unsigned int row, unsigned int col, double val);

	//functions for manipulating the matrix
	double LUsolve();

    private:
	void allocate();
	int assign();
	void pivot();
	int swap(unsigned int brow, unsigned int bcol);
	void LUdivide();
	int LUgausshub(unsigned int brow, unsigned int bcol, unsigned int iter, bool full);
	void LUgausscorner(unsigned int iter);
	void LUgausstop(unsigned int iter, unsigned int bcol);
	void LUgaussleft(unsigned int brow, unsigned int iter);
	void LUgausstrail(unsigned int brow, unsigned int bcol, unsigned int iter);
	int LUbacksubst();
	double checksolve(unsigned int row, unsigned int offset, bool complete);
	void print();

	int rows;			//number of rows in the matrix
	int brows;			//number of rows of blocks in the matrix
	int columns;			//number of columns in the matrix
	int bcolumns;			//number of columns of blocks in the matrix
	int allocblock;			//reflects amount of allocation done per thread
	int blocksize;			//reflects amount of computation per thread
	naming::id_type _gid;		//the instances gid
	LUblock*** datablock;		//stores the data being operated on
	double** factordata;		//stores factors for computations
	double** truedata;		//the original unaltered data
        double* solution;		//for storing the solution
	int* pivotarr;			//array for storing pivot elements

    public:
	//here we define the actions that will be used
	//the construct function
	typedef actions::result_action5<HPLMatreX3, int, hpl_construct, naming::id_type,
		unsigned int, unsigned int, unsigned int, unsigned int,
		&HPLMatreX3::construct> construct_action;
	//the destruct function
	typedef actions::action0<HPLMatreX3, hpl_destruct,
		&HPLMatreX3::destruct> destruct_action;
	//the get function
	typedef actions::result_action2<HPLMatreX3, double, hpl_get, unsigned int,
        	unsigned int, &HPLMatreX3::get> get_action;
	//the set function
	typedef actions::action3<HPLMatreX3, hpl_set, unsigned int,
        	unsigned int, double, &HPLMatreX3::set> set_action;
	//the solve function
	typedef actions::result_action0<HPLMatreX3, double, hpl_solve,
		&HPLMatreX3::LUsolve> solve_action;
	//the top gaussian function
	typedef actions::result_action4<HPLMatreX3, int, hpl_ghub, unsigned int,
		unsigned int, unsigned int, bool, &HPLMatreX3::LUgausshub> ghub_action;
	//backsubstitution function
	typedef actions::result_action0<HPLMatreX3, int, hpl_bsubst,
		&HPLMatreX3::LUbacksubst> bsubst_action;
	//checksolve function
	typedef actions::result_action3<HPLMatreX3, double, hpl_check, unsigned int,
		unsigned int, bool, &HPLMatreX3::checksolve> check_action;

	//the backsubst future is used to make sure all computations are complete before
	//returning from LUsolve, to avoid killing processes and erasing the leftdata while
	//it is still being worked on
	typedef lcos::eager_future<server::HPLMatreX3::bsubst_action> bsubst_future;

	//the final future type for the class is used for checking the accuracy of
	//the results of the LU decomposition
	typedef lcos::eager_future<server::HPLMatreX3::check_action> check_future;
    };
//////////////////////////////////////////////////////////////////////////////////////

    //the constructor initializes the matrix
    int HPLMatreX3::construct(naming::id_type gid, unsigned int h, unsigned int w,
		unsigned int ab, unsigned int bs){
// / / /initialize class variables/ / / / / / / / / / / /
	if(ab > std::ceil(((float)h)*.5)){
		allocblock=std::ceil(((float)h)*.5);}
	else{allocblock=ab;}
	if(bs > h){
		blocksize=h;}
	else{blocksize=bs;}
	rows=h;
	brows=std::ceil((float)h/blocksize);
	columns=w;
	bcolumns=std::ceil((float)w/blocksize);
	_gid = gid;
// / / / / / / / / / / / / / / / / / / / / / / / / / / /

	int i; 			 //just counting variable
	unsigned int offset = 1; //the initial offset used for the memory handling algorithm

	datablock = (LUblock***) std::malloc(brows*sizeof(LUblock**));
	factordata = (double**) std::malloc(h*sizeof(double*));
	truedata = (double**) std::malloc(h*sizeof(double*));
	pivotarr = (int*) std::malloc(h*sizeof(int));
	srand(time(NULL));

	//by making offset a power of two, the allocate and assign functions
	//are much simpler than they would be otherwise
	h=std::ceil(((float)h)*.5);
	while(offset < h){offset *= 2;}

	//allocate the data
	allocate();

	//assign initial values to the truedata
	assign();

	//initialize the pivot array
	for(i=0;i<rows;i++){pivotarr[i]=i;}

	return 1;
    }

    //allocate() allocates memory space for the matrix
    void HPLMatreX3::allocate(){
	for(unsigned int i = 0;i < rows;i++){
	    truedata[i] = (double*) std::malloc(columns*sizeof(double));
	    factordata[i] = (double*) std::malloc(i*sizeof(double));
	}
	for(unsigned int i = 0;i < brows;i++){
	    datablock[i] = (LUblock**)std::malloc(bcolumns*sizeof(LUblock*));
	}
    }

    //assign gives values to the empty elements of the array
    int HPLMatreX3::assign(){
	for(unsigned int i=0;i<rows;i++){
	    for(unsigned int j=0;j<columns;j++){
		truedata[i][j] = (double) (rand() % 1000);
	    }
	}
	return 1;
    }

    //the destructor frees the memory
    void HPLMatreX3::destruct(){
	unsigned int i;
	for(i=0;i<rows;i++){
		free(truedata[i]);
	}
	for(i=0;i<brows;i++){
		for(int j=i;j<bcolumns;j++){
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
    double HPLMatreX3::get(unsigned int row, unsigned int col){
	return truedata[row][col];
    }

    //set() assigns a value to an element in all matrices
    void HPLMatreX3::set(unsigned int row, unsigned int col, double val){
	truedata[row][col] = val;
    }

    //print out the matrix
    void HPLMatreX3::print(){
	for(int i = 0;i < rows; i++){
	    for(int j = 0;j < columns; j++){
		std::cout<<datablock[i/blocksize][j/blocksize]->get(i%blocksize,j%blocksize)<<" ";
	    }
	    std::cout<<std::endl;
	}
	    std::cout<<std::endl;
    }
//END DEBUGGING FUNCTIONS/////////////////////////////////////////////

    //LUsolve is simply a wrapper function for LUfactor and LUbacksubst
    double HPLMatreX3::LUsolve(){
	pivot();
	LUdivide();
	//allocate memory space to store the solution
	solution = (double*) std::malloc(rows*sizeof(double));
	bsubst_future bsub(_gid);

	unsigned int h = std::ceil(((float)rows)*.5);
	unsigned int offset = 1;
	while(offset < h){offset *= 2;}
	bsub.get();
	check_future chk(_gid,0,offset,false);
	return chk.get();
	return 1;
    }

    //pivot() finds the pivot element of each column and stores it. All
    //pivot elements are found before any swapping takes place so that
    //the amount of swapping performed is minimized
    void HPLMatreX3::pivot(){
	unsigned int max, max_row;
	unsigned int temp_piv;
	double temp;
	unsigned int h = std::ceil(((float)rows)*.5);

	//by using pivotarr[] we can obtain all pivot values
	//without actually needing to swap any rows until
	//creating the datablock matrix
	for(unsigned int i=0;i<rows-1;i++){
	    max_row = i;
	    max = fabs(truedata[pivotarr[i]][i]);
	    temp_piv = pivotarr[i];
	    for(unsigned int j=i+1;j<rows;j++){
		temp = fabs(truedata[pivotarr[j]][i]);
		if(temp > max){
		    max = temp;
		    max_row = j;
		}
	    }
	    pivotarr[i] = pivotarr[max_row];
	    pivotarr[max_row] = temp_piv;
	}
    }

    //swap() creates the datablocks and reorders the original
    //truedata matrix when assigning the initial values to the datablocks
    //according to the pivotarr data.  truedata itself remains unchanged
    int HPLMatreX3::swap(unsigned int brow, unsigned int bcol){
	bool onbottom=false,onright=false;
	unsigned int temp = rows/blocksize;
	unsigned int numrows = blocksize, numcols = blocksize;
	if(brow == temp){numrows = rows - temp*blocksize;}
	if(bcol == temp){numcols = columns - temp*blocksize;}

	datablock[brow][bcol] = new LUblock(numrows,numcols);
	for(unsigned int i=0;i<numrows;i++){
	    for(unsigned int j=0;j<numcols;j++){
		datablock[brow][bcol]->set(i, j,
		    truedata[pivotarr[brow*blocksize+i]][bcol*blocksize+j]);
    }   }   }

    //LUdivide is a wrapper function for the gaussian elimination functions,
    //although the process of properly dividing up the computations across
    //the different blocks using slightly different functions does require
    //the delegation provided by this wrapper
    void HPLMatreX3::LUdivide(){
	typedef lcos::eager_future<server::HPLMatreX3::ghub_action> ghub_future;

	std::vector<ghub_future> hub_futures;	//used to determine when all work completes
	unsigned int iteration = 0;		//the iteration of the Gaussian elimination
	unsigned int brow, bcol, temp;		//various helping variables

	//initialize the first row and column of blocks
	//for the datablock array
	for(int i=0;i<brows;i++){
		swap(i,0);
	}
	for(int i=1;i<bcolumns;i++){
		swap(0,i);
	}

	do{
		hub_futures.clear();
		//FIRST perform as much work in parallel as possible
		hub_futures.push_back(ghub_future(_gid,iteration,iteration,iteration,false));
		brow = bcol = iteration+1;
		if(iteration + 1 < bcolumns){
			//here we simultaneously perform gaussian elimination on
			//the top row of blocks and the left row of blocks
			while(bcol < bcolumns){
				hub_futures.push_back(ghub_future(_gid,iteration,
				    bcol,iteration,false));
				bcol++;
			}
			while(brow < brows){
				hub_futures.push_back(ghub_future(_gid,brow,
				    iteration,iteration,false));
				brow++;
			}
		}

		//while the row and column are being processed, go ahead and create
		//the next row and column in memory and delete unneeded blocks
		temp = iteration+1;
		//by using temp, we don't need to calculate iteration+1 for each call
		for(int i = temp;i<brows;i++){swap(i,temp);}
		for(int i=temp+1;i<bcolumns;i++){swap(temp,i);}
		if(iteration > 0){
			temp -= 2;
			for(int i=iteration;i<brows;i++){delete datablock[i][temp];}
		}

		//NEXT make sure that the corner block is ready to run its final
		//iteration of gaussian elimination
		hub_futures[0].get();

		//NOW perform the corner block's last iteration
		LUgausscorner(iteration);
		brow = bcol = iteration+1;
		BOOST_FOREACH(ghub_future gf, hub_futures){
			gf.get();
		}
		hub_futures.clear();

		//FINALLY finish Gaussian elimination on the top and left rows
		if(iteration + 1 < bcolumns){
			//here we simultaneously perform gaussian elimination on
			//the top row of blocks and the left row of blocks
			while(bcol < bcolumns){
				hub_futures.push_back(ghub_future(_gid,iteration,
				    bcol,iteration,true));
				bcol++;
			}
			while(brow < brows){
				hub_futures.push_back(ghub_future(_gid,brow,
				    iteration,iteration,true));
				brow++;
			}
		}

		iteration++;
		//make sure all computations are done before moving on to the
		//next iteration (or exiting the function)
		BOOST_FOREACH(ghub_future gf, hub_futures){
			gf.get();
		}
	}while(iteration < brows);
    }

    //LUgausshub() is a wrapper function for the gaussian elimination of a single data block.
    //it delegates which Gaussian elimination function should be performed when.
    int HPLMatreX3::LUgausshub(unsigned int brow, unsigned int bcol, unsigned int iter, bool final){
	if(final){
	    if(brow < bcol){LUgausstop(brow,bcol);}
	    else{LUgaussleft(brow,bcol);}
	    datablock[brow][bcol]->increment();
	}
	//if this is not the final iteration, then the location of the block on the matrix is
	//irrelevant, as the block needs to catch up to the final iteration through several
	//calls to LUgausstrail
	else{
	    while(iter > datablock[brow][bcol]->getiteration()){
		LUgausstrail(brow,bcol,datablock[brow][bcol]->getiteration());
		datablock[brow][bcol]->increment();
	    }
	}
	return 1;
    }

    //LUgausscorner peforms gaussian elimination on the topleft corner block
    //of data that has not yet completed all of it's gaussian elimination
    //computations. Once complete, this block will need no further computations
    void HPLMatreX3::LUgausscorner(unsigned int iter){
	unsigned int i, j, k;
	unsigned int offset = iter*blocksize;
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
    }	}	}    }

    //LUgausstop performs gaussian elimination on the topmost row of blocks
    //that have not yet finished all gaussian elimination computation.
    //Once complete, these blocks will no longer need further computations
    void HPLMatreX3::LUgausstop(unsigned int iter, unsigned int bcol){
	unsigned int i,j,k;
	unsigned int offset = iter*blocksize;

	for(i=0;i<datablock[iter][bcol]->getrows();i++){
		for(j=i+1;j<datablock[iter][bcol]->getrows();j++){
		    for(k=0;k<datablock[iter][bcol]->getcolumns();k++){
			datablock[iter][bcol]->set(j,k,datablock[iter][bcol]->get(j,k) -
			    factordata[j+offset][i+offset]*datablock[iter][bcol]->get(i,k));
    }	}	}    }

    //LUgaussleft performs gaussian elimination on the leftmost column of blocks
    //that have not yet finished all gaussian elimination computation.
    //Upon completion, no further computations need be done on these blocks.
    void HPLMatreX3::LUgaussleft(unsigned int brow, unsigned int iter){
	unsigned int i,j,k;
	unsigned int offsetd = brow*blocksize;	//offset down
	unsigned int offseta = iter*blocksize;	//offset across
	double f_factor;

	for(i=0;i<datablock[brow][iter]->getcolumns();i++){
		f_factor = 1/datablock[iter][iter]->get(i,i);
		for(j=0;j<datablock[brow][iter]->getrows();j++){
		    factordata[j+offsetd][i+offseta] = f_factor*datablock[brow][iter]->get(j,i);
		    for(k=i+1;k<datablock[brow][iter]->getcolumns();k++){
			datablock[brow][iter]->set(j,k,datablock[brow][iter]->get(j,k) -
			    factordata[j+offsetd][i+offseta]*datablock[iter][iter]->get(i,k));
    }	}	}    }

    //LUgausstrail performs gaussian elimination on the trailing submatrix of
    //the blocks operated on during the current iteration of the Gaussian elimination
    //computations. These blocks will still require further computations to be
    //performed in future iterations.
    void HPLMatreX3::LUgausstrail(unsigned int brow, unsigned int bcol, unsigned int iter){
	unsigned int i,j,k;
	unsigned int offsetd = brow*blocksize;	//offset down
	unsigned int offseta = iter*blocksize;	//offset across

	//outermost loop: iterates over the f_factors of the most recent corner block
	//	(f_factors are used indirectly through factordata)
	//middle loop: iterates over the rows of the current block
	//inner loop: iterates across the columns of the current block
	for(i=0;i<datablock[iter][iter]->getrows();i++){
		for(j=0;j<datablock[brow][bcol]->getrows();j++){
		    for(k=0;k<datablock[brow][bcol]->getcolumns();k++){
			datablock[brow][bcol]->set(j,k,datablock[brow][bcol]->get(j,k) -
			    factordata[j+offsetd][i+offseta]*datablock[iter][bcol]->get(i,k));
    }	}	}    }

    //this is an implementation of back substitution modified for use on
    //multiple datablocks instead of a single large data structure
    int HPLMatreX3::LUbacksubst(){
	//i,j,k,l standard loop variables, row and col keep
	//track of where the loops would be in terms of a single
	//large data structure; using it allows addition
	//where multiplication would need to be used without it
        int i,j,k,l,row,col;
	unsigned int temp = datablock[0][bcolumns-1]->getcolumns()-1;

	for(i=0;i<rows;i++){
	    solution[i]=datablock[i/blocksize][bcolumns-1]->get(i%blocksize,temp);
	}

        for(i=brows-1;i>=0;i--){
	    row = i*blocksize;
	    for(j=bcolumns-1;j>=i;j--){
		col = j*blocksize;
		//the block of code following the if statement handles all data blocks
		//that to not include elements on the diagonal
		if(i!=j){
		    for(k=datablock[i][j]->getcolumns()-((j>=bcolumns-1)?(2):(1));k>=0;k--){
			for(l=datablock[i][j]->getrows()-1;l>=0;l--){
                	    solution[row+l]-=datablock[i][j]->get(l,k)*solution[col+k];
			datablock[i][j]->set(l,k,3333);
            	}   }   }
		//this block of code following the else statement handles all data blocks
		//that do include elements on the diagonal
		else{
		    for(k=datablock[i][i]->getcolumns()-((i==bcolumns-1)?(2):(1));k>=0;k--){
			solution[row+k]/=datablock[i][i]->get(k,k);
			datablock[i][i]->set(k,k,9999);
//			std::cout<<solution[row+k]<<std::endl;
			for(l=k-1;l>=0;l--){
                	    solution[row+l]-=datablock[i][i]->get(l,k)*solution[col+k];
			datablock[i][i]->set(l,k,3333);
        }   }	}   }	}

	return 1;
    }

    //finally, this function checks the accuracy of the LU computation a few rows at
    //a time
    double HPLMatreX3::checksolve(unsigned int row, unsigned int offset, bool complete){
	double toterror = 0;	//total error from all checks

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
                    futures.push_back(check_future(_gid,row+offset,offset*.5,false));
                }
                offset*=.5;
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
