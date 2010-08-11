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
#include "LUblock.hpp"

namespace hpx { namespace components { namespace server
{
    class HPX_COMPONENT_EXPORT HPLMatreX2 : public simple_component_base<HPLMatreX2>
    {
    public:
	//enumerate all of the actions that will(or can) be employed
        enum actions{
                hpl_construct=0,
                hpl_destruct=1,
                hpl_assign=2,
                hpl_get=3,
                hpl_set=4,
                hpl_solve=5,
		hpl_swap=6,
		hpl_gtop=7,
		hpl_gleft=8,
		hpl_gtrail=9,
		hpl_bsubst=10,
		hpl_check=11
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
	int assign(unsigned int row, unsigned int offset, bool complete);
	void pivot();
	int swap(unsigned int brow, unsigned int bcol);
	void LUdivide();
	void LUgausscorner(unsigned int iter);
	int LUgausstop(unsigned int iter, unsigned int bcol);
	int LUgaussleft(unsigned int brow, unsigned int iter);
	int LUgausstrail(unsigned int brow, unsigned int bcol, unsigned int iter);
	int LUbacksubst();
	double checksolve(unsigned int row, unsigned int offset, bool complete);
	void print();
	void print2();

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
					//(maps original rows to pivoted rows)

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
	typedef actions::result_action3<HPLMatreX2, int, hpl_assign, unsigned int,
		unsigned int, bool, &HPLMatreX2::assign> assign_action;
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
	//the top gaussian function
	typedef actions::result_action2<HPLMatreX2, int, hpl_gtop, unsigned int,
		unsigned int, &HPLMatreX2::LUgausstop> gtop_action;
	//the left side gaussian function
	typedef actions::result_action2<HPLMatreX2, int, hpl_gleft, unsigned int,
		unsigned int, &HPLMatreX2::LUgaussleft> gleft_action;
	//the trailing submatrix gaussian function
	typedef actions::result_action3<HPLMatreX2, int, hpl_gtrail, unsigned int,
		unsigned int, unsigned int, &HPLMatreX2::LUgausstrail> gtrail_action;
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

	//the backsubst future is used to make sure all computations are complete before
	//returning from LUsolve, to avoid killing processes and erasing the leftdata while
	//it is still being worked on
	typedef lcos::eager_future<server::HPLMatreX2::bsubst_action> bsubst_future;

	//the final future type for the class is used for checking the accuracy of
	//the results of the LU decomposition
	typedef lcos::eager_future<server::HPLMatreX2::check_action> check_future;
    };
//////////////////////////////////////////////////////////////////////////////////////

    //the constructor initializes the matrix
    int HPLMatreX2::construct(naming::id_type gid, unsigned int h, unsigned int w,
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

	allocate();

	//here we initialize the the matrix
//	lcos::eager_future<server::HPLMatreX2::assign_action>
//		assign_future(gid,(unsigned int)0,offset,false);
	assign(1,1,true);
	//initialize the pivot array
	for(i=0;i<rows;i++){pivotarr[i]=i;}
	//make sure that everything has been allocated their memory
//	assign_future.get();

	return 1;
    }

    //allocate() allocates memory space for the matrix
    void HPLMatreX2::allocate(){
	for(unsigned int i = 0;i < rows;i++){
	    truedata[i] = (double*) std::malloc(columns*sizeof(double));
	    factordata[i] = (double*) std::malloc(2*blocksize*sizeof(double));
	}
	for(unsigned int i = 0;i < brows;i++){
	    datablock[i] = (LUblock**)std::malloc(bcolumns*sizeof(LUblock*));
	}
    }

    //assign gives values to the empty elements of the array
    int HPLMatreX2::assign(unsigned int row, unsigned int offset, bool complete){
	for(unsigned int i=0;i<rows;i++){
	    for(unsigned int j=0;j<columns;j++){
		truedata[i][j] = (double) (rand() % 1000);
	    }
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
    double HPLMatreX2::get(unsigned int row, unsigned int col){
	return truedata[row][col];
    }

    //set() assigns a value to an element in all matrices
    void HPLMatreX2::set(unsigned int row, unsigned int col, double val){
	truedata[row][col] = val;
    }

    //print out the matrix
    void HPLMatreX2::print(){
	for(int i = 0;i < rows; i++){
	    for(int j = 0;j < columns; j++){
		std::cout<<datablock[i/blocksize][j/blocksize]->get(i%blocksize,j%blocksize)<<" ";
	    }
	    std::cout<<std::endl;
	}
	    std::cout<<std::endl;
    }
    void HPLMatreX2::print2(){
	for(int i = 0;i < rows; i++){
	    int temp=0;
	    while(temp + blocksize <= i){temp+=blocksize;}
	    for(int j = temp;j < columns; j++){
		std::cout<<datablock[i/blocksize][j/blocksize]->get(i%blocksize,j%blocksize)<<" ";
	    }
	    std::cout<<std::endl;
	}
	    std::cout<<std::endl;
    }
//END DEBUGGING FUNCTIONS/////////////////////////////////////////////

    //LUsolve is simply a wrapper function for LUfactor and LUbacksubst
    double HPLMatreX2::LUsolve(){
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
    //the swapping can occur in parallel
    void HPLMatreX2::pivot(){
	unsigned int max, max_row;
	unsigned int temp_piv;
	double temp;
	unsigned int h = std::ceil(((float)rows)*.5);

	//This first section of the pivot() function finds all of the pivot
	//values to compute the final pivot array
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

	for(int i=0;i<brows;i++){
		for(int j=0;j<bcolumns;j++){
			swap(i,j);
		}
	}
    }

    //swap() creates the datablocks and reorders the original
    //truedata matrix when assigning the initial values to the datablocks
    //according to the pivotarr data.  truedata itself remains unchanged
    int HPLMatreX2::swap(unsigned int brow, unsigned int bcol){
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
	    }
	}
    }

    //LUdivide is a wrapper function for the gaussian elimination functions,
    //although the process of properly dividing up the computations across
    //the different blocks using slightly different functions does require
    //the delegation provided by this wrapper
    void HPLMatreX2::LUdivide(){
	typedef lcos::eager_future<server::HPLMatreX2::gtop_action> gtop_future;
	typedef lcos::eager_future<server::HPLMatreX2::gleft_action> gleft_future;
	typedef lcos::eager_future<server::HPLMatreX2::gtrail_action> gtrail_future;

	std::vector<gtop_future> top_futures;
	std::vector<gleft_future> left_futures;
	std::vector<gtrail_future> trail_futures;
	unsigned int iteration = 0;
	unsigned int brow, bcol;

	do{
		LUgausscorner(iteration);
		brow = bcol = iteration+1;

		//if toprow + offset >= rows, then all computations are done
		if(iteration + 1 < bcolumns){
			//here we simultaneously perform gaussian elimination on
			//the top row of blocks and the left row of blocks
			while(bcol < bcolumns){
				top_futures.push_back(gtop_future(_gid,iteration,bcol));
				bcol++;
			}
			while(brow < brows){
				left_futures.push_back(gleft_future(_gid,brow,iteration));
				brow++;
			}

			//now do as much(very little this time) work as possible while waiting
			//for the above threads to finish
			brow = bcol = iteration+1;
			trail_futures.clear();
			BOOST_FOREACH(gtop_future gf, top_futures){
				gf.get();
			}
			BOOST_FOREACH(gleft_future gf, left_futures){
				gf.get();
			}
			//here begins the gaussian elimination of the trailing
			//submatrix.
			for(brow;brow < brows;brow++){
			    for(unsigned int j = bcol;j < bcolumns;j++){
				trail_futures.push_back(gtrail_future(_gid,
					brow,j,iteration));
			    }
			}
		}
		//peform memory management while the threads(if any exist)are
		//performing computations
		top_futures.clear();
		left_futures.clear();
		for(unsigned int i = iteration+1;i<brows;i++){
			delete datablock[i][iteration];
		}
		for(unsigned int i = iteration*blocksize;
		    i<std::min((int)((iteration+1)*blocksize),rows);i++){
			free(factordata[i]);
		}

		iteration++;
		//make sure all computations are done before moving on to the
		//next iteration (or exiting the function)
		BOOST_FOREACH(gtrail_future gf, trail_futures){
			gf.get();
		}
	}while(iteration < brows);
    }

    //LUgausscorner peforms gaussian elimination on the topleft corner block
    //of data that has not yet completed all of it's gaussian elimination
    //computations. Once complete, this block will need no further computations
    void HPLMatreX2::LUgausscorner(unsigned int iter){
	unsigned int fac_i = 0;
	unsigned int i, j, k;
	unsigned int offset = iter*blocksize;
	double f_factor;

	for(i=0;i<datablock[iter][iter]->getrows();i++){
		if(datablock[iter][iter]->get(i,i) == 0){
		    std::cerr<<"Warning: divided by zero\n";
		}
		f_factor = 1/datablock[iter][iter]->get(i,i);
		for(j=i+1;j<datablock[iter][iter]->getrows();j++){
		    factordata[j+offset][fac_i] = f_factor*datablock[iter][iter]->get(j,i);
		    for(k=i+1;k<datablock[iter][iter]->getcolumns();k++){
			datablock[iter][iter]->set(j,k,datablock[iter][iter]->get(j,k) -
			    factordata[j+offset][fac_i]*datablock[iter][iter]->get(i,k));
		    }
		}
		fac_i++;
	}
    }

    //LUgausstop performs gaussian elimination on the topmost row of blocks
    //that have not yet finished all gaussian elimination computation.
    //Once complete, these blocks will no longer need further computations
    int HPLMatreX2::LUgausstop(unsigned int iter, unsigned int bcol){
	unsigned int i,j,k;
	unsigned int fac_i = 0;
	unsigned int offset = iter*blocksize;

	for(i=0;i<datablock[iter][bcol]->getrows();i++){
		for(j=i+1;j<datablock[iter][bcol]->getrows();j++){
		    for(k=0;k<datablock[iter][bcol]->getcolumns();k++){
			datablock[iter][bcol]->set(j,k,datablock[iter][bcol]->get(j,k) -
			    factordata[j+offset][fac_i]*datablock[iter][bcol]->get(i,k));
		    }
		}
		fac_i++;
	}
	return 1;
    }

    //LUgaussleft performs gaussian elimination on the leftmost column of blocks
    //that have not yet finished all gaussian elimination computation.
    //Upon completion, no further computations need be done on these blocks.
    int HPLMatreX2::LUgaussleft(unsigned int brow, unsigned int iter){
	unsigned int i,j,k;
	unsigned int fac_i = 0;
	unsigned int offset = brow*blocksize;
	double f_factor;

	for(i=0;i<datablock[brow][iter]->getcolumns();i++){
		f_factor = 1/datablock[iter][iter]->get(i,i);
		for(j=0;j<datablock[brow][iter]->getrows();j++){
		    factordata[j+offset][fac_i] = f_factor*datablock[brow][iter]->get(j,i);
		    for(k=i+1;k<datablock[brow][iter]->getcolumns();k++){
			datablock[brow][iter]->set(j,k,datablock[brow][iter]->get(j,k) -
			    factordata[j+offset][fac_i]*datablock[iter][iter]->get(i,k));
		    }
		}
		fac_i++;
	}
	return 1;
    }

    //LUgausstrail performs gaussian elimination on the trailing submatrix of
    //the blocks operated on during the current iteration of the Gaussian elimination
    //computations. These blocks will still require further computations to be
    //performed in future iterations.
    int HPLMatreX2::LUgausstrail(unsigned int brow, unsigned int bcol, unsigned int iter){
	unsigned int i,j,k;
	unsigned int fac_i = 0;
	unsigned int offset = brow*blocksize;

	//outermost loop: iterates over the f_factors of the most recent corner block
	//	(f_factors are used indirectly through factordata)
	//middle loop: iterates over the rows of the current block
	//inner loop: iterates across the columns of the current block
	for(i=0;i<datablock[iter][iter]->getrows();i++){
		for(j=0;j<datablock[brow][bcol]->getrows();j++){
		    for(k=0;k<datablock[brow][bcol]->getcolumns();k++){
			datablock[brow][bcol]->set(j,k,datablock[brow][bcol]->get(j,k) -
			    factordata[j+offset][fac_i]*datablock[iter][bcol]->get(i,k));
		    }
		}
		fac_i++;
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
    double HPLMatreX2::checksolve(unsigned int row, unsigned int offset, bool complete){
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
