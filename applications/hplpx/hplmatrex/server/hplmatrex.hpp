#ifndef _HPLMATREX_SERVER_HPP
#define _HPLMATREX_SERVER_HPP

/*This is the HPLMatrex class implementation header file.
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


namespace hpx { namespace components { namespace server
{
    class HPX_COMPONENT_EXPORT HPLMatrex : public simple_component_base<HPLMatrex>
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
	HPLMatrex(){}
	int construct(naming::id_type gid, unsigned int h, unsigned int w,
		unsigned int ab, unsigned int bs);
	~HPLMatrex(){destruct();}
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
	int swap(unsigned int row, unsigned int offset, bool complete);
	void LUdivide();
	void LUgausscorner(unsigned int start);
	int LUgausstop(unsigned int start_col, unsigned int start_row, bool till_end);
	int LUgaussleft(unsigned int start_row, unsigned int start_col, bool till_end);
	int LUgausstrail(unsigned int start_row, unsigned int start_col, bool to_rend,
		bool to_cend, unsigned int iter);
	int LUbacksubst();
	double checksolve(unsigned int row, unsigned int offset, bool complete);
	void print();

	int rows;
	int columns;
	int allocblock;			//reflects amount of allocation done per thread
	int blocksize;			//reflects amount of computation per thread
	naming::id_type _gid;		//the instances gid
	double** leftdata;		//for computations
	double** factordata;		//also for computations
	double** topdata;		//results of computations
	double** truedata;		//for accuracy checking
        double* solution;		//for storing the solution
	int* pivotarr;			//array for storing pivot elements
					//(maps original rows to pivoted rows)

    public:
	//here we define the actions that will be used
	//the construct function
	typedef actions::result_action5<HPLMatrex, int, hpl_construct, naming::id_type,
		unsigned int, unsigned int, unsigned int, unsigned int,
		&HPLMatrex::construct> construct_action;
	//the destruct function
	typedef actions::action0<HPLMatrex, hpl_destruct,
		&HPLMatrex::destruct> destruct_action;
	 //the assign function
	typedef actions::result_action3<HPLMatrex, int, hpl_assign, unsigned int,
		unsigned int, bool, &HPLMatrex::assign> assign_action;
	//the get function
	typedef actions::result_action2<HPLMatrex, double, hpl_get, unsigned int,
        	unsigned int, &HPLMatrex::get> get_action;
	//the set function
	typedef actions::action3<HPLMatrex, hpl_set, unsigned int,
        	unsigned int, double, &HPLMatrex::set> set_action;
	//the solve function
	typedef actions::result_action0<HPLMatrex, double, hpl_solve,
		&HPLMatrex::LUsolve> solve_action;
	 //the swap function
	typedef actions::result_action3<HPLMatrex, int, hpl_swap, unsigned int,
		unsigned int, bool, &HPLMatrex::swap> swap_action;
	//the top gaussian function
	typedef actions::result_action3<HPLMatrex, int, hpl_gtop, unsigned int,
		unsigned int, bool, &HPLMatrex::LUgausstop> gtop_action;
	//the left side gaussian function
	typedef actions::result_action3<HPLMatrex, int, hpl_gleft, unsigned int,
		unsigned int, bool, &HPLMatrex::LUgaussleft> gleft_action;
	//the trailing submatrix gaussian function
	typedef actions::result_action5<HPLMatrex, int, hpl_gtrail, unsigned int,
		unsigned int, bool, bool, unsigned int,
		&HPLMatrex::LUgausstrail> gtrail_action;
	//backsubstitution function
	typedef actions::result_action0<HPLMatrex, int, hpl_bsubst,
		&HPLMatrex::LUbacksubst> bsubst_action;
	//checksolve function
	typedef actions::result_action3<HPLMatrex, double, hpl_check, unsigned int,
		unsigned int, bool, &HPLMatrex::checksolve> check_action;

	//here begins the definitions of most of the future types that will be used
	//the first of which is for assign action
	typedef lcos::eager_future<server::HPLMatrex::assign_action> assign_future;

	//Here is the swap future, which works the same way as the assign future
	typedef lcos::eager_future<server::HPLMatrex::swap_action> swap_future;

	//the backsubst future is used to make sure all computations are complete before
	//returning from LUsolve, to avoid killing processes and erasing the leftdata while
	//it is still being worked on
	typedef lcos::eager_future<server::HPLMatrex::bsubst_action> bsubst_future;

	//the final future type for the class is used for checking the accuracy of
	//the results of the LU decomposition
	typedef lcos::eager_future<server::HPLMatrex::check_action> check_future;
    };
//////////////////////////////////////////////////////////////////////////////////////

    //the constructor initializes the matrix
    int HPLMatrex::construct(naming::id_type gid, unsigned int h, unsigned int w,
		unsigned int ab, unsigned int bs){
// / / /initialize class variables/ / / / / / / / / / / /
	rows=h;
	columns=w;
	if(ab > std::ceil(((float)h)*.5)){
		allocblock=std::ceil(((float)h)*.5);}
	else{allocblock=ab;}
	if(bs > h){
		blocksize=h;}
	else{blocksize=bs;}
	_gid = gid;
// / / / / / / / / / / / / / / / / / / / / / / / / / / /

	int i; 			 //just counting variable
	unsigned int offset = 1; //the initial offset used for the memory handling algorithm

	leftdata = (double**) std::malloc(h*sizeof(double*));
	factordata = (double**) std::malloc(h*sizeof(double*));
	topdata = (double**) std::malloc(h*sizeof(double*));
	truedata = (double**) std::malloc(h*sizeof(double*));
	pivotarr = (int*) std::malloc(h*sizeof(int));
	srand(time(NULL));

	//by making offset a power of two, the allocate and assign functions
	//are much simpler than they would be otherwise
	h=std::ceil(((float)h)*.5);
	while(offset < h){offset *= 2;}

	allocate();

	//here we initialize the the matrix
//	lcos::eager_future<server::HPLMatrex::assign_action>
//		assign_future(gid,(unsigned int)0,offset,false);
	assign(1,1,true);
	//initialize the pivot array
	for(i=0;i<rows;i++){pivotarr[i]=i;}
	//make sure that everything has been allocated their memory
//	assign_future.get();


	return 1;
    }

    //allocate() allocates a few rows of memory space for the matrix
    void HPLMatrex::allocate(){
	for(unsigned int i = 0;i < rows;i++){
	    leftdata[i] = (double*) std::malloc((i)*sizeof(double));
	    truedata[i] = (double*) std::malloc(columns*sizeof(double));
	    topdata[i] = (double*) std::malloc((columns-(i))*sizeof(double));
	    factordata[i] = (double*) std::malloc(2*blocksize*sizeof(double));
	}
    }

    //assign gives values to the empty elements of the array
    int HPLMatrex::assign(unsigned int row, unsigned int offset, bool complete){
	for(unsigned int i=0;i<rows;i++){
	    for(unsigned int j=0;j<columns;j++){
		truedata[i][j] = (double) (rand() % 1000);
	    }
	}
	return 1;
    }

    //the destructor frees the memory
    void HPLMatrex::destruct(){
	unsigned int i;
	for(i=0;i<rows;i++){
		free(topdata[i]);
		free(truedata[i]);
	}
	free(topdata);
	free(leftdata);
	free(factordata);
	free(truedata);
    }

//DEBUGGING FUNCTIONS/////////////////////////////////////////////////
    //get() gives back an element in the original matrix
    double HPLMatrex::get(unsigned int row, unsigned int col){
	return truedata[row][col];
    }

    //set() assigns a value to an element in all matrices
    void HPLMatrex::set(unsigned int row, unsigned int col, double val){
	truedata[row][col] = val;
	if(col < row){leftdata[row][col] = val;}
	else{topdata[row][col-row] = val;}
    }
    //print out the matrix
    void HPLMatrex::print(){
	for(int i = 0;i < rows; i++){
		for(int j = 0;j < i; j++){
			std::cout<<leftdata[i][j]<<" ";
		}
		for(int k = 0;k+i<columns;k++){
			std::cout<<topdata[i][k]<<" ";
		}
		std::cout<<std::endl;
	}
		std::cout<<std::endl;
    }
//END DEBUGGING FUNCTIONS/////////////////////////////////////////////

    //LUsolve is simply a wrapper function for LUfactor and LUbacksubst
    double HPLMatrex::LUsolve(){
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
    void HPLMatrex::pivot(){
	unsigned int max, max_row;
	unsigned int temp_piv;
	double temp;
	unsigned int h = std::ceil(((float)rows)*.5);
	unsigned int offset = 1;
	while(offset < h){offset *= 2;}

	//This first section of the pivot() function finds all of the pivot
	//values to compute the final pivot array
	for(unsigned int i=0;i<rows-1;i++){
	    max_row = i;
	    max = truedata[pivotarr[i]][i];
	    temp_piv = pivotarr[i];
	    for(unsigned int j=i+1;j<rows;j++){
		temp = std::fabs(truedata[pivotarr[j]][i]);
		if(temp > max){
		    max = temp;
		    max_row = j;
		}
	    }
	    pivotarr[i] = pivotarr[max_row];
	    pivotarr[max_row] = temp_piv;
	}

	//Here we finally initialize the leftdata and topdata arrays by
	//calling swap()
	lcos::eager_future<server::HPLMatrex::swap_action>
		swap_future(_gid,(unsigned int)0,offset,false);
	swap_future.get();
    }

    //swap() reorders the original truedata matrix when assigning the initial
    //values to the leftdata and topdata arrays according to the pivotarr data
    int HPLMatrex::swap(unsigned int row, unsigned int offset, bool complete){
	//futures is used to allow this thread to continue spinning off new threads
	//while other threads work, and is checked at the end to make certain all
	//threads are completed before returning.
	std::vector<swap_future> futures;

	//start spinning off work to do
	while(!complete){
	    if(offset <= allocblock){
		if(row + offset < rows){
		    futures.push_back(swap_future(_gid,row+offset,offset,true));
		}
		complete = true;
	    }
	    else{
		if(row + offset < rows){
		    futures.push_back(swap_future(_gid,row+offset,offset*.5,false));
		}
		offset*=.5;
	    }
	}

	unsigned int temp = std::min((int)offset, (int)(rows - row));
	for(unsigned int i=0;i<temp;i++){
	    for(unsigned int j=0;j<columns;j++){
		if(row+i>j){leftdata[row+i][j] = truedata[pivotarr[row+i]][j];}
		else{topdata[row+i][j-row-i] = truedata[pivotarr[row+i]][j];}
	    }
	}

	//ensure that every element has been initialized to avoid working on them
	//before they have a value
	BOOST_FOREACH(swap_future sf, futures){
		sf.get();
	}
	return 1;
    }

    //LUdivide is a wrapper function for the gaussian elimination functions,
    //although the process of properly dividing up the computations across
    //the different blocks using slightly different functions does require
    //the delegation provided by this wrapper
    void HPLMatrex::LUdivide(){
	typedef lcos::eager_future<server::HPLMatrex::gtop_action> gtop_future;
	typedef lcos::eager_future<server::HPLMatrex::gleft_action> gleft_future;
	typedef lcos::eager_future<server::HPLMatrex::gtrail_action> gtrail_future;

	std::vector<gtop_future> top_futures;
	std::vector<gleft_future> left_futures;
	std::vector<gtrail_future> trail_futures;
	unsigned int iteration = 0, toprow = 0;
	unsigned int farrow, farcol;
	unsigned int offset = 2*blocksize;

	do{
		//first we perform gaussian elimination on the top left
		//corner block
		LUgausscorner(toprow);
		farrow = toprow + blocksize;

		//if toprow + offset >= rows, then all computations are done
		if(toprow + offset < rows){
			//here we simultaneously perform gaussian elimination on
			//the top row of blocks and the left row of blocks, except
			//for the last block of the row/column
			while(farrow + offset < rows){
				top_futures.push_back(gtop_future(_gid,farrow, toprow, false));
				left_futures.push_back(gleft_future(_gid,farrow, toprow, false));
				farrow += blocksize;
			}

			//this is where we perform gaussian elimination on
			//the last block in the top/left row/column
			top_futures.push_back(gtop_future(_gid,farrow, toprow, true));
			left_futures.push_back(gleft_future(_gid,farrow, toprow, true));

			//now do as much(very little this time) work as possible while waiting
			//for the above threads to finish
			farrow = toprow + blocksize;
			farcol = farrow;
			trail_futures.clear();

			BOOST_FOREACH(gtop_future gf, top_futures){
				gf.get();
			}
			BOOST_FOREACH(gleft_future gf, left_futures){
				gf.get();
			}

			//here begins the gaussian elimination of the trailing
			//submatrix.
			for(farrow;farrow+offset<rows;farrow += blocksize){
			    //the inner for loop takes care of computations in
			    //all blocks except for those on the bottom row
			    //of blocks or rightmost column of blocks
			    for(farcol;farcol+offset<rows;farcol+=blocksize){
				trail_futures.push_back(gtrail_future(_gid,
					farrow,farcol,false,false,iteration));
			    }
			    //this call is what handles the far right column
			    //of data blocks' computations, not including
			    //the bottom right corner block
			    trail_futures.push_back(gtrail_future(_gid,
			    	farrow,farcol,false,true,iteration));
			    farcol = toprow+blocksize;
			}
			//this call handles the bottom row of data blocks,
			//not including the bottom right corner block
			for(farcol;farcol+offset<rows;farcol+=blocksize){
			    trail_futures.push_back(gtrail_future(_gid,
				farrow,farcol,true,false,iteration));
			}
			//this final call handles the bottom right block
			trail_futures.push_back(gtrail_future(_gid,
			    farrow,farcol,true,true,iteration));
		}

		//peform memory management while the threads(if any exist)are
		//performing computations
		top_futures.clear();
		left_futures.clear();

		if(toprow+offset<rows){farrow = toprow + blocksize;}
		else{farrow = rows;}
		for(unsigned int i = toprow;i<farrow;i++){
			free(leftdata[i]);
			free(factordata[i]);
		}
		toprow += blocksize;
		iteration++;
		//make sure all computations are done before moving on to the
		//next iteration (or exiting the function)
		BOOST_FOREACH(gtrail_future gf, trail_futures){
			gf.get();
		}
	}while(toprow+blocksize < rows);
    }

    //LUgausscorner peforms gaussian elimination on the topleft corner block
    //of data that has not yet completed all of it's gaussian elimination
    //computations. Once complete, this block will need no further computations
    void HPLMatrex::LUgausscorner(unsigned int start){
	unsigned int endr = start + blocksize;
	unsigned int endc, fac_i = 0;
	unsigned int i, j, k;
	double f_factor;

	if(endr + blocksize >= rows){
		endr = rows;
		endc = rows + 1;
	}
	else{endc = endr;}

	for(i=start;i<endr;i++){
		if(topdata[i][0] == 0){
		    std::cerr<<"Warning: divided by zero\n";
		}
		f_factor = 1/topdata[i][0];
		for(j=i+1;j<endr;j++){
		    factordata[j][fac_i] = f_factor*leftdata[j][i];
		    for(k=i+1;k<j;k++){
			leftdata[j][k] -= factordata[j][fac_i]*topdata[i][k-i];
		    }
		    for(k=0;k+j<endc;k++){
			topdata[j][k] -= factordata[j][fac_i]*topdata[i][k+j-i];
		    }
		}
		fac_i++;
	}
    }

    //LUgausstop performs gaussian elimination on the topmost row of blocks
    //that have not yet finished all gaussian elimination computation.
    //Once complete, these blocks will no longer need further computations
    int HPLMatrex::LUgausstop(unsigned int start_col, unsigned int start_row, bool till_end){
	unsigned int i,j,k;
	unsigned int endc, endr = start_row+blocksize, fac_i = 0;
	if(!till_end){endc = start_col + blocksize;}
	else{endc = columns;}

	for(i=start_row;i<endr;i++){
		for(j=i+1;j<endr;j++){
		    for(k=start_col;k<endc;k++){
			topdata[j][k-j] -= factordata[j][fac_i]*topdata[i][k-i];
		    }
		}
		fac_i++;
	}
	return 1;
    }

    //LUgaussleft performs gaussian elimination on the leftmost column of blocks
    //that have not yet finished all gaussian elimination computation.
    //Upon completion, no further computations need be done on these blocks.
    int HPLMatrex::LUgaussleft(unsigned int start_row, unsigned int start_col, bool till_end){
	unsigned int i,j,k;
	unsigned int endr, endc = start_col+blocksize, fac_i = 0;
	double f_factor;

	if(!till_end){endr = start_row + blocksize;}
	else{endr = rows;}
	for(i=start_col;i<endc;i++){
		f_factor = 1/topdata[i][0];
		for(j=start_row;j<endr;j++){
		    factordata[j][fac_i] = f_factor*leftdata[j][i];
		    for(k=i+1;k<endc;k++){
			leftdata[j][k] -= factordata[j][fac_i]*topdata[i][k-i];
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
    int HPLMatrex::LUgausstrail(unsigned int start_row, unsigned int start_col, bool till_rend,
	bool till_cend, unsigned int iter){
	unsigned int i,j,k;
	unsigned int endr, endc, fac_i = 0;
	unsigned int begin_i = iter*blocksize;

	if(!till_rend){endr = start_row + blocksize;}
	else{endr = rows;}
	if(!till_cend){endc = start_col + blocksize;}
	else{endc = columns;}

	for(i=begin_i;i<begin_i+blocksize;i++){
		for(j=start_row;j<endr;j++){
		    for(k=start_col;k<std::min(j,endc);k++){
			leftdata[j][k] -= factordata[j][fac_i]*topdata[i][k-i];
		    }
		    for(k;k<endc;k++){
			topdata[j][k-j] -= factordata[j][fac_i]*topdata[i][k-i];
		    }
		}
		fac_i++;
	}
	return 1;
    }

    //this is the old method of backsubst and is completely linear for now
    int HPLMatrex::LUbacksubst(){
        int i,j;

        for(i=rows-1;i>=0;i--){
    		solution[i] = topdata[i][rows - i];
                for(j=i+1;j<rows;j++){
                        solution[i]-=topdata[i][j-i]*solution[j];
                }
                solution[i] /= topdata[i][0];
//		std::cout<<solution[i]<<std::endl;
	}
	return 1;
    }

    //finally, this function checks the accuracy of the LU computation a few rows at
    //a time
    double HPLMatrex::checksolve(unsigned int row, unsigned int offset, bool complete){
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
