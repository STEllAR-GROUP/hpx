#ifndef _HPLMATREX_SERVER_HPP
#define _HPLMATREX_SERVER_HPP

/*This is the HPLMatrix class implementation header file.
In order to keep things simple, only operations necessary
to to perform LUP decomposition are declared, which is
basically just constructors, assignment operators, 
a destructor, and access operators.
*/

#include <time.h>

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>
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
        enum actions{
                hpl_construct=0,
                hpl_destruct=1,
                hpl_allocate=2,
                hpl_assign=3,
                hpl_get=4,
                hpl_set=5,
                hpl_solve=6,
		hpl_gauss=7,
		hpl_gline=8,
		hpl_bsubst=9,
		hpl_check=10,
		hpl_mkvec=11
        };

	//constructors and destructor
	HPLMatrex(){}
	int construct(naming::id_type gid, unsigned int h, unsigned int w, unsigned int bs);
	~HPLMatrex(){destruct();}
	void destruct();

	//operators for assignment and data access
	double get(unsigned int row, unsigned int col);
	void set(unsigned int row, unsigned int col, double val);

	//functions for manipulating the matrix
	double LUsolve();

    private:
	int mkvectors();
	int allocate(unsigned int row, unsigned int offset, bool complete);
	int assign(unsigned int row, unsigned int offset, bool complete);
	void LUdivide();
	void LUgauss(unsigned int top);
	int LUgaussline(unsigned int row, unsigned int startcol, double factor);
	int LUbacksubst();
	double checksolve(unsigned int row, unsigned int offset, bool complete);

	int rows;
	int columns;
	int blocksize;
	naming::id_type _gid;
	double** data;		//for computations
	double** data2;		//for accuracy checking
        double* solution;

    public:
	//here we define the actions that will be used
	//the construct function
	typedef actions::result_action4<HPLMatrex, int, hpl_construct, naming::id_type,
		unsigned int, unsigned int, unsigned int, &HPLMatrex::construct> construct_action;
	//the destruct function
	typedef actions::action0<HPLMatrex, hpl_destruct,
		&HPLMatrex::destruct> destruct_action;
	//the mkvectors function
	typedef actions::result_action0<HPLMatrex, int, hpl_mkvec,
		&HPLMatrex::mkvectors> mkvec_action;
	//the allocate function
	typedef actions::result_action3<HPLMatrex, int, hpl_allocate, unsigned int,
        	unsigned int, bool, &HPLMatrex::allocate> allocate_action;
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
	//the first gaussian function
	typedef actions::action1<HPLMatrex, hpl_gauss,
		unsigned int, &HPLMatrex::LUgauss> gauss_action;
	//the second gaussian function
	typedef actions::result_action3<HPLMatrex, int, hpl_gline, unsigned int,
		unsigned int, double, &HPLMatrex::LUgaussline> gline_action;
	//backsubstitution function
	typedef actions::result_action0<HPLMatrex, int, hpl_bsubst,
		&HPLMatrex::LUbacksubst> bsubst_action;
	//checksolve function
	typedef actions::result_action3<HPLMatrex, double, hpl_check, unsigned int,
		unsigned int, bool, &HPLMatrex::checksolve> check_action;

////////////////////////////////////////////////////////////////////
	//here begins the definitions of most of the future types that will be used
	//the first of which are for the allocate and assign actions
	typedef lcos::eager_future<server::HPLMatrex::allocate_action> allocate_future;
	typedef lcos::eager_future<server::HPLMatrex::assign_action> assign_future;

	//below is a future type used for determining when computations of the
	//type1 gaussian eliminations have advanced enough to proceed with the next
	//round of gaussian elimination computations
	typedef lcos::eager_future<server::HPLMatrex::gline_action> gline_future;

	//the backsubst future is used to make sure all computations are complete before
	//returning from LUsolve, to avoid killing processes and erasing the data while
	//it is still being worked on
	typedef lcos::eager_future<server::HPLMatrex::bsubst_action> bsubst_future;

	//the final future type for the class is used for checking the accuracy of
	//the results of the LU decomposition
	typedef lcos::eager_future<server::HPLMatrex::check_action> check_future;

    private:
	//this is a vector of vectors of futures and is used for determining when type1
	//computations have advanced enough for further computations to take place
	std::vector<std::vector<gline_future> > gaussian_futures;
    };


    //the constructor initializes the matrix
    int HPLMatrex::construct(naming::id_type gid, unsigned int h, unsigned int w, unsigned int bs){
	rows=h;
	columns=w;
	if(bs > std::ceil(((float)h)*.5)){
		blocksize=std::ceil(((float)h)*.5);}
	else{blocksize=bs;}
	_gid = gid;

	//begin allocating memory for the gaussian_futures vector of vectors
	lcos::eager_future<server::HPLMatrex::mkvec_action> mkvec_future(gid);

	unsigned int i,j; //just counting variables for debugging
	unsigned int offset = 1; //the initial offset used for the memory handling algorithm

	data = (double**) std::malloc(h*sizeof(double*));
	data2 = (double**) std::malloc(h*sizeof(double*));
	solution = (double*) std::malloc(h*sizeof(double));
	srand(time(NULL));

	//by making offset a power of two, the allocate and assign functions
	//are much simpler than they would be otherwise
	h=std::ceil(((float)h)*.5);
	while(offset < h){offset *= 2;}

	//here we allocate space for the matrix in parallel
	lcos::eager_future<server::HPLMatrex::allocate_action>
		allocate_future(gid,(unsigned int)0,offset,false);
	allocate_future.get();
	//here we initialize the the matrix
	lcos::eager_future<server::HPLMatrex::assign_action>
		assign_future(gid,(unsigned int)0,offset,false);

	//make sure that the vectors have been allocated their memory
	assign_future.get();
	mkvec_future.get();
	return 1;
    }

    //mkvectors() reserves memory for the gaussian_futures vector of vectors
    int HPLMatrex::mkvectors(){
	double factor = 1/blocksize;
	for(unsigned int i = 0;i < rows;i++){
	    	//also add a vector to the vector of vectors
	    	std::vector<gline_future> tmpvector;
	    	tmpvector.reserve(std::ceil((rows-i)*factor));
	    	gaussian_futures.push_back(tmpvector);
	}
    }

    //allocate() allocates a few rows of memory space for the matrix
    int HPLMatrex::allocate(unsigned int row, unsigned int offset, bool complete){
	std::vector<allocate_future> futures;
	//start spinning off work to do
	while(!complete){
	    if(offset <= blocksize){
		if(row + offset < rows){
		    futures.push_back(allocate_future(_gid,row+offset,offset,true));
		}
		complete = true;
	    }
	    else{
		if(row + offset < rows){
		    futures.push_back(allocate_future(_gid,row+offset,offset*.5,false));
		}
		offset*=.5;
	    }
	}

	//allocate space for the indicated row(s)
	unsigned int temp = std::min((int)offset, (int)(rows - row));
	hpx::lcos::mutex mtx;
	for(unsigned int i = 0;i < temp;i++){
	    data[row+i] = (double*) std::malloc(columns*sizeof(double*));
	    data2[row+i] = (double*) std::malloc(columns*sizeof(double*));
	}
        //ensure that all memory has been allocated to avoid initializing
        //unallocated space
        BOOST_FOREACH(allocate_future af, futures){
                af.get();
        }
        return 1;
    }

    //assign gives values to the empty elements of the array
    int HPLMatrex::assign(unsigned int row, unsigned int offset, bool complete){
	//futures is used to allow this thread to continue spinning off new threads
	//while other threads work, and is checked at the end to make certain all
	//threads are completed before returning.
	std::vector<assign_future> futures;

	//start spinning off work to do
	while(!complete){
	    if(offset <= blocksize){
		if(row + offset < rows){
		    futures.push_back(assign_future(_gid,row+offset,offset,true));
		}
		complete = true;
	    }
	    else{
		if(row + offset < rows){
		    futures.push_back(assign_future(_gid,row+offset,offset*.5,false));
		}
		offset*=.5;
	    }
	}

	unsigned int temp = std::min((int)offset, (int)(rows - row));
	for(unsigned int i=0;i<temp;i++){
	    for(unsigned int j=0;j<columns;j++){
                //need to keep the data size from varying too much
                //in order to avoid ever rounding to 0 during
                //computations
        	data[row+i][j] = (double) (rand() % 1000);
		data2[row+i][j] = data[row+i][j];
	    }
	}

	//ensure that each element has been initialized to avoid working on them
	//before they have a value
	BOOST_FOREACH(assign_future af, futures){
		af.get();
	}
	return 1;
    }


    //the deconstructor frees the memory
    void HPLMatrex::destruct(){
	unsigned int i;
	for(i=0;i<rows;i++){
		free(data[i]);
	}
	free(data);
    }

    //get() gives back an element in the matreix
    double HPLMatrex::get(unsigned int row, unsigned int col){
	return data[row][col];
    }

    //set() assigns a value to an element in the matrix
    void HPLMatrex::set(unsigned int row, unsigned int col, double val){
	data[row][col] = val;
    }

    //LUsolve is simply a wrapper function for LUfactor and LUbacksubst
    //the default stepsize is 1
    double HPLMatrex::LUsolve(){
	LUdivide();
	bsubst_future bsub(_gid);
	bsub.get();

	unsigned int h = std::ceil(((float)rows)*.5);
	unsigned int offset = 1;
	while(offset < h){offset *= 2;}

	check_future chk(_gid,0,offset,false);
	return chk.get();
    }

    //LUdivide is a wrapper function for the type1 gaussian elimination
    void HPLMatrex::LUdivide(){
	unsigned int row;

	LUgauss(0);
	for(row = 1;row < rows-1; row++){
		//make sure the row we are using is up to date
		gaussian_futures[row-1][0].get();
		LUgauss(row);
	}
	gaussian_futures[rows-2][0].get();
    }

    //LUgauss is used to spin off gaussian elimination work one
    //row at a time.  It uses the gaussian_futures member in order
    //to keep track of work.
    void HPLMatrex::LUgauss(unsigned int top){
	unsigned int i,check=1;
	double f_factor;

        //first make sure there are no divisions by zero
        if(data[top][top] == 0){
                std::cout<<"Warning: divided by zero"<<std::endl;
        }

	//f_factor stores the division to avoid divisions
	//during the loop
	f_factor = 1/data[top][top];

	//here we have the actual gaussian elimination
	//first is the special case where we do not need to wait to begin
	//computations. For all other cases we must make sure a row has
	//finished being updated before we update it again
	if(top == 0){
	    for(i=1;i<rows;i+=blocksize){
		gaussian_futures[0].push_back(gline_future(_gid,i,top,f_factor));
	    }
	}
	else{
	    for(i=top+1;i<rows;i+=blocksize){
		//if i+blocksize-1 exceeds the bottom row, then
		//the previous future.get() already indicates that the
		//next rows are ready to be worked on. This can be checked
		//as it can be shown graphically that if top+check*blocksize==rows,
		//the rows are already ready to be worked on. From this,
		//the equality top+check*blocksize=i+blocksize-1 can be derived
		if(i+blocksize-1 < rows){
			gaussian_futures[top-1][check++].get();
		}
		gaussian_futures[top].push_back(gline_future(_gid,i,top,f_factor));
	    }
	}
    }

    //LUgaussline performs the Gaussian elimination to the specified row
    int HPLMatrex::LUgaussline(unsigned int row, unsigned int startcol, double f_factor){
	unsigned int i;
	unsigned int temp = std::min((int)rows, (int)(row + blocksize));
	double factor;

	for(row;row<temp;row++){
		factor = data[row][startcol]*f_factor;
		for(i=startcol+1;i<=rows;i++){
			data[row][i]-=factor*data[startcol][i];
		}
	}
	return 1;
    }

    //this is the old method of backsubst
    int HPLMatrex::LUbacksubst(){
        int i,j;

        for(i=rows-1;i>=0;i--){
    		solution[i] = data[i][rows];
                for(j=i+1;j<rows;j++){
                        solution[i]-=data[i][j]*solution[j];
                }
                solution[i] /= data[i][i];
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
            if(offset <= blocksize){
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
                sum += data2[row+i][j] * solution[j];
            }
	    toterror += std::fabs(sum-data2[row+i][rows]);
        }

        //ensure that each element has been initialized to avoid working on them
        //before they have a value
        BOOST_FOREACH(check_future cf, futures){
                toterror += cf.get();
        }
        return toterror;
    }
}}}

#endif
