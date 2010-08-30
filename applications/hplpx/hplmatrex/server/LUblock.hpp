#ifndef _LUBLOCK_SERVER_HPP
#define _LUBLOCK_SERVER_HPP

/*This is the LUblock class implementation header file.
This is mostly just for storing data, but I hope to move
the Gaussian elimination functions here eventually.
*/

class LUblock
{
    public:
	//constructors and destructor
	LUblock(){}
	LUblock(unsigned int h, unsigned int w);
	~LUblock();

	//functions for assignment and data access
	bool getbusy(){return busy;}
	int getrows(){return rows;}
	int getcolumns(){return columns;}
	int getiteration(){return iteration;}
	double get(unsigned int row, unsigned int col);
	void set(unsigned int row, unsigned int col, double val);
	void setbusy(){busy = true;}
	void update(){
		iteration+=1;
		busy = false;
	}

    private://data members
	bool busy;
	int rows;
	int columns;
	int iteration;
	double** data;
};
//////////////////////////////////////////////////////////////////////////////////////

//the constructor initializes the matrix
LUblock::LUblock(unsigned int h, unsigned int w){
	busy = false;
	rows = h;
	columns = w;
	iteration = 0;

	data = (double**) std::malloc(h*sizeof(double*));
	for(int i=0;i<h;i++){data[i]=(double*) std::malloc(w*sizeof(double));}
}

//the destructor frees the memory
LUblock::~LUblock(){
	unsigned int i;
	for(i=0;i<rows;i++){
		free(data[i]);
	}
	free(data);
}

//get() gives back an element in the original matrix
double LUblock::get(unsigned int row, unsigned int col){
	return data[row][col];
}

//set() assigns a value to an element in all matrices
void LUblock::set(unsigned int row, unsigned int col, double val){
	data[row][col] = val;
}

#endif
