////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _LUBLOCK_SERVER_HPP
#define _LUBLOCK_SERVER_HPP

/*This is the LUblock class implementation header file.
This is to store data semi-contiguously.
*/

class LUblock
{
    public:
    //constructors and destructor
    LUblock(){}
    LUblock(unsigned int h, unsigned int w);
    ~LUblock();

    //data members
    int rows;
    int columns;
    double* workSpace;
    double** data;
};
//////////////////////////////////////////////////////////////////////////////////////

//the constructor initializes the matrix
LUblock::LUblock(unsigned int h, unsigned int w){
    workSpace = (double*) std::malloc((8+h*w)*sizeof(double));
    data = (double**) std::malloc(h*sizeof(double*));
    rows = h;
    columns = w;
    for(int i = 0;i < (int)h; i++){
        data[i] = workSpace + 8 + i*w;
    }
}

//the destructor frees the memory
LUblock::~LUblock(){
    free(workSpace);
    free(data);
}
#endif
