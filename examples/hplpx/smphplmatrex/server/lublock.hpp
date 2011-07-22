////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _LUBLOCK_SERVER_HPP
#define _LUBLOCK_SERVER_HPP

/*This is the lublock class implementation header file.
This is to store data semi-contiguously.
*/

class lublock
{
    public:
    //constructors and destructor
    lublock(){}
    lublock(unsigned int h, unsigned int w);
    ~lublock();

    //data members
    int rows;
    int columns;
    double* workSpace;
    double** data;
};
//////////////////////////////////////////////////////////////////////////////////////

//the constructor initializes the matrix
lublock::lublock(unsigned int h, unsigned int w){
    workSpace = new double[8+h*w];
    data = new double*[h];
    rows = h;
    columns = w;
    for(int i = 0;i < (int)h; i++){
        data[i] = workSpace + 8 + i*w;
    }
}

//the destructor frees the memory
lublock::~lublock(){
    free(workSpace);
    free(data);
}
#endif
