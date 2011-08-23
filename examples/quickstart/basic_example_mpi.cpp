// Copyright (c) 2011 Matt Anderson
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// MPI version of the basic_example.cpp example

#include <iostream>

#include "mpi.h"

int getnumber(int a,int b);

int main(int argc, char *argv[] )
{
  MPI::Init(argc,argv);
  int p = MPI::COMM_WORLD.Get_size ( );
  int id = MPI::COMM_WORLD.Get_rank ( );

  int tag1 = 0;
  int tag2 = 1;

  MPI::Status status;
  if ( p > 1 ) {
    if ( id == 1 ) {
      int r1 = getnumber(2,3); 
      int r2 = getnumber(2,3);
      MPI::COMM_WORLD.Send (&r1, 1, MPI::INT, 0, tag1 );    
      MPI::COMM_WORLD.Send (&r2, 1, MPI::INT, 0, tag2 );    
    } else {
      int g1,g2;
      MPI::COMM_WORLD.Recv ( &g1, 1,  MPI::INT, 1, tag1, status );
      MPI::COMM_WORLD.Recv ( &g2, 1,  MPI::INT, 1, tag2, status );
      std::cout << " Result: " << g1 + g2 << std::endl; 
    }
  } else {
    int r1 = getnumber(2,3); 
    int r2 = getnumber(2,3);
    std::cout << " Result: " << r1 + r2 << std::endl; 
  }

  MPI::Finalize();
}


int getnumber(int a,int b) {
  return a*b;
}
