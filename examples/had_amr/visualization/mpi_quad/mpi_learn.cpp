// 27 Sep 2010
// Matt Anderson
// FMR nlsm code for strong scaling comparison with HPX

#include <iostream>
#include <vector>
#include <math.h>
#include <sdf.h>
#include <mpi.h>
#include "parse.h"
#include "mpreal.h"

typedef mpfr::mpreal had_double_type;
//typedef double had_double_type;

using namespace std;

int main(int argc,char* argv[]) {

  mpfr::mpreal::set_default_prec(256);

  int myid,numprocs;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid); 

  int tag = 99;
  if ( numprocs > 1 ) {
    if (myid == 0) {
      had_double_type n;
      char c_n[] = "0.000000001875000026575627008461288879635329944698826788978426034113560035351407471672013";

      std::cout << " TEST " << c_n << std::endl;
      n = c_n;

      std::string s = n.to_string();
      char const *word = s.c_str();
      int length = s.size();
  
      MPI_Send(&length,1,MPI_INT,1,tag+1,MPI_COMM_WORLD);  // send the length of the string
      MPI_Send((void*) word,length+1,MPI_CHAR,1,tag+2,MPI_COMM_WORLD);  // send the length of the string
    } else {
      MPI_Status status;
      had_double_type n;
      int length;
      char* word;
      MPI_Recv(&length,1,MPI_INT,0,tag+1,MPI_COMM_WORLD,&status); // get the length of the string
      word = (char *)malloc(sizeof(char)*length+1) ;
      MPI_Recv(word,length+1,MPI_CHAR,0,tag+2,MPI_COMM_WORLD,&status); // get the string
      std::cout << " TEST          " << word << std::endl;

      n = word;
    }
  }

#if 0
  int tag = 99;
  if ( numprocs > 1 ) {
    if (myid == 0) {
      //mpf_t n;
      had_double_type n;
      char c_n[] = "0.000000001875000026575627008461288879635329944698826788978426034113560035351407471672013";
      mpfr_init_set_str(n, c_n, 10);


      mp_exp_t n_exp;
      char *n_mant10 = mpf_get_str(NULL, &n_exp, 10, 0, n);
      int length = strlen(n_mant10);

      std::cout << " mant10 : " << n_mant10 << std::endl;
      std::cout << " n_exp : " << n_exp << std::endl;
      //printf("\nn_mant10 = %s\n n_exp10  = %d\n\n  length %d", n_mant10, n_exp,length);
    
      MPI_Send(&n_exp,1,MPI_LONG,1,tag,MPI_COMM_WORLD);  // send the exponent
     // MPI_Send(&length,1,MPI_INT,1,tag+1,MPI_COMM_WORLD);  // send the length of the string
     // MPI_Send(&n_mant10,length+1,MPI_CHAR,1,tag+2,MPI_COMM_WORLD); // send the string
    } else if ( myid == 1 ) {
      MPI_Status status;
      long int lExponent;
      int length;
      char *word;
      MPI_Recv(&lExponent,1,MPI_LONG,0,tag,MPI_COMM_WORLD,&status); // get the exponent
      std::cout << " exponent " << lExponent << std::endl;
    //  MPI_Recv(&length,1,MPI_INT,0,tag+1,MPI_COMM_WORLD,&status); // get the length of the string
    //  std::cout << " length " << length << std::endl;
    //  MPI_Recv(word,length+1,MPI_CHAR,0,tag+2,MPI_COMM_WORLD,&status); // get the string
    }
  }
#endif
  MPI_Finalize();
  return 0;
}
