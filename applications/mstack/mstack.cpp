/* 
Mstack is a benchmark that simulates one stage of post-processing 
(by means of median stacking) done on seismic reflection data. 
More information is available here: 
http://www.eecis.udel.edu/~mpellegr/mstack/

-Mark Pellegrini (markpell@udel.edu)
August, 2008 
*/

#include "mstack.hpp"

int mstack::initialize (int j, int numchn){
  int m, l, k;
  switch (j) {
  case 1:
  case 4:
    {
      for (m=1; m <= 1000; m++){
	for (l=1; l <= 1000; l++){
	  for (k=1; k <= numchn; k++){
	    mytraces[m][l][k] = 1.0;
	  };
	};
      };
      break;
    };
  case 2:
  case 5:
    {
      for (m=1; m <= 1000; m++){
	for (l=1; l <= 1000; l++){
	  for (k=1; k <= numchn; k++){
	    mytraces[m][l][k] = k;
	  };
	};
      }
      break;
    };
  case 3:
  case 6: 
    {
      for (m=1; m <= 1000; m++){
	for (l=1; l <= 1000; l++){
	  for (k=1; k <= numchn; k++){
	    mytraces[m][l][k] = numchn + 1 - k;
	  };
	};
      };
      break;
    };
  default: 
    {
      printf("INVALID INITIALIZATION - FATAL ERROR\n");
      return 1;
    }
  } //end of switch
  return 0;
}

int mstack::sort_and_store (int numchn, int m, int l){
  float scratch [129], temp;
  int k, k1, k2;

  for (k=1; k <= numchn ; k++){
    scratch[k] = mytraces [m][l-1+k][k];
  }

  for (k1=1; k1 <= numchn; k1++){
    for (k2=1; k2 <= numchn-1; k2++){
      if (scratch[k2] > scratch[k2+1]){
	temp = scratch[k2];
	scratch[k2]=scratch[k2+1];
	scratch[k2+1]=temp;
      }
    }
  }

  k1 = (numchn +1)/2;
  k2 = numchn - ((numchn - 1)/2);
  mytraces [m][l][129] = 0.5 * (scratch[k1] + scratch[k2]);

  return 0;
}

hpx::threads::thread_state 
mstack::sort_plane(hpx::threads::thread_self& s, hpx::applier::applier &app, int m, int numchn, hpx::lcos::barrier & b)
{
  //this loop is a candidate for HPX parallelization later: 
  for (int l=1; l <= 1001-numchn; l++){
    sort_and_store (numchn, m, l);
  }
  b.wait(s);
  return hpx::threads::terminated;
}

//thread that runs on a remote location
//run mstack, then return the result to one of the 6 the master node threads
hpx::threads::thread_state 
mstack::do_run (hpx::threads::thread_self& s, hpx::applier::applier &app, return_set * r, int j, int numchn) 
{
  initialize (j, numchn);
  hpx::lcos::barrier b (1001);
  for (int m=1; m <= 1000; m++){
    hpx::applier::register_work (app, boost::bind(&mstack::sort_plane, this, _1, boost::ref(app), m, numchn, boost::ref(b)));
  }
  b.wait(s);
  (*r).x = mytraces[1][1][129];
  (*r).y = mytraces[1][800][129];
  (*r).z = mytraces[1000][1][129]; 

  return hpx::threads::terminated; 
}
