/* 
Mstack is a benchmark that simulates one stage of post-processing 
(by means of median stacking) done on seismic reflection data. 
More information is available here: 
http://www.eecis.udel.edu/~mpellegr/mstack/

-Mark Pellegrini (markpell@udel.edu)
August, 2008 
*/

#include <iostream>
#include <cstdio>
#include <hpx/hpx.hpp> 
#include <hpx/lcos/barrier.hpp>
#include <boost/bind.hpp>

#define MAGICNUM1 100 
#define MAGICNUM2 (hpx::components::component_type) 200 

struct return_set{
  float x;
  float y;
  float z; 
};

hpx::naming::id_type nonlocal_gids [6]; 


class mstack: public hpx::components::simple_component_base <mstack>{
 private:
  float mytraces [1001][1001][130];
  int sort_and_store(int, int, int);
  int initialize (int, int);
  hpx::threads::thread_state sort_plane (hpx::threads::thread_self &, hpx::applier::applier &, int, int, hpx::lcos::barrier &);

 public:
  hpx::threads::thread_state do_run (hpx::threads::thread_self &, hpx::applier::applier &, return_set *, int, int);  
  typedef hpx::actions::result_action2 <mstack, return_set, MAGICNUM1, int, int, &mstack::do_run> do_run_action;  
  enum {value = MAGICNUM2};
};

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

//master node thread - to be run 6 times concurrently 
//create the remote instance of mstack, run it, print the results, then terminate
hpx::threads::thread_state 
onepass (hpx::threads::thread_self& s, hpx::applier::applier &app, int numchn, int j)
{
  printf("Invoking onepass - %d\n", j);

  hpx::naming::id_type gid0 = nonlocal_gids[0]; //should be [j-1] instead of 0 -- debugging  
  hpx::naming::id_type gid = create (app, s, gid0, MAGICNUM2);  
  hpx::lcos::eager_future <mstack::do_run_action, return_set> ef (app, gid, j, numchn); 
  return_set r = ef.get_result(s);

  printf ("mytraces[1][1][129] %f\n", r.x);
  printf ("mytraces[1][800][129] %f\n", r.y);
  printf ("mytraces[1000][1][129] %f\n", r.z);
  return hpx::threads::terminated;
}


/*
prompt the user for the number of channels
invoke 6 threads on the current (master) node to fork
the invoking thread terminates, while (in the onepass function) 
each of those 6 forked threads invokes 1 instance of mstack on a remote location
*/
hpx::threads::thread_state 
hpx_main (hpx::threads::thread_self& s, hpx::applier::applier &app)
{
  //prompt for number of channels
  int numchn, valid_value; 


  //get all non-local GIDS and store six of them 
  std::vector<hpx::naming::id_type> prefixes;
  app.get_dgas_client().get_prefixes(prefixes);

  int x = 0; 
  std::vector<hpx::naming::id_type>::iterator end = prefixes.end();
  for (std::vector<hpx::naming::id_type>::iterator it = prefixes.begin(); it != end; ++it){
    // do something with the gid if it's not the current locality
    if (hpx::naming::get_prefix_from_id(app.get_prefix()) != hpx::naming::get_prefix_from_id(*it))
      {
	//do_something_with_locality_gid(*it);
	if (x < 6){
	  nonlocal_gids[x] = hpx::naming::get_prefix_from_id(*it); 
	  x++;
	}
      }
  }


  valid_value = 0;
  while ( ! valid_value){
    printf ("How many channels (2-128)?\n");
    scanf ("%d",&numchn);
    if (numchn < 2 || numchn > 128)
      printf ("Invalid Response, please try again\n");
    else
      valid_value = 1;
  };  

  for (int j = 1; j <= 6; j++){
    hpx::applier::register_work(app, boost::bind(onepass, _1, boost::ref(app), numchn, j));
  }
  return hpx::threads::terminated;
}

//copied from https://svn.cct.lsu.edu/repos/projects/parallex/trunk/hpx/examples/accumulator/accumulator_client.cpp
int main(int argc, char* argv[])
{
  try {
    // Check command line arguments.
    std::string hpx_host, dgas_host;
    unsigned short hpx_port, dgas_port;

    // Check command line arguments.
    if (argc != 5) {
      std::cerr << "Usage: mstack hpx_addr hpx_port dgas_addr "
	"dgas_port" << std::endl;
      std::cerr << "Try: mstack <your_ip_addr> 7911 "
	"<your_ip_addr> 7912" << std::endl;
      return -3;
    }
    else {
      hpx_host = argv[1];
      hpx_port = boost::lexical_cast<unsigned short>(argv[2]);
      dgas_host = argv[3];
      dgas_port  = boost::lexical_cast<unsigned short>(argv[4]);
    }

    // initialize and start the HPX runtime
    hpx::runtime rt(dgas_host, dgas_port, hpx_host, hpx_port);
    rt.run(hpx_main);

  }
  catch (std::exception& e) {
    std::cerr << "std::exception caught: " << e.what() << "\n";
    return -1;
  }
  catch (...) {
    std::cerr << "unexpected exception caught\n";
    return -2;
  }
  return 0;
}
