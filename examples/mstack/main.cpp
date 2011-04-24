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
#include "mstack.cpp"

hpx::naming::id_type nonlocal_gids [6]; 

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
