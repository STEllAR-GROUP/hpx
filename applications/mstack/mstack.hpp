/* 
Mstack is a benchmark that simulates one stage of post-processing 
(by means of median stacking) done on seismic reflection data. 
More information is available here: 
http://www.eecis.udel.edu/~mpellegr/mstack/

-Mark Pellegrini (markpell@udel.edu)
August, 2008 
*/

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
