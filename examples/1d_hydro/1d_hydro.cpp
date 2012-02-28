//  Copyright (c)        2012 Zach Byerly
//  Copyright (c) 2011 - 2012 Bryce Adelstein-Lelbach 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This is a 1 dimensional hydrodynamics code, using a simple first-order
//  upwind advection scheme (to keep dependencies simple).  1st order in time.  
//  It employs a predictive timestep method in order to eliminate global 
//  barriers after every timestep. Gravity, if any, is given by a static
//  potential.  A dual-energy formalism is also used to allow for heating
//  in the shocks.  

// INCLUDES
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/async_future_wait.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>

using hpx::naming::id_type;
using hpx::naming::invalid_id;

using hpx::actions::plain_result_action2;

using hpx::lcos::promise;
using hpx::lcos::async;
using hpx::lcos::wait;
using hpx::lcos::eager_future;

using hpx::util::high_resolution_timer;

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using hpx::cout;
using hpx::flush;


// USING STATEMENTS

////////////////////////////////////////
// globals

// initialized in hpx_main
boost::uint64_t nt = 0;
boost::uint64_t nx = 0;

// this is a single element of a "time array" that will be 1D array
// containing information about timestep size. the index will be 
// the integer timestep number, so time_array[timestep].dt will be 
// the timestep size at that timestep
struct time{
  // default constructor
  time()
    : mtx()
    , dt(0.0)
    , elapsed_time(0.0)
    , computed(false)
  {}
  
  // copy constructor
  time(
       time const& other
       )
    : mtx()
    , dt(other.dt)                     // timestep size
    , elapsed_time(other.elapsed_time) // elapsed time
    , computed(other.computed)
  {}

  hpx::lcos::local_mutex mtx;
  double dt;
  double elapsed_time;
  bool computed;
}
// declaring time_array
  std::vector<time> time_array;


// this is the fundimental element of the hydrodynamics code, the 
// individual cell.  It stores all of the variables that the code 
// tracks their changes in time.  
struct cell{
  // default constructor
  cell()
    : mtx()
    , rho(0.0)    // mass density
    , mom(0.0)    // momentum density
    , etot(0.0)   // total energy density
    , eint(0.0)   // internal energy density (etot - kinetic energy)
    , computed(false)
  {}

  // copy constructor
  cell(
       cell const& other
       )
    : mtx()
    , rho(other.rho)
    , mom(other.mom)
    , etot(other.etot)
    , eint(other.eint)
    , computed(other.computed)
  {}

  hpx::lcos::local_mutex mtx;
  double rho;
  double mom;
  double etot;
  double eint;
  bool computed;
}
  
// declaring grid of all cells for all timesteps
  std::vector<std::vector<cell> > grid; 


// forward declaration of the compute function
  cell compute(boost::uint64_t timestep, boost::uint64_t location);

// Wrapping in plain_action
typedef plain_result_action2<
  // result type
  cell,

  // arguments 
  boost::uint64_t,
  boost::uint64_t,

  // function
  compute
  > compute_action;

// This generates the required boilerplate we need for remote invocation.
HPX_REGISTER_PLAIN_ACTION(compute_action);

typedef eager_future<compute_action> compute_future;

// this will return the timestep size.  The timestep index will refer to the 
// timestep where it will be USED rather than the timestep where it was 
// calculated.
double timestep_size(uint64_t timestep)
{
  // locking
  hpx::lcos::local_mutex::scoped_lock l(time_array[timestep].mtx);

  // look at all of the cells at a timestep, then pick the smallest
  // dt_cfl = cfl_factor*dx/(soundspeed+absolute_velocity)

}

cell 

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    
  here = find_here();

  cout << (boost::format("nt = %1%\n") % nt) << flush;
  cout << (boost::format("nx = %1%\n") % nx) << flush;

  // allocating the time array
  time_array = std::vector<time>(nt);
  
  // allocating the grid 2d array of all of the cells for all timesteps
  grid = std::vector<std::vector<cell> >(nt, std::vector<cell>(nx));

  {
    //HPX stuff goes here
  }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}

