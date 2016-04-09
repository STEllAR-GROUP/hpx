//  Copyright (c)        2012 Zach Byerly
//  Copyright (c) 2011 - 2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2012    Jonathan Parziale
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This is a 1 dimensional hydrodynamics code, using a simple first-order
//  upwind advection scheme (to keep dependencies simple).  1st order in time.
//  It employs a predictive timestep method in order to eliminate global
//  barriers after every timestep. No gravity. A dual-energy formalism is also
//  used to allow for heating in the shocks.

// INCLUDES
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/lcos/wait_each.hpp>

#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>

#include <mutex>

using hpx::naming::id_type;
using hpx::naming::invalid_id;

using hpx::lcos::future;
using hpx::lcos::wait;
using hpx::lcos::wait_each;
using hpx::async;

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
id_type here = invalid_id;
boost::uint64_t nt = 0;
boost::uint64_t nx = 0;
boost::uint64_t n_predict = 0;
double fluid_gamma = 0.0;
double x_min = 0.0;
double x_max = 0.0;
double dx = 0.0;
double cfl_factor = 0.0;
double cfl_predict_factor = 0.0;
double ptime;

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
    , tau(0.0)   // internal energy density (etot - kinetic energy)
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
    , tau(other.tau)
    , computed(other.computed)
  {}

  // assignment operator:
  //
  //   cell c1, c2;
  //   c1 = c2; // invoked by this syntax
  //  Bryce:  I think i'd like to change this so the "calculated" is not
  //  copied in the assignment op
  cell& operator=(
       cell const& other
       )
  {
    // first, we lock both the mutex of this cell, and the mutex of the other
    // cell
    //    std::lock_guard<hpx::lcos::local::mutex> this_lock(mtx),
    //      other_lock(other.mtx);

    rho = other.rho;
    mom = other.mom;
    etot = other.etot;
    tau = other.tau;
    computed = other.computed;

    // return a reference to ourselves
    return *this;
  }

  // dummy serialization functionality
  template <typename Archive>
  void serialize(Archive &, unsigned) {}

  mutable hpx::lcos::local::mutex mtx;
  double rho;
  double mom;
  double etot;
  double tau;
  bool computed;
};
// this is a single element of a "time array" that will be 1D array
// containing information about timestep size. the index will be
// the integer timestep number, so time_array[timestep].dt will be
// the timestep size at that timestep
struct time_element{
  // default constructor
  time_element()
    : dt(0.0)
    , elapsed_time(0.0)
    , computed(false),fluid_future(0),fluid(0)
  {}

  time_element(boost::uint64_t number_of_cells)
      :fluid_future(number_of_cells)
      ,fluid(number_of_cells)
      {}
  // copy constructor
  time_element(
       time_element const& other
       )
    : dt(other.dt)                     // timestep size
    , elapsed_time(other.elapsed_time) // elapsed time
    , computed(other.computed)
    ,fluid_future(other.fluid_future)
    ,fluid(other.fluid)
    {}

  time_element& operator=(time_element const& rhs)
  {
    if (this != &rhs)
    {
      dt = rhs.dt;
      elapsed_time = rhs.elapsed_time;
      computed = rhs.computed;
      fluid=rhs.fluid;
      fluid_future=rhs.fluid_future;

    }
    return *this;
  }

  hpx::lcos::local::mutex mtx;
  double dt;
  double elapsed_time;
  bool computed;
  double physics_time;
  std::vector<hpx::lcos::shared_future<cell> > fluid_future;//future for each cell
  std::vector<cell> fluid;
};
// declaring time_array




// Object to store the Fluid separated in cell and computed by a Time Zone of tie Steps
// Will store the 2d grid created by the division of the fluid
// into cells and the computation over time
// Will be able to Retrieve,remove,add a timestep to the grid

class One_Dimension_Grid
{
public:
    One_Dimension_Grid():time_array(0)
    {}
    // ~One_Dimension_Grid();
    void remove_bottom_time_step();//takes the timesteps position in the vector
    void addNewTimeStep();

public:
   std::vector<time_element> time_array;
   //pointer to the Grid we will create whden the user starts a simulation

};
void One_Dimension_Grid::remove_bottom_time_step()
{
    time_array.pop_back();
}
void One_Dimension_Grid::addNewTimeStep()
{
    time_array.insert(time_array.begin(),time_array.at(nx-1));
}
/*One_Dimension_Grid::~One_Dimension_Grid()
{
    for(boost::uint64_t i=0;i<nt;i++)
    {
        time_array.pop_back();
    }
    delete &time_array;
}*/

// declaring grid of all cells for all timesteps
One_Dimension_Grid grid;
// forward declaration of the compute function
cell compute(boost::uint64_t timestep, boost::uint64_t location);
double timestep_size(boost::uint64_t timestep);
cell initial_sod(boost::uint64_t location);
double get_pressure(cell input);

// Wrapping in plain_action
HPX_PLAIN_ACTION(compute);

typedef hpx::lcos::future<cell> compute_future;

// this will return the timestep size.  The timestep index will refer to the
// timestep where it will be USED rather than the timestep where it was
// calculated.
double timestep_size(boost::uint64_t timestep)
{
  // locking

  std::lock_guard<hpx::lcos::local::mutex> l(grid.time_array.at(timestep).mtx);

  // if it has already been calculated, then just return the value
  if (grid.time_array.at(timestep).computed)
  {return grid.time_array.at(timestep).dt;}
  //  cout << (boost::format("calculating timestep, ts=%1% \n") % timestep) << flush;


  // if the current timestep is less than n_predict, then we manually
  // decide the timestep
  if (timestep < n_predict)
    {
      grid.time_array.at(timestep).computed = true;
      grid.time_array.at(timestep).dt = dx*0.033;// this should be fine unless
        // the initial conditions are changed
        if(timestep>0&&grid.time_array.at(timestep-1).computed)
            grid.time_array[timestep].physics_time =
            (grid.time_array.at(timestep-1).physics_time+grid.
                time_array.at(timestep).dt);
  //    time_array[timestep].dt = cfl_predict_factor*dt_cfl;
        else if(timestep==0)
        {
            grid.time_array.at(timestep).physics_time=grid.time_array.at(timestep).dt;
        }
      return grid.time_array.at(timestep).dt;
    }

  // send back the compute futures for the whole grid
  // n_predict timesteps previous to the one we want to decide
  // the timestep for
  //  cout << (boost::format("pushing back futures for ts calc, ts=%1% \n") % timestep)
  //  << flush;
  if(timestep>=n_predict)
  {
  for (boost::uint64_t i=0;i<nx;i++)
      grid.time_array.at(timestep).fluid_future.push_back(async<compute_action>
          (here,timestep-n_predict,i));
  }

  double dt_cfl = 1000.0;

   wait_each(
      hpx::util::unwrapped([&](cell const& this_cell)
      {
      // look at all of the cells at a timestep, then pick the smallest
      // dt_cfl = cfl_factor*dx/(soundspeed+absolute_velocity)
      double abs_velocity = this_cell.mom/this_cell.rho;
      double pressure = get_pressure(this_cell);
      double soundspeed = sqrt(fluid_gamma*pressure/this_cell.rho);
      double dt_cfl_here = cfl_factor*dx/(soundspeed+abs_velocity);
      if (dt_cfl_here <=  0.0)
        {
          cout << (boost::format("error: CFL value can't be zero")) << flush;
          //error, quit everything
        }
      if (dt_cfl_here < dt_cfl)
        dt_cfl = dt_cfl_here;
     }),
     grid.time_array.at(timestep).fluid_future);

  // initialize dt_cfl to some arbitrary high value


  // wait for an array of futures
  /*wait_each(grid.time_array,
    hpx::util::unwrapped([&](cell const& this_cell)
    {
      // look at all of the cells at a timestep, then pick the smallest
      // dt_cfl = cfl_factor*dx/(soundspeed+absolute_velocity)
      double abs_velocity = this_cell.mom/this_cell.rho;
      double pressure = get_pressure(this_cell);
      double soundspeed = sqrt(fluid_gamma*pressure/this_cell.rho);
      double dt_cfl_here = cfl_factor*dx/(soundspeed+abs_velocity);
      if (dt_cfl_here <=  0.0)
        {
          cout << (boost::format("error: CFL value can't be zero")) << flush;
          //error, quit everything
        }
      if (dt_cfl_here < dt_cfl)
        dt_cfl = dt_cfl_here;
    }));
*/


  if(dt_cfl > 999.0)
    {
      cout << (boost::format("error: CFL value too high")) << flush;
      // error, quit everything
    }

  // we don't want to let the timestep increase too quickly, so
  // we only let it increase by 25% each timestep
  grid.time_array.at(timestep).computed = true;
  grid.time_array.at(timestep).dt = (std::min)(
                                     cfl_predict_factor*dt_cfl
                                     ,
                                     1.25*grid.time_array.at(timestep-1).dt);

  //  cout << (boost::format("timestep = %1%, dt = %2%\n")
  //  % timestep % time_array[timestep].dt) << flush;
  return grid.time_array[timestep].dt;
}

cell compute(boost::uint64_t timestep, boost::uint64_t location)
{
    std::lock_guard<hpx::lcos::local::mutex> l(grid.time_array.at(timestep)
        .fluid.at(location).mtx);

  // if it is already computed then just return the value
    if (grid.time_array.at(timestep).fluid.at(location).computed == true)
        return grid.time_array.at(timestep).fluid.at(location);

  //  cout << (boost::format("computing new value, loc = %1%,ts=%2% \n")
  //  % location % timestep) << flush;

  //initial values
  if (timestep == 0)
    {
      //  cout << (boost::format("calling initial_sod, loc = %1%,ts=%2% \n")
      //  % location % timestep) << flush;
        grid.time_array.at(timestep).fluid.at(location) = initial_sod(location);
      //  cout << (boost::format("returning value, loc = %1%,ts=%2% \n")
      //  % location % timestep) << flush;
        grid.time_array.at(timestep).fluid.at(location).computed = true;
        return grid.time_array.at(timestep).fluid.at(location);
    }

   //boundary conditions (using sod shock tube boundaries)
  if ( (location == 0) || (location == nx-1) )
    {
      grid.time_array.at(timestep).fluid.at(location) = initial_sod(location);
      grid.time_array.at(timestep).fluid.at(location).computed = true;
      return grid.time_array.at(timestep).fluid.at(location);
    }

  //now we have to actually compute some values.

  //these are the dependencies, or "stencil"
  //if(timestep<0) // unsigned comparision always false
  //      return grid.time_array.at(timestep).fluid.at(location);

  compute_future nleft = async<compute_action>(here,timestep-1,location-1);
  compute_future nmiddle = async<compute_action>(here,timestep-1,location);
  compute_future nright = async<compute_action>(here,timestep-1,location+1);

  // OR is this the correct way to do it?
  //future<cell> left;
  //left = async<compute_action>(here,timestep-1,location-1);
  //future<cell> middle;
  //middle = async<compute_action>(here,timestep-1,location);
  //future<cell> right;
  //right = async<compute_action>(here,timestep-1,location+1);

  cell now;

  cell left   = nleft.get();
  cell middle = nmiddle.get();
  cell right  = nright.get();

  // calling this function may or may not make futures
  double dt = timestep_size(timestep);

  now.rho = middle.rho;
  now.mom = middle.mom;
  now.etot = middle.etot;
  now.tau = middle.tau;

  // now that we have all of the information we need, we can proceed with
  // the physics part of the update

  double right_pressure = get_pressure(right);
  double left_pressure = get_pressure(left);
  double middle_pressure = get_pressure(middle);

  // first we will calculate the advection of all of the variables
  // through the left face of the cell.
  // start by calculating the velocity on the left face
  // if this velocity is positive, then fluid flows from the cell to the left.
  // if it is negative, fluid flows out of the middle cell.
  double velocity_left = (left.mom+middle.mom)/(left.rho+middle.rho);
  if (velocity_left > 0.0)
    {
      now.rho +=  left.rho*velocity_left*dt/dx;
      now.mom +=  left.mom*velocity_left*dt/dx;
      now.etot += (left.etot+left_pressure)*velocity_left*dt/dx;
      now.tau +=  left.tau*velocity_left*dt/dx;
    }
  else
    {
      now.rho +=  middle.rho*velocity_left*dt/dx;
      now.mom +=  middle.mom*velocity_left*dt/dx;
      now.etot += (middle.etot+middle_pressure)*velocity_left*dt/dx;
      now.tau +=  middle.tau*velocity_left*dt/dx;
    }

  // now repeat the process for the right side
  double velocity_right = (right.mom+middle.mom)/(right.rho+middle.rho);
  if (velocity_right < 0.0)
    {
      now.rho -=  right.rho*velocity_right*dt/dx;
      now.mom -=  right.mom*velocity_right*dt/dx;
      now.etot -= (right.etot+right_pressure)*velocity_right*dt/dx;
      now.tau -=  right.tau*velocity_right*dt/dx;
    }
  else
    {
      now.rho -=  middle.rho*velocity_right*dt/dx;
      now.mom -=  middle.mom*velocity_right*dt/dx;
      now.etot -= (middle.etot+middle_pressure)*velocity_right*dt/dx;
      now.tau -=  middle.tau*velocity_right*dt/dx;
    }

  // source terms
  now.mom += 0.5*dt*(left_pressure - right_pressure)/(dx);


  // check for CFL (courant friedrichs levy) violation (makes code unstable)
  double soundspeed = std::sqrt(fluid_gamma*middle_pressure/middle.rho);
  double abs_velocity = (std::max)(velocity_right,velocity_right);
  double dt_cfl_here = cfl_factor*dx/(soundspeed+abs_velocity);
  if (dt_cfl_here > timestep)
    {
      cout << (boost::format("error! cfl violation!\n")) << flush;
      cout << (boost::format("loc=%1% ts=%2%\n") % location % timestep) << flush;
      cout << (boost::format("dt_cfl_here=%1% dt=%2%\n") % dt_cfl_here % dt ) << flush;

      // Bryce: I should add some real error handling. can you help me with this?
      // error, quit everything
    }

  double e_kinetic = 0.5*middle.mom*middle.mom/middle.rho;
  double e_internal = middle.etot - e_kinetic;
  //dual energy formalism
  if ( std::abs(e_internal) > 0.1*middle.etot)
    {
      //cout << (boost::format("gas is shocking!\n")) << flush;
      now.tau = pow(e_internal,(1.0/fluid_gamma));
    }
  // cout << (boost::format("computing new value, loc = %1%, ts= %2%\n")
  //           % location % timestep) << flush;
  // cout << (boost::format("loc = %1%, rho = %2%\n") % location % left.rho) << flush;
  // cout << (boost::format("loc = %1%, mom = %2%\n") % location % left.mom) << flush;
  // cout << (boost::format("loc = %1%, etot = %2%\n") % location % left.etot) << flush;
  // cout << (boost::format("loc = %1%, vel left = %2%\n") % location % velocity_left)
  //      << flush;

  //  if (location == 1)
  //    cout << (boost::format("calculating timestep = %1%\n") % timestep) << flush;

  grid.time_array.at(timestep).fluid.at(location) = now;
  grid.time_array.at(timestep).fluid.at(location).computed = true;
  bool time_step_complete= false;
  for(boost::uint64_t i=0;i<nx;i++)
  {
      if(grid.time_array[0].fluid.at(i).computed
          &&grid.time_array[1].fluid.at(i).computed)
        time_step_complete=true;
      else
          time_step_complete=false;
  }
  if(time_step_complete&&!(grid.time_array.at(nt-1).physics_time>=ptime))
  {
      grid.remove_bottom_time_step();
      grid.addNewTimeStep();
  }

  return grid.time_array.at(timestep).fluid.at(location);
}

double get_pressure(cell input)
{
  double pressure = 0.0;
  double e_kinetic = 0.5*input.mom*input.mom/input.rho;
  double e_internal = input.etot - e_kinetic;

  // dual energy
  if ( std::abs(e_internal) > 0.001*input.etot )
    {
      pressure = (fluid_gamma-1.0)*e_internal;
    }
  else
    {
      pressure = (fluid_gamma-1.0)*pow(input.tau,fluid_gamma);
    }

  return pressure;
}

cell initial_sod(boost::uint64_t location)
{

  //  cout << (boost::format("initial_sod, loc = %1%\n") % location) << flush;

  // calculate what the x coordinate is here
  double x_here = (location-0.5)*dx+x_min;

  cell cell_here;
  double e_internal = 0.0;

  // This is the Sod Shock Tube problem, which has a known analytical
  // solution.  A mass and energy contact discontinuity is placed on the grid.
  // A shockwave then forms and propogates through the grid.
  if (x_here < -0.1)
    {
      cell_here.rho = 1.0;
      e_internal = 2.5;
    }
  else
    {
      cell_here.rho = 0.125;
      e_internal = 0.25;
    }

  cell_here.mom = 0.0;
  cell_here.tau = pow(e_internal,(1.0/fluid_gamma));
  cell_here.etot = e_internal;  // ONLY true when mom=0, not in general!

  //  cout << (boost::format("returning from initial_sod, loc = %1%\n") % location)
  //       << flush;
  return cell_here;
}

cell get_analytic(double x_here, double time)
{
  cell output;

  // values for analytic solution come from Patrick Motl's dissertation

  //  cout << (boost::format("calculating analytic... x=%1% t=%2%\n")
  //  % x_here % time) << flush;

  double x_0 = -0.1;

  double c_1 = 1.183;
  double x_head = x_0 - c_1*time;

  double w_3 = 0.9274;
  double c_3 = 0.9978;

  double x_tail = x_0 + (w_3 - c_3)*time;

  double w_4 = 0.9274;

  double x_contact = x_0 + w_4*time;

  double c_5 = 1.058;
  double p_4 = 0.3031;
  double p_5 = 0.1;

  double W = c_5*std::pow( (1.0+(fluid_gamma+1.0)*(p_4 - p_5)/(2.0*fluid_gamma*p_5)),
      0.5  );

  double x_shock = x_0 + W*time;

  if (x_here < x_head)  // region 1
    {
      output.rho = 1.0;
      output.tau = 1.924;
      output.mom = 0.0;
    }
  else if (x_here < x_tail) // region 2
    {
      double w_2 = 2.0*(c_1+(x_here-x_0)/time)/(fluid_gamma+1.0);
      double exponent = 2.0/(fluid_gamma-1.0);
      output.rho = std::pow( (1.0-(fluid_gamma-1.0)*w_2/(2.0*c_1)), exponent);
      output.mom = output.rho*w_2;
      output.tau = pow( (1.0-0.5*(fluid_gamma-1)*(w_2/c_1)),
          exponent)/pow(fluid_gamma-1.0,1.0/fluid_gamma);
    }
  else if (x_here < x_contact) // region 3
    {
      output.rho = 0.4263;
      output.mom = 0.9274*output.rho;
      output.tau = 0.8203;
    }
  else if (x_here < x_shock) // region 4
    {
      output.rho = 0.2656;
      output.mom = 0.9274*output.rho;
      output.tau = 0.8203;
    }
  else // region 5
    {
      output.rho = 0.125;
      output.mom = 0.0;
      output.tau = 0.3715;
    }


  return output;
}



///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
  here = find_here();

  // some physics parameters
  nx = vm["nx-value"].as<boost::uint64_t>();
  nt = vm["nt-value"].as<boost::uint64_t>();
  n_predict = vm["npredict-value"].as<boost::uint64_t>();
  ptime=vm["ptime-value"].as<double>();
  fluid_gamma = 1.4;

  x_min = -0.5;
  x_max = 0.5;
  dx = (x_max - x_min)/(nx-2);

  cfl_factor = 0.5;
  cfl_predict_factor = 0.8;

  cout << (boost::format("nt = %1%\n") % nt) << flush;
  cout << (boost::format("nx = %1%\n") % nx) << flush;
  cout << (boost::format("n_predict = %1%\n") % n_predict) << flush;

  // allocating the time array
  grid.time_array = std::vector<time_element>(nt);

  // allocating the grid 2d array of all of the cells for all timesteps
  for(boost::uint64_t i=0;i<nt;i++)
  {
      grid.time_array[i].fluid=std::vector<cell>(nx);
  }
    //HPX stuff goes here

    // Keep track of the time required to execute.
    high_resolution_timer t;

    timestep_size(0);
    for (boost::uint64_t i=0;i<nx;i++)
           grid.time_array[0].fluid_future.push_back(async<compute_action>(here,nt-1,i));
    // open file for output
    std::ofstream outfile;
    outfile.open ("output.dat");

    wait(grid.time_array[0].fluid_future, [&](std::size_t i, cell n)
         { double x_here = (i-0.5)*dx+x_min;
           double pressure_here = get_pressure(n);
           //double tauoverrho = n.tau/n.rho;
           double velocity_here = n.mom/n.rho;

           double e_kinetic = 0.5*n.mom*n.mom/n.rho;
           double e_internal = n.etot - e_kinetic;
           double tauoverrho =  pow(e_internal,(1.0/fluid_gamma))/n.rho;
           //           double e_internal2 = pow(n.tau,fluid_gamma);

           outfile << (boost::format("%1% %2% %3% %4% %5%\n")
               % x_here % n.rho % pressure_here % tauoverrho % velocity_here)
               << flush; });
           //           outfile << (boost::format("%1% %2% %3% %4% %5%\n")
           // % x_here % n.rho % pressure_here % e_internal % e_internal2)
           // << flush; });

    outfile.close();

    std::ofstream outfile2;
    outfile2.open ("time.dat");

    boost::uint64_t i;
    // writing the "time array" to a file
    grid.time_array[0].elapsed_time = grid.time_array[0].dt;
    for (i=1;i<nt;i++)
      grid.time_array[i].elapsed_time = grid.time_array[i-1].elapsed_time
        + grid.time_array[i].dt;

    for (i =0;i<nt;i++)
      {
        outfile2 << (boost::format("%1% %2% %3%\n")
            % i % grid.time_array[i].dt % grid.time_array[i].elapsed_time)
            << flush;
      }
    outfile2.close();

    // writing the analytic solution for the final time to a file
    std::ofstream analytic_file;
    analytic_file.open ("analytic.dat");
    double total_mass = 0.0;
    for (i =0;i<nx;i++)
      {
        double x_here = (i-0.5)*dx+x_min;
        cell analytic = get_analytic(x_here,grid.time_array[nt-1].elapsed_time);
        double velocity_here = analytic.mom/analytic.rho;
        double tauoverrho = analytic.tau/analytic.rho;
        double pressure_here = get_pressure(analytic);
        analytic_file << (boost::format("%1% %2% %3% %4% %5%\n")
            % x_here % analytic.rho % pressure_here % tauoverrho % velocity_here)
            << flush;
        total_mass += grid.time_array[nt-1].fluid[i].rho*dx;
      }
    analytic_file.close();

    cout << (boost::format("total mass = %1%\n") % total_mass ) << flush;
    char const* fmt = "wall elapsed time: %1% [s]\n";
    std::cout << (boost::format(fmt) % t.elapsed());
    char const* fmt0 = "code elapsed time: %1%\n";
    double t_code_time= grid.time_array[0].elapsed_time;

        t_code_time+=grid.time_array[nt-1].elapsed_time;

      std::cout << (boost::format(fmt0) %  t_code_time);


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

  cmdline.add_options()
    ( "nx-value"
      , value<boost::uint64_t>()->default_value(5)
      , "nx parameter of the wave equation")

    ( "nt-value"
      , value<boost::uint64_t>()->default_value(10)
      , "nt parameter of the wave equation")

    ( "npredict-value"
      , value<boost::uint64_t>()->default_value(10)
      , "prediction parameter of the wave equation")

      ( "ptime-value"
      , value<double>()->default_value(2.50)
      , "Physics time to run the simulation to")
    ;



  // Initialize and run HPX.
  return init(cmdline, argc, argv);
}

