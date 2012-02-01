/////////////////////////////////////////////
// 1-D wave equation example
// Zach Byerly Jan 27 2012
/////////////////////////////////////////////

//physics parameters
#define alpha_squared 0.25
#define dt 0.1
#define dx 0.1
#define c 0.5
#define nx 10
#define nt 10
#define pi 3.14

//include statements
#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/eager_future.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;
using hpx::actions::plain_result_action1;
using hpx::lcos::eager_future;
using hpx::util::high_resolution_timer;
using hpx::init;
using hpx::finalize;
using hpx::find_here;



//forward declaration of the wave function
double wave(int t, int x);

// Any global function needs to be wrapped into a plain_action if it should be
// invoked as a HPX-thread.
typedef plain_result_action2<
    double,          // result type
    int,          // arguments
    int,
    wave                 // function
> wave_action;

// this is to generate the required boilerplate we need for the remote
// invocation to work
HPX_REGISTER_PLAIN_ACTION(wave_action);

///////////////////////////////////////////////////////////////////////////////
// An eager_future is a HPX construct exposing the semantics of a Future
// object. It starts executing the bound action immediately (eagerly).
typedef eager_future<wave_action> wave_future;



double calculate_u_tplus_x(double u_t_xplus, double u_t_x, double u_t_xminus, double u_tminus_x)
{
	u_tplus_x = alpha_squared*(u_t_xplus + u_t_xminus) + 2.0*(1-alpha_squared)*u_t_x - u_tminus_x;
	return u_tplus_x;
}

double calculate_u_tplus_x_1st(double u_t_xplus, double u_t_x, double u_t_xminus, double u_dot)
{
	u_tplus_x = alpha_squared*(u_t_xplus + u_t_xminus) + 2.0*(1-alpha_squared)*u_t_x + dt*u_dot(t,x);
	return u_tplus_x;
}


double wave(int t, int x)
{
	if (t = 0) //first timestep are initial values
	{
		return sin(2*pi*x); // initial u(x) value
	}


	// NOT using ghost zones here... just letting the stencil cross the periodic boundary
	if (x = 0)
	{
		wave_future n1(find_here(),t-1,nx-1);
	} else
	{
		wave_future n1(find_here(),t-1,x-1);
	}

	wave_future n2(find_here(),t-1,x);

	if (x = nx-1)
	{
		wave_future n3(find_here(),t-1,0);
	} else
	{
		wave_future n3(find_here(),t-1,x+1);
	}

	u_t_xminus = n1.get(); //get the futures
	u_t_x = n2.get();
	u_t_xplus = n3.get();


	if (t = 1) //second time coordinate handled differently
	{
		u_dot = 0;// initial du/dt(x)
		return calculate_u_tplus_x_1st(u_t_xplus,u_t_x,u_t_xminus,u_dot);
	} else {
		wave_future n4(find_here(),t-2,x);
		u_tminus_x = n4.get();
		return calculate_u_tplus_x(u_t_xplus,u_t_x,u_t_xminus,_u_tminus_x);
	}



}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    // extract command line argument nt
    int nt_2 = vm["n-value"].as<boost::uint64_t>();

	int i;
	for(i=0;i<nx;i++) {
		wave_future n(find_here(),i,nt);
		double u = get.n();
		//decide how to output
	}
    finalize(); //finalize HPX
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "n-value" , value<boost::uint64_t>()->default_value(10),
            "number of timesteps")
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}
