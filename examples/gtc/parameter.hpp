//  Copyright (c) 2009-2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_GTCPARAMETER_NOV_19_2011_0834AM)
#define HPX_COMPONENTS_GTCPARAMETER_NOV_19_2011_0834AM

#include <vector>

#include <boost/config.hpp>

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable:4251)
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace gtc
{

namespace detail {

struct HPX_COMPONENT_EXPORT parameter
{
    // Default control parameters
    bool irun;  // 0 for initial run, any non-zero value for restart
    std::size_t mstep; // # of time steps
    std::size_t msnap; // # of snapshots
    std::size_t ndiag; // do diag when mod(istep,ndiag)=0
    double nonlinear; // 1.0 nonlinear run, 0.0 linear run
    std::size_t nhybrid; // 0: adiabatic electron, >1: kinetic electron
    double paranl; // 1: keep parallel nonlinearity
    bool mode00; // 1 include (0,0) mode, 0 exclude (0,0) mode

    // run size
    double tstep;  // time step (unit=L_T/v_th), tstep*\omega_transit<0.1
    std::size_t micell; // # of ions per grid cell
    std::size_t mecell; // # of electrons per grid cell
    std::size_t mpsi;   // total # of radial grid points
    std::size_t mthetamax; // poloidal grid, even and factors of 2,3,5 for FFT
    std::size_t mzetamax; // total # of toroidal grid points, domain decomp.
    std::size_t npartdom; // number of particle domain partitions per tor dom.
    std::size_t ncycle; //  subcycle electron

    // run geometry
    double a; // minor radius, unit=R_0
    double a0; // inner boundary, unit=a
    double a1; // outer boundary, unit=a
    double q0; // q_profile, q=q0 + q1*r/a + q2 (r/a)^2
    double q1;
    double q2;
    double rc; //  kappa=exp{-[(r-rc)/rw]**6}
    double rw; // rc in unit of (a1+a0) and rw in unit of (a1-a0)

    // species information
    double aion; // species isotope #
    double qion; // charge state
    double aelectron;
    double qelectron;

    // equilibrium unit: R_0=1, Omega_c=1, B_0=1, m=1, e=1
    double kappati; // grad_T/T
    double kappate;  
    double kappan;       //  inverse of eta_i, grad_n/grad_T
    bool fixed_Tprofile; // Maintain Temperature profile (0=no, >0 =yes)
    double tite;  // T_i/T_e
    double flow0; // d phi/dpsi=gyroradius*[flow0+flow1*r/a+flow2*(r/a)**2]
    double flow1;
    double flow2;

    // physical unit
    double r0; // major radius (unit=cm)
    double b0; // on-axis vacuum field (unit=gauss)
    double temperature; // electron temperature (unit=ev)
    double edensity0;   // electron number density (1/cm^3)

    double utime;
    double gyroradius;
    double tauii;

    std::size_t mflux;
    std::size_t num_mode;
    std::size_t m_poloidal;

    // Output
    std::size_t output;
    std::size_t nbound; // 0 for periodic, >0 for zero boundary
    double umax; // unit=v_th, maximum velocity in each direction
    bool iload; // 0: uniform, 1: non-uniform
    bool track_particles; // 1: keep track of some particles
    bool nptrack; // track nptrack particles every time step
    bool rng_control; // controls seed and algorithm for random num. gen.
                      // rng_control>0 uses the portable random num. gen.
    std::size_t isnap;
    std::size_t idiag1;
    std::size_t idiag2;

    std::size_t numberpe;
    std::size_t ntoroidal;

    std::vector<std::size_t> nmode,mmode;
};

} // detail

struct HPX_COMPONENT_EXPORT parameter
{
    typedef detail::parameter value_type;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef value_type* pointer;
    typedef value_type const* const_pointer;

    boost::shared_ptr<value_type> p;

    parameter() : p(new value_type) {}

    pointer operator->()
    { return p.get(); }

    const_pointer operator->() const
    { return p.get(); }

    reference operator*()
    { return *p; }

    const_reference operator*() const
    { return *p; }
};

///////////////////////////////////////////////////////////////////////////////
}}}

#include <examples/gtc/serialize_parameter.hpp>

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif

