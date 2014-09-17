//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/util/unwrapped.hpp>

#define UW hpx::util::unwrapped2

int main()
{
    boost::uint64_t nt = 45; // Number of steps.
    boost::uint64_t nx = 10; // Number of grid points.

    typedef std::vector<hpx::shared_future<double> > space;
    typedef std::vector<space> spacetime;
     
    typedef spacetime::size_type index;

    // U[t][i] is the state of position i at time t.
    spacetime U(nt); for (space& s : U) s.resize(nx);

    // Initial conditions:
    //   f(0, i) = i
    for (index i = 0; i < nx; ++i)
        U[0][i] = hpx::make_ready_future(double(i));

    for (index t = 0; t < nt - 1; ++t) {
        // Our operator:
        //   f(t+1, i) = f(t, i-1) + f(t, i) + f(t, i+1)  
        auto Op = [](double a, double b, double c) { return a+b+c; };

        // Boundary conditions:
        //   f(t+1, 0)    = f(t, 0)    + f(t, 1) 
        //   f(t+1, nx-1) = f(t, nx-2) + f(t, nx-1) 
        auto BC = [](double a, double b) { return a+b; }; 

        // WHEN U[t][0] and U[t][1] have been computed, THEN we can compute U[t+1][0]
        U[t+1][0]    = hpx::when_all(U[t][0], U[t][1]).then(UW(BC));

        // WHEN U[t][nx-2] and U[t][nx-1] have been computed, THEN we can compute U[t+1][nx-1]
        U[t+1][nx-1] = hpx::when_all(U[t][nx-2], U[t][nx-1]).then(UW(BC));

        // WHEN U[t][i-1], U[t][i] and U[t][i+1] have been computed, THEN we can compute U[t+1][i]
        for (index i = 1; i < nx - 1; ++i)
            U[t+1][i] = hpx::when_all(U[t][i-1], U[t][i], U[t][i+1]).then(UW(Op));
    }

    // Now the asynchronous computation is running; the above for loop does not
    // wait on anything. There is no implicit waiting at the end of each timestep;
    // the computation of each U[t][i] will begin when as soon as its dependencies
    // are ready and hardware is available.

    // Wait for the solution at time 'nt'.
    space solution = hpx::when_all(U[nt-1]).get(); 

    for (index i = 0; i < nx; ++i)
        std::cout << "U[" << (nt-1) << "][" << i << "] = " << solution[i].get()
                  << std::endl;

    return 0;
}


