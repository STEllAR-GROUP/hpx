//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_FLUX_ACCUMULATE_HPP
#define HPX_EXAMPLES_MINI_FLUX_ACCUMULATE_HPP

#include <examples/mini_ghost/grid.hpp>
#include <examples/mini_ghost/stencils.hpp>

namespace mini_ghost {
    template <typename Real>
    Real flux_accumulate(
        grid<Real> const & g
      , std::size_t stencil
      , std::size_t my_px
      , std::size_t my_py
      , std::size_t my_pz
      , std::size_t npx
      , std::size_t npy
      , std::size_t npz
    )
    {
        Real flux = 0.0;
        Real divisor = 0.0;
        switch (stencil)
        {
            case STENCIL_2D5PT:
                divisor = 1.0f/5.0f;
                break;
            case STENCIL_2D9PT:
                divisor = 1.0f/9.0f;
                break;
            case STENCIL_3D7PT:
                divisor = 1.0f/7.0f;
                break;
            case STENCIL_3D27PT:
                divisor = 1.0f/27.0f;
                break;
        }

        if(my_px == 0)
        {
            for(std::size_t z = 1; z < g.nz_-1; ++z)
            {
                for(std::size_t y = 1; y < g.ny_-1; ++y)
                {
                    flux += g(1, y, z) * divisor;
                }
            }
        }

        if(my_px == npx - 1)
        {
            for(std::size_t z = 1; z < g.nz_-1; ++z)
            {
                for(std::size_t y = 1; y < g.ny_-1; ++y)
                {
                    flux += g(g.nx_ - 1, y, z) * divisor;
                }
            }
        }

        if(my_py == 0)
        {
            for(std::size_t z = 1; z < g.nz_-1; ++z)
            {
                for(std::size_t x = 1; x < g.nx_-1; ++x)
                {
                    flux += g(x, 1, z) * divisor;
                }
            }
        }

        if(my_py == npy - 1)
        {
            for(std::size_t z = 1; z < g.nz_-1; ++z)
            {
                for(std::size_t x = 1; x < g.nx_-1; ++x)
                {
                    flux += g(x, g.ny_ - 1, z) * divisor;
                }
            }
        }

        if(my_pz == 0)
        {
            for(std::size_t y = 1; y < g.ny_-1; ++y)
            {
                for(std::size_t x = 1; x < g.nx_-1; ++x)
                {
                    flux += g(x, y, 1) * divisor;
                }
            }
        }

        if(my_pz == npz - 1)
        {
            for(std::size_t y = 1; y < g.ny_-1; ++y)
            {
                for(std::size_t x = 1; x < g.nx_-1; ++x)
                {
                    flux += g(x, y, g.nz_ - 1) * divisor;
                }
            }
        }
        return flux;
    }
}

#endif
