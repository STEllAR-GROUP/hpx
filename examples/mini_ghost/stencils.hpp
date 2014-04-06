//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_STENCILS_HPP
#define HPX_EXAMPLES_MINI_GHOST_STENCILS_HPP

#include <examples/mini_ghost/grid.hpp>

namespace mini_ghost {

    static const std::size_t STENCIL_NONE   = 20;
    static const std::size_t STENCIL_2D5PT  = 21;
    static const std::size_t STENCIL_2D9PT  = 22;
    static const std::size_t STENCIL_3D7PT  = 23;
    static const std::size_t STENCIL_3D27PT = 24;

    template <std::size_t Stencil>
    struct stencils;

    template <>
    struct stencils<STENCIL_NONE>
    {
        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src)
        {
            for(std::size_t z = 1; z != dst.nz_-1; ++z)
            {
                for(std::size_t y = 1; y != dst.ny_-1; ++y)
                {
                    for(std::size_t x = 1; x != dst.nx_-1; ++x)
                    {
                        dst(x, y, z) = src(x, y, z);
                    }
                }
            }
        }
    };

    template <>
    struct stencils<STENCIL_2D5PT>
    {
        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src)
        {
            for(std::size_t z = 1; z != dst.nz_-1; ++z)
            {
                for(std::size_t y = 1; y != dst.ny_-1; ++y)
                {
                    for(std::size_t x = 1; x != dst.nx_-1; ++x)
                    {
                        dst(x, y, z)
                            = (
                                src(x-1, y, z)
                              + src(x, y-1, z)
                              + src(x, y, z)
                              + src(x+1, y, z)
                              + src(x, y+1, z)
                            )/ 5.0;
                    }
                }
            }
        }
    };

    template <>
    struct stencils<STENCIL_2D9PT>
    {
        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src)
        {
            for(std::size_t z = 1; z != dst.nz_-1; ++z)
            {
                for(std::size_t y = 1; y != dst.ny_-1; ++y)
                {
                    for(std::size_t x = 1; x != dst.nx_-1; ++x)
                    {
                        dst(x, y, z)
                            = (
                                src(x-1, y-1, z)
                              + src(x-1, y, z)
                              + src(x-1, y+1, z)
                              + src(x, y-1, z)
                              + src(x, y, z)
                              + src(x, y+1, z)
                              + src(x+1, y-1, z)
                              + src(x+1, y, z)
                              + src(x+1, y+1, z)
                            )/ 9.0;
                    }
                }
            }
        }
    };

    template <>
    struct stencils<STENCIL_3D7PT>
    {
        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src)
        {
            for(std::size_t z = 1; z != dst.nz_-1; ++z)
            {
                for(std::size_t y = 1; y != dst.ny_-1; ++y)
                {
                    for(std::size_t x = 1; x != dst.nx_-1; ++x)
                    {
                        dst(x, y, z)
                            = (
                                src(x, y, z-1)
                              + src(x-1, y, z)
                              + src(x, y-1, z)
                              + src(x, y, z)
                              + src(x+1, y, z)
                              + src(x, y+1, z)
                              + src(x, y, z+1)
                            )/ 7.0;

                    }
                }
            }
        }
    };

    template <>
    struct stencils<STENCIL_3D27PT>
    {
        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src)
        {
            for(std::size_t z = 1; z != dst.nz_-1; ++z)
            {
                for(std::size_t y = 1; y != dst.ny_-1; ++y)
                {
                    for(std::size_t x = 1; x != dst.nx_-1; ++x)
                    {
                        dst(x, y, z)
                            = (
                                src(x-1, y-1, z-1)
                              + src(x-1, y-1, z)
                              + src(x-1, y-1, z+1)
                              + src(x-1, y, z-1)
                              + src(x-1, y, z)
                              + src(x-1, y, z+1)
                              + src(x-1, y+1, z-1)
                              + src(x-1, y+1, z)
                              + src(x-1, y+1, z+1)
                              + src(x, y-1, z-1)
                              + src(x, y-1, z)
                              + src(x, y-1, z+1)
                              + src(x, y, z-1)
                              + src(x, y, z)
                              + src(x, y, z+1)
                              + src(x, y+1, z-1)
                              + src(x, y+1, z)
                              + src(x, y+1, z+1)
                              + src(x+1, y-1, z-1)
                              + src(x+1, y-1, z)
                              + src(x+1, y-1, z+1)
                              + src(x+1, y, z-1)
                              + src(x+1, y, z)
                              + src(x+1, y, z+1)
                              + src(x+1, y+1, z-1)
                              + src(x+1, y+1, z)
                              + src(x+1, y+1, z+1)
                            )/ 27.0;

                    }
                }
            }
        }
    };
}

#endif
