//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_STENCILS_HPP
#define HPX_EXAMPLES_MINI_GHOST_STENCILS_HPP

#include <examples/mini_ghost/grid.hpp>

namespace mini_ghost {

    typedef std::pair<std::size_t, std::size_t> range_type;

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
        static void call(grid<Real> & dst, grid<Real> const & src,
            range_type x_range, range_type y_range, range_type z_range)
        {
            hpx::util::high_resolution_timer timer;
            for(std::size_t z = z_range.first; z != z_range.second; ++z)
            {
                for(std::size_t y = y_range.first; y != y_range.second; ++y)
                {
                    for(std::size_t x = x_range.first; x != x_range.second; ++x)
                    {
                        dst(x, y, z) = src(x, y, z);
                    }
                }
            }
            profiling::data().time_stencil(timer.elapsed());
        }
    };

    template <>
    struct stencils<STENCIL_2D5PT>
    {
        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src,
            range_type x_range, range_type y_range, range_type z_range)
        {
            hpx::util::high_resolution_timer timer;
            Real const divisor = Real(1.0)/Real(5.0);
            for(std::size_t z = z_range.first; z != z_range.second; ++z)
            {
                for(std::size_t y = y_range.first; y != y_range.second; ++y)
                {
                    for(std::size_t x = x_range.first; x != x_range.second; ++x)
                    {
                        dst(x, y, z)
                            = (
                                src(x-1, y, z)
                              + src(x, y-1, z)
                              + src(x, y, z)
                              + src(x+1, y, z)
                              + src(x, y+1, z)
                            ) * divisor;
                    }
                }
            }
            std::size_t num_pts = 1;
            num_pts *= x_range.second - x_range.first;
            num_pts *= y_range.second - y_range.first;
            num_pts *= z_range.second - z_range.first;
            profiling::data().num_adds(4*num_pts);
            profiling::data().num_divides(num_pts);
            profiling::data().time_stencil(timer.elapsed());
        }
    };

    template <>
    struct stencils<STENCIL_2D9PT>
    {
        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src,
            range_type x_range, range_type y_range, range_type z_range)
        {
            hpx::util::high_resolution_timer timer;
            Real const divisor = Real(1.0)/Real(5.0);
            for(std::size_t z = z_range.first; z != z_range.second; ++z)
            {
                for(std::size_t y = y_range.first; y != y_range.second; ++y)
                {
                    for(std::size_t x = x_range.first; x != x_range.second; ++x)
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
                            ) * divisor;
                    }
                }
            }
            std::size_t num_pts = 1;
            num_pts *= x_range.second - x_range.first;
            num_pts *= y_range.second - y_range.first;
            num_pts *= z_range.second - z_range.first;
            profiling::data().num_adds(8*num_pts);
            profiling::data().num_divides(num_pts);
            profiling::data().time_stencil(timer.elapsed());
        }
    };

    template <>
    struct stencils<STENCIL_3D7PT>
    {
        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src,
            range_type x_range, range_type y_range, range_type z_range)
        {
            hpx::util::high_resolution_timer timer;
            Real const divisor = Real(1.0)/Real(5.0);
            for(std::size_t z = z_range.first; z != z_range.second; ++z)
            {
                for(std::size_t y = y_range.first; y != y_range.second; ++y)
                {
                    for(std::size_t x = x_range.first; x != x_range.second; ++x)
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
                            ) * divisor;

                    }
                }
            }
            std::size_t num_pts = 1;
            num_pts *= x_range.second - x_range.first;
            num_pts *= y_range.second - y_range.first;
            num_pts *= z_range.second - z_range.first;
            profiling::data().num_adds(6*num_pts);
            profiling::data().num_divides(num_pts);
            profiling::data().time_stencil(timer.elapsed());
        }
    };

    template <>
    struct stencils<STENCIL_3D27PT>
    {
        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src,
            range_type x_range, range_type y_range, range_type z_range)
        {
            hpx::util::high_resolution_timer timer;
            Real const divisor = Real(1.0)/Real(27.0);
            for(std::size_t z = z_range.first; z != z_range.second; ++z)
            {
                for(std::size_t y = y_range.first; y != y_range.second; ++y)
                {
                    for(std::size_t x = x_range.first; x != x_range.second; ++x)
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
                            ) * divisor;

                    }
                }
            }
            std::size_t num_pts = 1;
            num_pts *= x_range.second - x_range.first;
            num_pts *= y_range.second - y_range.first;
            num_pts *= z_range.second - z_range.first;
            profiling::data().num_adds(26*num_pts);
            profiling::data().num_divides(num_pts);
            profiling::data().time_stencil(timer.elapsed());
        }
    };
}

#endif
