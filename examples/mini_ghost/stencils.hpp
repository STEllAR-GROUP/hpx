//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_STENCILS_HPP
#define HPX_EXAMPLES_MINI_GHOST_STENCILS_HPP

#include <examples/mini_ghost/grid.hpp>

namespace mini_ghost {

    ///////////////////////////////////////////////////////////////////////////
    typedef std::pair<std::size_t, std::size_t> range_type;

    template <typename Real>
    struct stencil_call
    {
        typedef void (*type)(grid<Real>&, grid<Real> const&,
            range_type, range_type, range_type);
    };

    static const std::size_t STENCIL_NONE   = 20;
    static const std::size_t STENCIL_2D5PT  = 21;
    static const std::size_t STENCIL_2D9PT  = 22;
    static const std::size_t STENCIL_3D7PT  = 23;
    static const std::size_t STENCIL_3D27PT = 24;

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t Stencil>
    struct stencils;

    template <>
    struct stencils<STENCIL_NONE>
    {
        static const std::size_t num_adds = 0;

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
        }
    };

    template <>
    struct stencils<STENCIL_2D5PT>
    {
        static const std::size_t num_adds = 4;

        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src,
            range_type x_range, range_type y_range, range_type z_range)
        {
            Real const divisor = Real(1.0)/Real(5.0);
            std::size_t i = 0;
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
                        ++i;
                    }
                }
            }
        }
    };

    template <>
    struct stencils<STENCIL_2D9PT>
    {
        static const std::size_t num_adds = 7;

        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src,
            range_type x_range, range_type y_range, range_type z_range)
        {
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
        }
    };

    template <>
    struct stencils<STENCIL_3D7PT>
    {
        static const std::size_t num_adds = 6;

        template <typename Real>
        static void call(grid<Real> & dst, grid<Real> const & src,
            range_type x_range, range_type y_range, range_type z_range)
        {
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
        }
    };

    template <>
    struct stencils<STENCIL_3D27PT>
    {
        static const std::size_t num_adds = 26;

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
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Real>
    typename stencil_call<Real>::type
    get_stencil_call_op(std::size_t stencil)
    {
        switch (stencil)
        {
            case STENCIL_NONE:
                return &stencils<STENCIL_NONE>::template call<Real>;
            case STENCIL_2D5PT:
                return &stencils<STENCIL_2D5PT>::template call<Real>;
            case STENCIL_2D9PT:
                return &stencils<STENCIL_2D9PT>::template call<Real>;
            case STENCIL_3D7PT:
                return &stencils<STENCIL_3D7PT>::template call<Real>;
            case STENCIL_3D27PT:
                return &stencils<STENCIL_3D27PT>::template call<Real>;

            default:
                std::cerr << "Unknown stencil\n";
                hpx::terminate();
                break;
        }
        return 0;
    }
}

#endif
