//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_UNPACK_BUFFER_HPP
#define HPX_EXAMPLES_MINI_GHOST_UNPACK_BUFFER_HPP

#include <examples/mini_ghost/grid.hpp>
#include <hpx/util/serialize_buffer.hpp>

namespace mini_ghost {
    template <std::size_t Zone>
    struct unpack_buffer;

    template <>
    struct unpack_buffer<NORTH>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> & g, BufferType buffer)
        {
            typename BufferType::value_type * src = buffer.data();
            HPX_ASSERT(buffer.size() == g.nx_ * g.nz_);
            for(std::size_t z = 0; z != g.nz_; ++z)
            {
                for(std::size_t x = 0; x != g.nx_; ++x)
                {
                    g(x, g.ny_-1, z) = *src;
                    ++src;
                }
            }
        }
    };

    template <>
    struct unpack_buffer<SOUTH>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> & g, BufferType buffer)
        {
            typename BufferType::value_type * src = buffer.data();
            HPX_ASSERT(buffer.size() == g.nx_ * g.nz_);
            for(std::size_t z = 0; z != g.nz_; ++z)
            {
                for(std::size_t x = 0; x != g.nx_; ++x)
                {
                    g(x, 0, z) = *src;
                    ++src;
                }
            }
        }
    };

    template <>
    struct unpack_buffer<EAST>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> & g, BufferType buffer)
        {
            typename BufferType::value_type * src = buffer.data();
            HPX_ASSERT(buffer.size() == g.nz_ * g.ny_);
            for(std::size_t z = 0; z != g.nz_; ++z)
            {
                for(std::size_t y = 0; y != g.ny_; ++y)
                {
                    g(g.nx_-1, y, z) = *src;
                    ++src;
                }
            }
        }
    };

    template <>
    struct unpack_buffer<WEST>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> & g, BufferType buffer)
        {
            typename BufferType::value_type * src = buffer.data();
            HPX_ASSERT(buffer.size() == g.nz_ * g.ny_);
            for(std::size_t z = 0; z != g.nz_; ++z)
            {
                for(std::size_t y = 0; y != g.ny_; ++y)
                {
                    g(0, y, z) = *src;
                    ++src;
                }
            }
        }
    };

    template <>
    struct unpack_buffer<BACK>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> & g, BufferType buffer)
        {
            typename BufferType::value_type * src = buffer.data();
            HPX_ASSERT(buffer.size() == g.nx_ * g.ny_);
            for(std::size_t y = 0; y != g.ny_; ++y)
            {
                for(std::size_t x = 0; x != g.nx_; ++x)
                {
                    g(x, y, g.nz_-1) = *src;
                    ++src;
                }
            }
        }
    };

    template <>
    struct unpack_buffer<FRONT>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> & g, BufferType buffer)
        {
            typename BufferType::value_type * src = buffer.data();
            HPX_ASSERT(buffer.size() == g.nx_ * g.ny_);
            for(std::size_t y = 0; y != g.ny_; ++y)
            {
                for(std::size_t x = 0; x != g.nx_; ++x)
                {
                    g(x, y, 0) = *src;
                    ++src;
                }
            }
        }
    };
}

#endif
