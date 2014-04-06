//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_PACK_BUFFER_HPP
#define HPX_EXAMPLES_MINI_GHOST_PACK_BUFFER_HPP

#include <examples/mini_ghost/grid.hpp>
#include <hpx/util/serialize_buffer.hpp>

namespace mini_ghost {
    template <std::size_t Zone>
    struct pack_buffer;

    template <>
    struct pack_buffer<NORTH>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> const & g, BufferType & buffer)
        {
            Real * data = new Real[g.nx_ * g.nz_];
            buffer.take_buffer(data, g.nx_ * g.nz_);
            typename BufferType::value_type * src = buffer.data();
            for(std::size_t z = 0; z != g.nz_; ++z)
            {
                for(std::size_t x = 0; x != g.nx_; ++x)
                {
                    *src = g(x, g.ny_-2, z);
                    ++src;
                }
            }
        }
    };

    template <>
    struct pack_buffer<SOUTH>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> const & g, BufferType & buffer)
        {
            Real * data = new Real[g.nx_ * g.nz_];
            buffer.take_buffer(data, g.nx_ * g.nz_);
            typename BufferType::value_type * src = buffer.data();
            for(std::size_t z = 0; z != g.nz_; ++z)
            {
                for(std::size_t x = 0; x != g.nx_; ++x)
                {
                    *src = g(x, 1, z);
                    ++src;
                }
            }
        }
    };

    template <>
    struct pack_buffer<EAST>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> const & g, BufferType & buffer)
        {
            Real * data = new Real[g.ny_ * g.nz_];
            buffer.take_buffer(data, g.ny_ * g.nz_);
            typename BufferType::value_type * src = buffer.data();
            for(std::size_t z = 0; z != g.nz_; ++z)
            {
                for(std::size_t y = 0; y != g.ny_; ++y)
                {
                    *src = g(g.nx_-2, y, z);
                    ++src;
                }
            }
        }
    };

    template <>
    struct pack_buffer<WEST>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> const & g, BufferType & buffer)
        {
            Real * data = new Real[g.ny_ * g.nz_];
            buffer.take_buffer(data, g.ny_ * g.nz_);
            typename BufferType::value_type * src = buffer.data();
            for(std::size_t z = 0; z != g.nz_; ++z)
            {
                for(std::size_t y = 0; y != g.ny_; ++y)
                {
                    *src = g(1, y, z);
                    ++src;
                }
            }
        }
    };

    template <>
    struct pack_buffer<BACK>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> const & g, BufferType & buffer)
        {
            Real * data = new Real[g.nx_ * g.ny_];
            buffer.take_buffer(data, g.nx_ * g.ny_);
            typename BufferType::value_type * src = buffer.data();
            for(std::size_t y = 0; y != g.ny_; ++y)
            {
                for(std::size_t x = 0; x != g.nx_; ++x)
                {
                    *src = g(x, y, g.nz_-2);
                    ++src;
                }
            }
        }
    };

    template <>
    struct pack_buffer<FRONT>
    {
        template <typename Real, typename BufferType>
        static void call(grid<Real> const & g, BufferType & buffer)
        {
            Real * data = new Real[g.nx_ * g.ny_];
            buffer.take_buffer(data, g.nx_ * g.ny_);
            typename BufferType::value_type * src = buffer.data();
            for(std::size_t y = 0; y != g.ny_; ++y)
            {
                for(std::size_t x = 0; x != g.nx_; ++x)
                {
                    *src = g(x, y, 0);
                    ++src;
                }
            }
        }
    };
}

#endif
