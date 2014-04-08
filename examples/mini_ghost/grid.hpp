//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GRID_HPP
#define HPX_EXAMPLES_MINI_GRID_HPP

#include <vector>

namespace mini_ghost {

    static const std::size_t NORTH   = 0;
    static const std::size_t SOUTH   = 1;
    static const std::size_t EAST    = 2;
    static const std::size_t WEST    = 3;
    static const std::size_t BACK    = 4;
    static const std::size_t FRONT   = 5;

    template <typename T>
    struct grid
    {
        typedef T value_type;

        grid()
          : nx_(0)
          , ny_(0)
          , nz_(0)
        {}

        grid(std::size_t nx, std::size_t ny, std::size_t nz = 1)
          : nx_(nx)
          , ny_(ny)
          , nz_(nz)
          , data_(nx*ny*nz)
        {}

        void resize(std::size_t nx, std::size_t ny, std::size_t nz = 1)
        {
            nx_ = nx;
            ny_ = ny;
            nz_ = nz;
            data_.resize(nx*ny*nz);
        }

        T & operator()(std::size_t x, std::size_t y, std::size_t z = 0)
        {
            return data_[x + y * nx_ + z * nx_ * ny_];
        }

        T const & operator()(std::size_t x, std::size_t y, std::size_t z = 0) const
        {
            return data_[x + y * nx_ + z * nx_ * ny_];
        }

        std::size_t nx_;
        std::size_t ny_;
        std::size_t nz_;
        std::vector<T> data_;
    };
}

#endif
