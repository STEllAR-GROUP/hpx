//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/compute/host/numa_binding_allocator.hpp>
//
#include <cstddef>
#include <sstream>
#include <string>

// ------------------------------------------------------------------------
// Example of an allocator binder for 2D matrices that are tiled.
// Data is stored in column major order and columns are padded
// to align with a page boundary.
// Note: Column major ordering flips offsets (caution)
// ------------------------------------------------------------------------
template <typename T>
struct matrix_numa_binder : hpx::compute::host::numa_binding_helper<T>
{
    matrix_numa_binder(std::size_t Ncols, std::size_t Nrows, std::size_t Ntile,
        std::size_t Ntiles_per_domain, std::size_t Ncolprocs = 1,
        std::size_t Nrowprocs = 1)
      : hpx::compute::host::numa_binding_helper<T>()
      , cols_(Ncols)
      , rows_(Nrows)
      , tile_size_(Ntile)
      , tiles_per_domain_(Ntiles_per_domain)
      , colprocs_(Ncolprocs)
      , rowprocs_(Nrowprocs)
    {
        int const cache_line_size = hpx::threads::get_cache_line_size();
        int const page_size = hpx::threads::get_memory_page_size();
        int const alignment = (std::max)(page_size, cache_line_size);
        int const elems_align = (alignment / sizeof(T));
        rows_page_ = elems_align;
        leading_dim_ =
            elems_align * ((rows_ * sizeof(T) + alignment - 1) / alignment);
        // @TODO : put tiles_per_domain_ back in
    }

    // return the domain that a given page should be bound to
    virtual std::size_t operator()(const T* const base_ptr,
        const T* const page_ptr, std::size_t const /* pagesize */,
        std::size_t const domains) const override
    {
        std::intptr_t offset = page_ptr - base_ptr;
        std::size_t col = (offset / leading_dim_);
        std::size_t row = (offset % leading_dim_);
        std::size_t index = col / rows_page_;
        index += (row / rows_page_);
        return index % domains;
    }

    // for debug purposes
    virtual std::string description() const override
    {
        std::ostringstream temp;
        temp << "Matrix " << std::dec << " columns " << cols_ << " rows "
             << rows_ << " tile_size " << tile_size_ << " leading_dim "
             << leading_dim_ << " tiles_per_domain " << tiles_per_domain_
             << " colprocs " << colprocs_ << " rowprocs " << rowprocs_
             << " rows_page " << rows_page_ << " domains(col) "
             << rows_ / rows_page_ << " display_step (" << display_step(0)
             << ',' << display_step(1) << ')';
        return temp.str();
    }

    // Total memory consumption in bytes
    virtual std::size_t memory_bytes() const override
    {
        return sizeof(T) * (cols_ / colprocs_) * (leading_dim_ / rowprocs_);
    }

    // Using how many dimensions should this data be displayed
    virtual std::size_t array_rank() const override
    {
        return 2;
    }

    // The number of elements along dimension x=0,y=1,z=2,...
    virtual std::size_t array_size(std::size_t axis) const override
    {
        if (axis == 0)
            return cols_ / colprocs_;
        return rows_ / rowprocs_;
    }

    // When counting along elements in a given dimension,
    // how large a step should be taken in units of elements.
    // This should include padding along an axis
    virtual std::size_t memory_step(std::size_t axis) const override
    {
        if (axis == 0)
            return leading_dim_ / rowprocs_;
        return 1;
    }

    // When displaying the data, what step size should be used
    virtual std::size_t display_step(std::size_t axis) const override
    {
        if (axis == 0)
            return rows_page_;
        return rows_page_;
    }
    //
    std::size_t cols_;
    std::size_t rows_;
    std::size_t tile_size_;
    std::size_t leading_dim_;
    std::size_t tiles_per_domain_;
    std::size_t colprocs_;
    std::size_t rowprocs_;
    std::size_t rows_page_;
};
