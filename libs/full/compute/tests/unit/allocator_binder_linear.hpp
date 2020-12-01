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
// Example of an allocator binder for 1D arrays.
// This simply binds pages according to a linear sequence
// with increasing numa domain number modulo the total domains available
// ------------------------------------------------------------------------
template <typename T>
struct linear_numa_binder : hpx::compute::host::numa_binding_helper<T>
{
    explicit linear_numa_binder(std::size_t num_pages)
      : hpx::compute::host::numa_binding_helper<T>()
    {
        std::size_t const cache_line_size = hpx::threads::get_cache_line_size();
        std::size_t const page_size = hpx::threads::get_memory_page_size();
        std::size_t const alignment = (std::max)(page_size, cache_line_size);
        elements_page_ = (alignment / sizeof(T));
        N_ = num_pages * elements_page_;
    }

    // return the domain that a given page should be bound to
    virtual std::size_t operator()(const T* const base_ptr,
        const T* const page_ptr, std::size_t const /* pagesize */,
        std::size_t const domains) const override
    {
        std::intptr_t offset = page_ptr - base_ptr;
        std::size_t index = (offset / elements_page_);
        return index % domains;
    }

    // This is for debug purposes
    virtual std::string description() const override
    {
        std::ostringstream temp;
        temp << "Linear " << std::dec << " N " << N_ << " elements_page_ "
             << elements_page_;
        return temp.str();
    }

    // Total memory consumption in bytes
    virtual std::size_t memory_bytes() const override
    {
        return sizeof(T) * N_;
    }

    // Using how many dimensions should this data be displayed
    virtual std::size_t array_rank() const override
    {
        return 1;
    }

    // The number of elements along dimension x=0,y=1,z=2,...
    virtual std::size_t array_size(std::size_t axis) const override
    {
        if (axis == 0)
            return N_;
        return 1;
    }

    // When counting along elements in a given dimension,
    // how large a step should be taken in units of elements.
    // This should include padding along an axis
    virtual std::size_t memory_step(std::size_t) const override
    {
        return 1;
    }

    // When displaying the data, what step size should be used
    virtual std::size_t display_step(std::size_t) const override
    {
        return elements_page_;
    }
    //
    std::size_t N_;
    std::size_t elements_page_;
};
