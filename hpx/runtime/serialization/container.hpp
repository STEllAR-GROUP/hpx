//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_CONTAINER_HPP
#define HPX_SERIALIZATION_CONTAINER_HPP

#include <hpx/runtime/serialization/binary_filter.hpp>
#include <hpx/util/assert.hpp>

namespace hpx { namespace serialization {
    struct container
    {
        virtual ~container() {}

        virtual void set_filter(util::binary_filter* filter) = 0;
        virtual void save_binary(void const* address, std::size_t count)
        {
            HPX_ASSERT(false
                && "hpx::serialization::container::save_binary not implemented");
        }
        virtual void save_binary_chunk(void const* address, std::size_t count)
        {
            HPX_ASSERT(false
                && "hpx::serialization::container::save_binary_chunk not implemented");
        }
        virtual void load_binary(void * address, std::size_t count)
        {
            HPX_ASSERT(false
                && "hpx::serialization::container::save_binary not implemented");
        }
        virtual void load_binary_chunk(void * address, std::size_t count)
        {
            HPX_ASSERT(false
                && "hpx::serialization::container::save_binary_chunk not implemented");
        }
    };
}}

#endif
