//  Copyright (c) 1998-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_WRAPPER_HEAP_BASE_HPP
#define HPX_UTIL_WRAPPER_HEAP_BASE_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/util/generate_unique_ids.hpp>

#include <cstddef>

namespace hpx { namespace util
{
    struct wrapper_heap_base
    {
        virtual ~wrapper_heap_base() {}

        virtual bool alloc(void** result, std::size_t count = 1) = 0;
        virtual bool did_alloc (void *p) const = 0;
        virtual void free(void *p, std::size_t count = 1) = 0;

        virtual naming::gid_type get_gid(util::unique_id_ranges& ids, void* p,
            components::component_type type) = 0;

        virtual std::size_t heap_count() const = 0;
        virtual std::size_t size() const = 0;
        virtual std::size_t free_size() const = 0;
    };
}}

#endif
