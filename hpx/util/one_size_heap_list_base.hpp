//  Copyright (c) 1998-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ONE_SIZE_HEAP_LIST_BASE_OCT_12_2013_0413PM)
#define HPX_UTIL_ONE_SIZE_HEAP_LIST_BASE_OCT_12_2013_0413PM

#include <hpx/runtime/naming/name.hpp>
#include <cstddef>

namespace hpx { namespace util
{
    struct one_size_heap_list_base
    {
        virtual ~one_size_heap_list_base() {}

        virtual void* alloc(std::size_t count = 1) = 0;
        virtual bool did_alloc(void* p) const = 0;
        virtual void free(void* p, std::size_t count = 1) = 0;

        virtual naming::gid_type get_gid(void* p) = 0;
    };
}}

#endif

