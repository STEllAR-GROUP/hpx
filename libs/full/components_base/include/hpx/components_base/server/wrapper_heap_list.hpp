//  Copyright (c) 1998-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/server/one_size_heap_list.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/thread_support/unlock_guard.hpp>

#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::components::detail {

    ///////////////////////////////////////////////////////////////////////////
    // list of managed_component heaps
    template <typename Heap>
    class wrapper_heap_list : public util::one_size_heap_list
    {
        using base_type = util::one_size_heap_list;
        using value_type = typename Heap::value_type;

        using storage_type = std::aligned_storage_t<sizeof(value_type),
            std::alignment_of<value_type>::value>;

        enum
        {
            // default initial number of elements
            heap_capacity = 0xFFF,
            // Alignment of one element
            heap_element_alignment = std::alignment_of_v<value_type>,    //-V103
            // size of one element in the heap
            heap_element_size = sizeof(storage_type)
        };

    public:
        wrapper_heap_list()
          : base_type(
                get_component_type_name(
                    get_component_type<typename value_type::wrapped_type>()),
                base_type::heap_parameters{
                    heap_capacity, heap_element_alignment, heap_element_size},
                (Heap*) nullptr)
          , type_(get_component_type<typename value_type::wrapped_type>())
        {
        }

        naming::gid_type get_gid(void* p)
        {
            std::unique_lock guard(this->mtx_);

            using iterator = typename base_type::const_iterator;

            iterator end = this->heap_list_.end();
            for (iterator it = this->heap_list_.begin(); it != end; ++it)
            {
                if ((*it)->did_alloc(p))
                {
                    unlock_guard ul(guard);
                    return (*it)->get_gid(p, type_);
                }
            }
            return naming::invalid_gid;
        }

    private:
        components::component_type type_;
    };
}    // namespace hpx::components::detail
