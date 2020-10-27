//  Copyright (c) 1998-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/components_base/component_type.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/one_size_heap_list.hpp>

#include <iostream>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // list of managed_component heaps
    template <typename Heap>
    class wrapper_heap_list : public util::one_size_heap_list
    {
        typedef util::one_size_heap_list base_type;
        typedef typename Heap::value_type value_type;

        typedef typename std::aligned_storage<
                sizeof(value_type), std::alignment_of<value_type>::value
            >::type storage_type;

        enum
        {
            // default initial number of elements
            heap_capacity = 0xFFF,
            // Alignment of one element
            heap_element_alignment = std::alignment_of<value_type>::value,
            // size of one element in the heap
            heap_element_size = sizeof(storage_type)
        };

    public:
        wrapper_heap_list()
          : base_type(get_component_type_name(
                get_component_type<typename value_type::wrapped_type>()),
                base_type::heap_parameters{heap_capacity, heap_element_alignment,
                heap_element_size}, (Heap*) nullptr)
          , type_(get_component_type<typename value_type::wrapped_type>())
        {}

        ///
        naming::gid_type get_gid(void* p)
        {
            typename base_type::unique_lock_type guard(this->mtx_);

            typedef typename base_type::const_iterator iterator;
            iterator end = this->heap_list_.end();
            for (iterator it = this->heap_list_.begin(); it != end; ++it)
            {
                if ((*it)->did_alloc(p))
                {
                    util::unlock_guard<typename base_type::unique_lock_type> ul(guard);
                    return (*it)->get_gid(id_range_, p, type_);
                }
            }
            return naming::invalid_gid;
        }

        void set_range(
            naming::gid_type const& lower
          , naming::gid_type const& upper)
        {
            typename base_type::unique_lock_type guard(this->mtx_);
            id_range_.set_range(lower, upper);
        }

    private:
        util::unique_id_ranges id_range_;
        components::component_type type_;
    };

}}} // namespace hpx::components::detail

