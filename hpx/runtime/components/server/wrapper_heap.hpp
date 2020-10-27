//  Copyright (c) 1998-2019 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/wrapper_heap_base.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

#define HPX_DEBUG_WRAPPER_HEAP 0

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    namespace one_size_heap_allocators
    {
        ///////////////////////////////////////////////////////////////////////
        // TODO: this interface should conform to the Boost.Pool allocator
        // interface, to maximize code reuse and consistency - wash.
        //
        // simple allocator which gets the memory from the default malloc,
        // but which does not reallocate the heap (it doesn't grow)
        struct fixed_mallocator
        {
            static void* alloc(std::size_t size)
            {
                return alloc_.allocate(size);
            }
            static void free(void* p, std::size_t count)
            {
                alloc_.deallocate(static_cast<char*>(p), count);
            }
            static void* realloc(std::size_t &, void *)
            {
                // normally this should return ::realloc(p, size), but we are
                // not interested in growing the allocated heaps, so we just
                // return nullptr
                return nullptr;
            }

            HPX_EXPORT static util::internal_allocator<char> alloc_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT wrapper_heap : public util::wrapper_heap_base
    {
    public:
        HPX_NON_COPYABLE(wrapper_heap);

    public:
        typedef one_size_heap_allocators::fixed_mallocator allocator_type;
        typedef hpx::lcos::local::spinlock mutex_type;
        typedef std::unique_lock<mutex_type> scoped_lock;
        typedef wrapper_heap_base::heap_parameters heap_parameters;

#if HPX_DEBUG_WRAPPER_HEAP != 0
        enum guard_value
        {
            initial_value = 0xcc,           // memory has been initialized
            freed_value = 0xdd,             // memory has been freed
        };
#endif

    public:
        explicit wrapper_heap(char const* class_name, std::size_t count,
            heap_parameters parameters);

        wrapper_heap();
        ~wrapper_heap() override;

        std::size_t size() const override;
        std::size_t free_size() const override;

        bool is_empty() const;
        bool has_allocatable_slots() const;

        bool alloc(void** result, std::size_t count = 1) override;
        void free(void *p, std::size_t count = 1) override;
        bool did_alloc (void *p) const override;

        // Get the global id of the managed_component instance given by the
        // parameter p.
        //
        // The pointer given by the parameter p must have been allocated by
        // this instance of a wrapper_heap
        naming::gid_type get_gid(util::unique_id_ranges& ids, void* p,
            components::component_type type) override;

        void set_gid(naming::gid_type const& g);

    protected:
        bool test_release(scoped_lock& lk);
        bool ensure_pool(std::size_t count);

        bool init_pool();
        void tidy();

    protected:
        char* pool_;
        char* first_free_;
        heap_parameters const parameters_;
        std::size_t free_size_;

        // these values are used for AGAS registration of all elements of this
        // managed_component heap
        naming::gid_type base_gid_;

        mutable mutex_type mtx_;

    public:
        std::string const class_name_;
#if defined(HPX_DEBUG)
        std::size_t alloc_count_;
        std::size_t free_count_;
        std::size_t heap_count_;
#endif

        // make sure the ABI of this is stable across configurations
#if defined(HPX_DEBUG)
        std::size_t heap_count() const override { return heap_count_; }
#else
        std::size_t heap_count() const override { return 0; }
#endif

    private:
        util::itt::heap_function heap_alloc_function_;
        util::itt::heap_function heap_free_function_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // heap using malloc and friends
    template <typename T>
    class fixed_wrapper_heap : public wrapper_heap
    {
    private:
        typedef wrapper_heap base_type;
        typedef base_type::heap_parameters heap_parameters;

    public:
        typedef T value_type;

        explicit fixed_wrapper_heap(char const* class_name,
                std::size_t count, heap_parameters parameters)
          : base_type(class_name, count, parameters)
        {}
    };
}}} // namespace hpx::components::detail

#include <hpx/config/warnings_suffix.hpp>

