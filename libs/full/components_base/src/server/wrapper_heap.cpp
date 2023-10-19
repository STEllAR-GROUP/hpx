//  Copyright (c) 1998-2023 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/server/wrapper_heap.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/unlock_guard.hpp>

#include <cstddef>
#include <cstdint>
#if HPX_DEBUG_WRAPPER_HEAP != 0
#include <cstring>
#endif
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::components::detail {

    namespace one_size_heap_allocators {

        util::internal_allocator<char> fixed_mallocator::alloc_;
    }

#if HPX_DEBUG_WRAPPER_HEAP != 0
#define HPX_WRAPPER_HEAP_INITIALIZED_MEMORY 1

    namespace debug {

        ///////////////////////////////////////////////////////////////////////
        // Test memory area for being filled as expected
        inline bool test_fill_bytes(void* p, unsigned char c, std::size_t cnt)
        {
            unsigned char* uc = (unsigned char*) p;
            for (std::size_t i = 0; i < cnt; ++i)
            {
                if (*uc++ != c)
                {
                    return false;
                }
            }
            return true;
        }

        // Fill memory area
        inline void fill_bytes(void* p, unsigned char c, std::size_t cnt)
        {
            using namespace std;    // some systems have memset in namespace std
            memset(p, c, cnt);
        }
    }    // namespace debug
#else
#define HPX_WRAPPER_HEAP_INITIALIZED_MEMORY 0
#endif

    ///////////////////////////////////////////////////////////////////////////
    wrapper_heap::wrapper_heap(char const* class_name,
        [[maybe_unused]] std::size_t count, heap_parameters const& parameters)
      : pool_(nullptr)
      , parameters_(parameters)
      , first_free_(nullptr)
      , free_size_(0)
      , base_gid_(naming::invalid_gid)
      , class_name_(class_name)
#if defined(HPX_DEBUG)
      , alloc_count_(0)
      , free_count_(0)
      , heap_count_(count)
#endif
      , heap_alloc_function_("wrapper_heap::alloc", class_name)
      , heap_free_function_("wrapper_heap::free", class_name)
    {
        [[maybe_unused]] util::itt::heap_internal_access hia;

        if (!init_pool())
        {
            throw std::bad_alloc();
        }

        // use the pool's base address as the first gid, this will also
        // allow for the ids to be locally resolvable
        base_gid_ = naming::replace_locality_id(
            naming::replace_component_type(naming::gid_type(pool_), 0),
            agas::get_locality_id());

        naming::detail::set_credit_for_gid(
            base_gid_, static_cast<std::int64_t>(HPX_GLOBALCREDIT_INITIAL));
    }

    wrapper_heap::wrapper_heap()
      : pool_(nullptr)
      , parameters_({0, 0, 0})
      , first_free_(nullptr)
      , free_size_(0)
      , base_gid_(naming::invalid_gid)
#if defined(HPX_DEBUG)
      , alloc_count_(0)
      , free_count_(0)
      , heap_count_(0)
#endif
      , heap_alloc_function_("wrapper_heap::alloc", "<unknown>")
      , heap_free_function_("wrapper_heap::free", "<unknown>")
    {
        HPX_ASSERT(false);    // shouldn't ever be called
    }

    wrapper_heap::~wrapper_heap()
    {
        [[maybe_unused]] util::itt::heap_internal_access hia;

        tidy();
    }

    std::size_t wrapper_heap::size() const
    {
        [[maybe_unused]] util::itt::heap_internal_access hia;

        return parameters_.capacity - free_size_;
    }

    std::size_t wrapper_heap::free_size() const
    {
        [[maybe_unused]] util::itt::heap_internal_access hia;

        return free_size_;
    }

    bool wrapper_heap::is_empty() const
    {
        [[maybe_unused]] util::itt::heap_internal_access hia;

        return nullptr == pool_;
    }

    bool wrapper_heap::has_allocatable_slots() const
    {
        [[maybe_unused]] util::itt::heap_internal_access hia;

        std::size_t const num_bytes =
            parameters_.capacity * parameters_.element_size;
        return first_free_ < pool_ + num_bytes;
    }

    bool wrapper_heap::alloc(void** result, std::size_t count)
    {
        util::itt::heap_allocate heap_allocate(heap_alloc_function_, result,
            count * parameters_.element_size,
            HPX_WRAPPER_HEAP_INITIALIZED_MEMORY);

        if (nullptr == pool_)
        {
            return false;
        }

        std::size_t const num_bytes = count * parameters_.element_size;
        std::size_t const total_num_bytes =
            parameters_.capacity * parameters_.element_size;

        if (first_free_ + num_bytes > pool_ + total_num_bytes)
        {
            return false;
        }

#if defined(HPX_DEBUG)
        alloc_count_ += count;
#endif

        char* p = first_free_.fetch_add(
            static_cast<std::ptrdiff_t>(count * parameters_.element_size),
            std::memory_order_relaxed);

        if (p + num_bytes > pool_ + total_num_bytes)
        {
            return false;
        }

#if HPX_DEBUG_WRAPPER_HEAP != 0
        // init memory blocks
        debug::fill_bytes(p, initial_value, count * parameters_.element_size);
#endif

        *result = p;
        return true;
    }

    void wrapper_heap::free([[maybe_unused]] void* p, std::size_t count)
    {
        util::itt::heap_free heap_free(heap_free_function_, p);

#if HPX_DEBUG_WRAPPER_HEAP != 0
        HPX_ASSERT(did_alloc(p));
        char* p1 = p;
        std::size_t const total_num_bytes =
            parameters_.capacity * parameters_.element_size;
        std::size_t const num_bytes = count * parameters_.element_size;

        HPX_ASSERT(nullptr != pool_ && p1 >= pool_);
        HPX_ASSERT(
            nullptr != pool_ && p1 + num_bytes <= pool_ + total_num_bytes);
        HPX_ASSERT(first_free_ == nullptr || p1 != first_free_);
        HPX_ASSERT(free_size_ + count <= parameters_.capacity);
        // make sure this has not been freed yet
        HPX_ASSERT(!debug::test_fill_bytes(p1, freed_value, num_bytes));

        // give memory back to pool
        debug::fill_bytes(p1, freed_value, num_bytes);
#endif

#if defined(HPX_DEBUG)
        free_count_ += count;
#endif
        size_t const current_free_size =
            free_size_.fetch_add(count, std::memory_order_relaxed) + count;

        // release the pool if this one was the last allocated item
        if (current_free_size == parameters_.capacity)
            free_pool();
    }

    bool wrapper_heap::did_alloc(void* p) const
    {
        // no lock is necessary here as all involved variables are immutable
        [[maybe_unused]] util::itt::heap_internal_access hia;

        if (nullptr == pool_)
            return false;
        if (nullptr == p)
            return false;

        std::size_t const total_num_bytes =
            parameters_.capacity * parameters_.element_size;
        return p >= pool_ && static_cast<char*>(p) < pool_ + total_num_bytes;
    }

    naming::gid_type wrapper_heap::get_gid(
        void* p, components::component_type type)
    {
        [[maybe_unused]] util::itt::heap_internal_access hia;

        HPX_ASSERT(did_alloc(p));
        HPX_ASSERT(base_gid_);

        naming::gid_type result = base_gid_;
        if (type)
        {
            result = naming::replace_component_type(result, type);
        }
        result.set_lsb(p);

        // We have to assume this credit was split as otherwise the gid returned
        // at this point will control the lifetime of the component.
        naming::detail::set_credit_split_mask_for_gid(result);

        return result;
    }

    void wrapper_heap::set_gid(naming::gid_type const& g)
    {
        [[maybe_unused]] util::itt::heap_internal_access hia;

        std::unique_lock l(mtx_);
        base_gid_ = g;
    }

    bool wrapper_heap::free_pool()
    {
        HPX_ASSERT(pool_);
        HPX_ASSERT(first_free_ ==
            pool_ + parameters_.capacity * parameters_.element_size);

        // unbind in AGAS service
        if (base_gid_)
        {
            naming::gid_type base_gid = naming::invalid_gid;
            {
                std::unique_lock l(mtx_);
                if (base_gid_)
                {
                    base_gid = base_gid_;
                    base_gid_ = naming::invalid_gid;
                }
            }
            if (base_gid)
                agas::unbind_range_local(base_gid, parameters_.capacity);
        }

        tidy();
        return true;
    }

    bool wrapper_heap::init_pool()
    {
        HPX_ASSERT(first_free_ == nullptr);

        std::size_t const total_num_bytes =
            parameters_.capacity * parameters_.element_size;
        pool_ = static_cast<char*>(allocator_type::alloc(total_num_bytes));
        if (nullptr == pool_)
        {
            return false;
        }

        if (reinterpret_cast<std::size_t>(pool_) %
                parameters_.element_alignment ==
            0)
        {
            first_free_.store(pool_, std::memory_order_relaxed);
        }
        else
        {
            first_free_.store(pool_ + parameters_.element_alignment,
                std::memory_order_relaxed);
        }

        free_size_.store(parameters_.capacity, std::memory_order_release);

        LOSH_(info).format("wrapper_heap ({}): init_pool ({}) size: {}.",
            !class_name_.empty() ? class_name_.c_str() : "<Unknown>",
            static_cast<void*>(pool_), total_num_bytes);

        return true;
    }

    void wrapper_heap::tidy()
    {
        if (pool_ != nullptr)
        {
            LOSH_(debug)
                    .format("wrapper_heap ({})",
                        !class_name_.empty() ? class_name_.c_str() :
                                               "<Unknown>")
#if defined(HPX_DEBUG)
                    .format(": releasing heap: alloc count: {}, free "
                            "count: {}",
                        alloc_count_, free_count_)
#endif
                << ".";

            if (wrapper_heap::size() > 0
#if defined(HPX_DEBUG)
                || alloc_count_ != free_count_
#endif
            )
            {
                LOSH_(warning).format("wrapper_heap ({}): releasing heap ({}) "
                                      "with {} allocated object(s)!",
                    !class_name_.empty() ? class_name_.c_str() : "<Unknown>",
                    static_cast<void*>(pool_), wrapper_heap::size());
            }

            std::size_t const total_num_bytes =
                parameters_.capacity * parameters_.element_size;
            allocator_type::free(pool_, total_num_bytes);
            pool_ = nullptr;

            first_free_.store(nullptr, std::memory_order_relaxed);
            free_size_.store(0, std::memory_order_release);
        }
    }
}    // namespace hpx::components::detail
