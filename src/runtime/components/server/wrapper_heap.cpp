//  Copyright (c) 1998-2019 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/applier/applier.hpp>
#include <hpx/async_distributed/applier/bind_naming_wrappers.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/components/server/wrapper_heap.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/util/generate_unique_ids.hpp>

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
namespace hpx { namespace components { namespace detail
{
    namespace one_size_heap_allocators
    {
        util::internal_allocator<char> fixed_mallocator::alloc_;
    }

#if HPX_DEBUG_WRAPPER_HEAP != 0
#define HPX_WRAPPER_HEAP_INITIALIZED_MEMORY 1

    namespace debug
    {
        ///////////////////////////////////////////////////////////////////////
        // Test memory area for being filled as expected
        inline
        bool test_fill_bytes (void *p, unsigned char c, std::size_t cnt)
        {
            unsigned char* uc = (unsigned char*)p;
            for (std::size_t i = 0; i < cnt; ++i)
            {
                if (*uc++ != c)
                    return false;
            }
            return true;
        }

        // Fill memory area
        inline
        void fill_bytes (void *p, unsigned char c, std::size_t cnt)
        {
            using namespace std;    // some systems have memset in namespace std
            memset (p, c, cnt);
        }
    } // namespace debug
#else
#  define HPX_WRAPPER_HEAP_INITIALIZED_MEMORY 0
#endif

    ///////////////////////////////////////////////////////////////////////////
    wrapper_heap::wrapper_heap(char const* class_name
#if defined(HPX_DEBUG)
        , std::size_t count
#else
        , std::size_t
#endif
        , heap_parameters parameters)
      : pool_(nullptr)
      , first_free_(nullptr)
      , parameters_(parameters)
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
        util::itt::heap_internal_access hia;
        if (!init_pool())
            throw std::bad_alloc();
    }

    wrapper_heap::wrapper_heap()
      : pool_(nullptr)
      , first_free_(nullptr)
      , parameters_({0, 0, 0})
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
        HPX_ASSERT(false);      // shouldn't ever be called
    }

    wrapper_heap::~wrapper_heap()
    {
        util::itt::heap_internal_access hia; HPX_UNUSED(hia);
        tidy();
    }

    std::size_t wrapper_heap::size() const
    {
        util::itt::heap_internal_access hia; HPX_UNUSED(hia);
        return parameters_.capacity - free_size_;
    }

    std::size_t wrapper_heap::free_size() const
    {
        util::itt::heap_internal_access hia; HPX_UNUSED(hia);
        return free_size_;
    }

    bool wrapper_heap::is_empty() const
    {
        util::itt::heap_internal_access hia; HPX_UNUSED(hia);
        return nullptr == pool_;
    }

    bool wrapper_heap::has_allocatable_slots() const
    {
        util::itt::heap_internal_access hia; HPX_UNUSED(hia);
        std::size_t const num_bytes = parameters_.capacity * parameters_.element_size;
        return first_free_ < pool_ + num_bytes;
    }

    bool wrapper_heap::alloc(void** result, std::size_t count)
    {
        util::itt::heap_allocate heap_allocate(
            heap_alloc_function_, result, count * parameters_.element_size,
            HPX_WRAPPER_HEAP_INITIALIZED_MEMORY);

        scoped_lock l(mtx_);

        if (!ensure_pool(count))
            return false;

#if defined(HPX_DEBUG)
        alloc_count_ += count;
#endif

        void* p = first_free_;
        HPX_ASSERT(p != nullptr);

        first_free_ = first_free_ + count * parameters_.element_size;

        HPX_ASSERT(free_size_ >= count);
        free_size_ -= count;

#if HPX_DEBUG_WRAPPER_HEAP != 0
        // init memory blocks
        debug::fill_bytes(p, initial_value, count * parameters_.element_size);
#endif

        *result = p;
        return true;
    }

    void wrapper_heap::free(void *p, std::size_t count)
    {
        util::itt::heap_free heap_free(heap_free_function_, p);

#if HPX_DEBUG_WRAPPER_HEAP != 0
        HPX_ASSERT(did_alloc(p));
#endif
        scoped_lock l(mtx_);

#if HPX_DEBUG_WRAPPER_HEAP != 0
        char* p1 = p;
        std::size_t const total_num_bytes =
            parameters_.capacity * parameters_.element_size;
        std::size_t const num_bytes = count * parameters_.element_size;

        HPX_ASSERT(nullptr != pool_ && p1 >= pool_);
        HPX_ASSERT(nullptr != pool_ &&
            p1 + num_bytes <=
                pool_ + total_num_bytes);
        HPX_ASSERT(first_free_ == nullptr || p1 != first_free_);
        HPX_ASSERT(free_size_ + count <= parameters_.capacity);
        // make sure this has not been freed yet
        HPX_ASSERT(!debug::test_fill_bytes(p1, freed_value,
            num_bytes));

        // give memory back to pool
        debug::fill_bytes(p1, freed_value, num_bytes);
#else
        HPX_UNUSED(p);
#endif

#if defined(HPX_DEBUG)
        free_count_ += count;
#endif
        free_size_ += count;

        // release the pool if this one was the last allocated item
        test_release(l);
    }

    bool wrapper_heap::did_alloc (void *p) const
    {
        // no lock is necessary here as all involved variables are immutable
        util::itt::heap_internal_access hia; HPX_UNUSED(hia);
        if (nullptr == pool_) return false;
        if (nullptr == p) return false;

        std::size_t const total_num_bytes =
            parameters_.capacity * parameters_.element_size;
        return p >= pool_ &&
            static_cast<char*>(p) < pool_ + total_num_bytes;
    }

    naming::gid_type wrapper_heap::get_gid(
        util::unique_id_ranges& ids, void* p, components::component_type type)
    {
        util::itt::heap_internal_access hia; HPX_UNUSED(hia);

        HPX_ASSERT(did_alloc(p));

        scoped_lock l(mtx_);

        if (!base_gid_)
        {
            naming::gid_type base_gid;

            {
                // this is the first call to get_gid() for this heap - allocate
                // a sufficiently large range of global ids
                util::unlock_guard<scoped_lock> ul(l);
                base_gid = ids.get_id(parameters_.capacity);

                // register the global ids and the base address of this heap
                // with the AGAS
                if (!applier::bind_range_local(base_gid, parameters_.capacity,
                        naming::address(hpx::get_locality(), type, pool_),
                        parameters_.element_size))
                {
                    return naming::invalid_gid;
                }
            }

            // if some other thread has already set the base GID for this
            // heap, we ignore the result
            if (!base_gid_)
            {
                // this is the first thread succeeding in binding the new gid
                // range
                base_gid_ = base_gid;
            }
            else
            {
                // unbind the range which is not needed anymore
                util::unlock_guard<scoped_lock> ul(l);
                applier::unbind_range_local(base_gid, parameters_.capacity);
            }
        }

        std::uint64_t const distance = static_cast<char*>(p) - pool_;
        return base_gid_ + distance / parameters_.element_size;
    }

    void wrapper_heap::set_gid(naming::gid_type const& g)
    {
        util::itt::heap_internal_access hia; HPX_UNUSED(hia);

        scoped_lock l(mtx_);
        base_gid_ = g;
    }

    bool wrapper_heap::test_release(scoped_lock& lk)
    {
        if (pool_ == nullptr)
            return false;

        std::size_t const total_num_bytes =
            parameters_.capacity * parameters_.element_size;

        if (first_free_ < pool_ + total_num_bytes ||
            free_size_ < parameters_.capacity)
        {
            return false;
        }

        HPX_ASSERT(free_size_ == parameters_.capacity);
        HPX_ASSERT(first_free_ == pool_ + total_num_bytes);

        // unbind in AGAS service
        if (base_gid_)
        {
            naming::gid_type base_gid = base_gid_;
            base_gid_ = naming::invalid_gid;

            util::unlock_guard<scoped_lock> ull(lk);
            applier::unbind_range_local(base_gid, parameters_.capacity);
        }

        tidy();
        return true;
    }

    bool wrapper_heap::ensure_pool(std::size_t count)
    {
        if (nullptr == pool_)
        {
            return false;
        }

        std::size_t const num_bytes =
            count * parameters_.element_size;
        std::size_t const total_num_bytes =
            parameters_.capacity * parameters_.element_size;

        if (first_free_ + num_bytes > pool_ + total_num_bytes)
        {
            return false;
        }
        return true;
    }

    bool wrapper_heap::init_pool()
    {
        HPX_ASSERT(first_free_ == nullptr);

        std::size_t const total_num_bytes =
            parameters_.capacity * parameters_.element_size;
        pool_ = static_cast<char *>(allocator_type::alloc(total_num_bytes));
        if (nullptr == pool_)
        {
            return false;
        }

        first_free_ = (reinterpret_cast<std::size_t>(pool_)
            % parameters_.element_alignment == 0) ? pool_ :
            pool_ + parameters_.element_alignment;

        free_size_ = parameters_.capacity;

        LOSH_(info)    //-V128
            << "wrapper_heap ("
            << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
            << "): init_pool (" << std::hex << static_cast<void*>(pool_) << ")"
            << " size: " << total_num_bytes << ".";

        return true;
    }

    void wrapper_heap::tidy()
    {
        if (pool_ != nullptr)
        {
            LOSH_(debug) //-V128
                << "wrapper_heap ("
                << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                << ")"
#if defined(HPX_DEBUG)
                << ": releasing heap: alloc count: " << alloc_count_
                << ", free count: " << free_count_
#endif
                << ".";

            if (size() > 0
#if defined(HPX_DEBUG)
                    || alloc_count_ != free_count_
#endif
                )
            {
                LOSH_(warning)    //-V128
                    << "wrapper_heap ("
                    << (!class_name_.empty() ? class_name_.c_str() :
                                               "<Unknown>")
                    << "): releasing heap (" << std::hex
                    << static_cast<void*>(pool_) << ")"
                    << " with " << size() << " allocated object(s)!";
            }

            std::size_t const total_num_bytes =
                parameters_.capacity * parameters_.element_size;
            allocator_type::free(pool_, total_num_bytes);
            pool_ = first_free_ = nullptr;
            free_size_ = 0;
        }
    }
}}}
