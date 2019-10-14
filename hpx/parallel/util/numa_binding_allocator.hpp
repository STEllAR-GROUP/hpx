//  Copyright (c) 2017-2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_HOST_NUMA_BINDING_ALLOCATOR_HPP
#define HPX_COMPUTE_HOST_NUMA_BINDING_ALLOCATOR_HPP

#include <hpx/config.hpp>

#include <hpx/assertion.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/runtime/threads/executors/guided_pool_executor.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/topology/topology.hpp>

#include <sstream>
#include <string>
#include <vector>

#include <cstddef>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#if defined(__linux) || defined(linux) || defined(__linux__)
#include <linux/unistd.h>
#include <sys/mman.h>
#define NUMA_ALLOCATOR_LINUX
#endif

// Can be used to enable debugging of the allocator page mapping
//#define NUMA_BINDING_ALLOCATOR_INIT_MEMORY
//#define NUMA_BINDING_ALLOCATOR_DEBUG_PAGE_BINDING

#if defined(NUMA_BINDING_ALLOCATOR_DEBUG_PAGE_BINDING) && !defined(HPX_MSVC)
#include <plugins/parcelport/parcelport_logging.hpp>
#define LOG_NUMA_MSG(x) std::cout << "<NUMA> " << THREAD_ID << ' ' << x << "\n"
#else
#define LOG_NUMA_MSG(x)
#endif

namespace hpx { namespace threads { namespace executors {
    struct numa_binding_allocator_tag
    {
    };

    template <>
    struct HPX_EXPORT pool_numa_hint<numa_binding_allocator_tag>
    {
        // The call operator () must return an int type
        // The arguments must be const ref versions of the equivalent task arguments
        int operator()(const int& domain) const
        {
            LOG_NUMA_MSG("allocator pool_numa_hint returns domain " << domain);
            return domain;
        }
    };
}}}

namespace hpx { namespace compute { namespace host {
    template <typename T>
    struct numa_binding_helper
    {
        // After memory has been allocated, this operator will be called
        // for every page of memory in the allocation by one thread in each
        // numa domain (implying that this function will be called for each page
        // by N threads, where N = number of domains.
        // The return from this function should be the domain number that
        // should touch this page. The thread with the matching domain will
        // perform a memory read/write on the page.
        virtual std::size_t operator()(const T* const /*base_ptr*/,
            const T* const /*page_ptr*/,
            const std::size_t /*page_size*/,
            const std::size_t /*domains*/) const
        {
            return 0;
        }
        // virtual destructor to quiet compiler warnings
        virtual ~numa_binding_helper() = default;

        // The allocator uses the pool name to get numa bitmap masks needed by
        // the allocation function. The "default" pool is assumed
        // @TODO use an executor to retrieve the pool name
        virtual const std::string& pool_name() const
        {
            return pool_name_;
        }

        // Return the total memory consumption in bytes
        virtual std::size_t memory_bytes() const
        {
            return 0;
        }

        // Using how many dimensions should this data be displayed
        // (This function is only required/used) for debug purposes
        virtual std::size_t array_rank() const
        {
            return 1;
        }

        // The number of elements along dimension x=0,y=1,z=2,...
        // This function is only required for debug purposes
        virtual std::size_t array_size(std::size_t axis) const
        {
            return memory_bytes() / sizeof(T);
        };

        // When counting along elements in a given dimension,
        // how large a step should be taken in units of elements.
        // This should include padding along an axis
        // This function is only required for debug purposes
        virtual std::size_t memory_step(std::size_t axis) const
        {
            return 1;
        };

        // When displaying the data, what step size should be used
        // This function is only required for debug purposes
        virtual std::size_t display_step(std::size_t axis) const
        {
            return 1;
        };

        // This is for debug/information purposes only and anything may
        // be returned that helps illuminate the mapping of pages->indices
        virtual std::string description() const
        {
            return "";
        };
        //#endif

        std::string pool_name_ = "default";
    };

    template <typename T>
    using numa_binding_helper_ptr = std::shared_ptr<numa_binding_helper<T>>;

    /// The numa_binding_allocator allocates memory using a policy based on
    /// hwloc flags for memory binding.
    /// This allocator can be used to request data that is bound
    /// to one or more numa domains via the bitmap mask supplied

    template <typename T>
    struct numa_binding_allocator
    {
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef T const& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        //
        template <typename U>
        struct rebind
        {
            typedef numa_binding_allocator<U> other;
        };

        using numa_binding_helper_ptr = std::shared_ptr<numa_binding_helper<T>>;

        // default constructor produces an unusable allocator
        numa_binding_allocator() = default;

        // construct without a memory binder function
        // only first touch or interleave policies are valid
        numa_binding_allocator(threads::hpx_hwloc_membind_policy policy,
            unsigned int flags)
          : policy_(policy)
          , flags_(flags)
          , init_mutex()
        {
            LOG_NUMA_MSG("numa_binding_allocator : no binder function");
            HPX_ASSERT(
                policy_ != threads::hpx_hwloc_membind_policy::membind_user);
        }

        // construct using a memory binder function for placing pages
        // according to a user defined pattern using first touch policy
        numa_binding_allocator(numa_binding_helper_ptr bind_func,
            threads::hpx_hwloc_membind_policy policy,
            unsigned int flags)
          : binding_helper_(bind_func)
          , policy_(policy)
          , flags_(flags)
          , init_mutex()
        {
            LOG_NUMA_MSG("numa_binding_allocator : allocator");
        }

        // copy constructor
        numa_binding_allocator(numa_binding_allocator const& rhs)
          : binding_helper_(rhs.binding_helper_)
          , policy_(rhs.policy_)
          , flags_(rhs.flags_)
          , init_mutex()
        {
            LOG_NUMA_MSG("numa_binding_allocator : Copy allocator");
        }

        // copy constructor using rebind type
        template <typename U>
        numa_binding_allocator(numa_binding_allocator<U> const& rhs)
          : binding_helper_(rhs.binding_helper_)
          , policy_(rhs.policy_)
          , flags_(rhs.flags_)
          , init_mutex()
        {
            LOG_NUMA_MSG("numa_binding_allocator : Copy allocator rebind");
        }

        // Move constructor
        numa_binding_allocator(numa_binding_allocator&& rhs)
          : binding_helper_(std::move(rhs.binding_helper_))
          , policy_(rhs.policy_)
          , flags_(rhs.flags_)
          , init_mutex()
        {
            LOG_NUMA_MSG("numa_binding_allocator : Move constructor");
        }

        // Assignment operator
        numa_binding_allocator& operator=(numa_binding_allocator const& rhs)
        {
            binding_helper_ = rhs.binding_helper_;
            policy_ = rhs.policy_;
            flags_ = rhs.flags_;

            LOG_NUMA_MSG("numa_binding_allocator : Assignment operator");
            return *this;
        }

        // Move assignment
        numa_binding_allocator& operator=(numa_binding_allocator&& rhs)
        {
            binding_helper_ = rhs.binding_helper_;
            policy_ = rhs.policy_;
            flags_ = rhs.flags_;

            LOG_NUMA_MSG("numa_binding_allocator : Move assignment");
            return *this;
        }

        // Returns the actual address of x even in presence of overloaded
        // operator&
        pointer address(reference x) const noexcept
        {
            return &x;
        }

        // Returns the actual address of x even in presence of overloaded
        // operator&
        const_pointer address(const_reference x) const noexcept
        {
            return &x;
        }

        // Allocates n * sizeof(T) bytes of uninitialized storage and
        // then spawns threads to touch memory if membind_user is selected
        pointer allocate(size_type n)
        {
            pointer result = nullptr;

            if (policy_ ==
                threads::hpx_hwloc_membind_policy::membind_firsttouch)
            {
                threads::hwloc_bitmap_ptr bitmap =
                    threads::get_thread_manager().get_pool_numa_bitmap(
                        binding_helper_->pool_name());
                //
                result = reinterpret_cast<pointer>(
                    threads::topology().allocate_membind(
                        n * sizeof(T), bitmap, policy_, 0));
            }
            else if (policy_ ==
                threads::hpx_hwloc_membind_policy::membind_interleave)
            {
                threads::hwloc_bitmap_ptr bitmap =
                    threads::get_thread_manager().get_pool_numa_bitmap(
                        binding_helper_->pool_name());
                //
                result = reinterpret_cast<pointer>(
                    threads::topology().allocate_membind(
                        n * sizeof(T), bitmap, policy_, 0));
            }
            else if (policy_ == threads::hpx_hwloc_membind_policy::membind_user)
            {
                threads::hwloc_bitmap_ptr bitmap =
                    threads::get_thread_manager().get_pool_numa_bitmap(
                        binding_helper_->pool_name());
                //
                result = reinterpret_cast<pointer>(
                    threads::topology().allocate_membind(n * sizeof(T), bitmap,
                        threads::hpx_hwloc_membind_policy::membind_firsttouch,
                        0));
#if defined(NUMA_ALLOCATOR_LINUX)
                // if Transparent Huge Pages (THP) are enabled, this prevents
                // pages from being merged into a single numa bound block
                int ret = madvise(result, n * sizeof(T), MADV_NOHUGEPAGE);
                // a return of -1 probably means there are no transparent huge pages
                // so they can't be disabled, we can ignore it
                if ((ret != 0) && (ret != -1))
                {
                    std::cerr << "ERROR: MADVISE " << strerror(ret)
                              << std::endl;
                }
#endif
                initialize_pages(result, n);
            }
            return result;
        }

        // Deallocates the storage referenced by the pointer p, which must be a
        // pointer obtained by an earlier call to allocate(). The argument n
        // must be equal to the first argument of the call to allocate() that
        // originally produced p; otherwise, the behavior is undefined.
        void deallocate(pointer p, size_type n)
        {
            LOG_NUMA_MSG("Calling deallocate membind for size (bytes) "
                << std::hex << (n * sizeof(T)));
#ifdef NUMA_BINDING_ALLOCATOR_DEBUG_PAGE_BINDING
            display_binding(p, binding_helper_);
#endif
            threads::topology().deallocate(p, n * sizeof(T));
        }

        // Returns the maximum theoretically possible value of n, for which the
        // call allocate(n, 0) could succeed. In most implementations, this
        // returns std::numeric_limits<size_type>::max() / sizeof(value_type).
        size_type max_size() const noexcept
        {
            return (std::numeric_limits<size_type>::max)();
        }

        // Constructs an object of type T in allocated uninitialized storage
        // pointed to by p, using placement-new
        template <class U, class... A>
        void construct(U* const p, A&&... args)
        {
            new (p) U(std::forward<A>(args)...);
        }

        template <class U>
        void destroy(U* const p)
        {
            p->~U();
        }

        // a utility function that is slightly faster than the hwloc provided one
        // @TODO, move this into hpx::topology for cleanliness
        int get_numa_domain(void* page)
        {
            HPX_ASSERT((std::size_t(page) & 4095) == 0);
#if defined(NUMA_ALLOCATOR_LINUX)
            // This is an optimized version of the hwloc equivalent
            void* pages[1] = {page};
            int status[1] = {-1};
            if (syscall(__NR_move_pages, 0, 1, pages, nullptr, status, 0) == 0)
            {
                if (status[0] >= 0 &&
                    status[0] <= HPX_HAVE_MAX_NUMA_DOMAIN_COUNT)
                {
                    return status[0];
                }
                // if (status[0]<0) std::cout << "." << decnumber(status[0]) << ".";
                return -1;
            }
            HPX_THROW_EXCEPTION(kernel_error,
                "get_numa_domain",
                "Error getting numa domain with syscall");
#else
            return threads::get_topology().get_numa_domain(page);
#endif
        }

        std::string get_page_numa_domains(void* addr, std::size_t len) const
        {
#if defined(NUMA_ALLOCATOR_LINUX)
            // @TODO replace with topology::page_size
            int pagesize = threads::get_memory_page_size();
            HPX_ASSERT((std::size_t(addr) & (pagesize - 1)) == 0);

            std::size_t count = (len + pagesize - 1) / pagesize;
            std::vector<void*> pages(count, nullptr);
            std::vector<int> status(count, 0);

            for (std::size_t i = 0; i < count; i++)
                pages[i] = ((char*) addr) + i * pagesize;

            if (syscall(__NR_move_pages, 0, count, pages.data(), nullptr,
                    status.data(), 0) < 0)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "get_page_numa_domains",
                    "Error getting numa domains with syscall");
            }

            std::stringstream temp;
            temp << "Numa page binding for page count " << count << "\n";
            for (std::size_t i = 0; i < count; i++)
            {
                if (status[i] >= 0)
                    temp << status[i];
                else
                    temp << "-";
            }
            return temp.str();
#endif
            return "";
        }

        void initialize_pages(pointer p, size_t n) const
        {
            std::unique_lock<std::mutex> lk(init_mutex);
            //
            threads::hwloc_bitmap_ptr bitmap =
                threads::get_thread_manager().get_pool_numa_bitmap(
                    binding_helper_->pool_name());
            std::vector<threads::hwloc_bitmap_ptr> nodesets =
                create_nodesets(bitmap);
            //
            using namespace threads::executors;
            using allocator_hint_type =
                pool_numa_hint<numa_binding_allocator_tag>;
            //
            // Warning :low priority tasks are used here, because the scheduler does
            // not steal across numa domains for those tasks, so they are sure
            // to remain on the right queue and be executed on the right domain.
            guided_pool_executor<allocator_hint_type> numa_executor(
                binding_helper_->pool_name(), threads::thread_priority_low);

            LOG_NUMA_MSG("Launching First-Touch tasks");
            // for each numa domain, we must launch a task to 'touch' the memory
            // that should be bound to the domain
            std::vector<hpx::future<void>> tasks;
            for (size_type i = 0; i < nodesets.size(); ++i)
            {
#ifdef NUMA_BINDING_ALLOCATOR_DEBUG_PAGE_BINDING
                size_type i1 = hwloc_bitmap_first(nodesets[i]->get_bmp());
                size_type i2 = hwloc_bitmap_last(nodesets[i]->get_bmp());
                HPX_ASSERT(i1 == i2);
#endif
                size_type domain = hwloc_bitmap_first(nodesets[i]->get_bmp());
                LOG_NUMA_MSG("Launching First-Touch task for domain "
                    << domain << " " << nodesets[i]);
                // if (domain==7 || domain==3)
                tasks.push_back(hpx::async(numa_executor,
                    util::bind(&numa_binding_allocator::touch_pages, this, p, n,
                        binding_helper_, util::placeholders::_1, nodesets),
                    domain));
            }
            wait_all(tasks);
            LOG_NUMA_MSG("Done First-Touch tasks");
        }

#ifdef NUMA_BINDING_ALLOCATOR_DEBUG_PAGE_BINDING
        std::string display_binding(pointer p, numa_binding_helper_ptr helper)
        {
            std::unique_lock<std::mutex> lk(init_mutex);
            //
            std::ostringstream display;
            auto N = helper->array_rank();
            display << helper->description() << "\n";
            std::size_t pagesize = threads::get_memory_page_size();
            pointer p2 = p;
            if (N == 2)
            {
                std::size_t Nc = helper->array_size(0);
                std::size_t Nr = helper->array_size(1);
                std::size_t xinc =
                    (std::min)(helper->display_step(0), pagesize);
                std::size_t yinc =
                    (std::min)(helper->display_step(1), pagesize);
                std::size_t xoff = helper->memory_step(0);
                std::size_t yoff = helper->memory_step(1);
                std::size_t m = helper->memory_bytes();
#ifdef numa_binding_allocator_PRETTY_DISPLAY
                display << ' ';
                for (std::size_t c = 0; c < Nc; c += xinc)
                {
                    if (c % (xinc * 8) == 0)
                        display << '|';
                    else
                        display << '.';
                }
#endif
                display << '\n';
                for (std::size_t r = 0; r < Nr; r += yinc)
                {
#ifdef numa_binding_allocator_PRETTY_DISPLAY
                    if (r % (xinc * 8) == 0)
                        display << '-';
                    else
                        display << '.';
#endif
                    //
                    for (std::size_t c = 0; c < Nc; c += xinc)
                    {
                        p2 = p + (c * xoff) + (r * yoff);
                        if (p2 >= (p + m))
                        {
                            display << '*';
                        }
                        else
                        {
                            size_type dom = get_numa_domain(p2);
                            if (dom == size_type(-1))
                            {
                                display << '-';
                            }
                            else
                                display << std::hex << dom;
                        }
                    }
                    display << "\n";
                }
            }
            else
            {
                display << "Not yet implemented output for non 2D arrays"
                        << "\n";
            }
            return display.str();
        }
#endif

    protected:
        std::vector<threads::hwloc_bitmap_ptr> create_nodesets(
            threads::hwloc_bitmap_ptr bitmap) const
        {
            // for each numa domain, we need a nodeset object
            threads::mask_type numa_mask =
                dynamic_cast<const threads::topology*>(&threads::get_topology())
                    ->bitmap_to_mask(bitmap->get_bmp(), HWLOC_OBJ_NUMANODE);

            LOG_NUMA_MSG("Pool numa mask is " << numa_mask);

            std::vector<threads::hwloc_bitmap_ptr> nodesets;
            for (size_type i = 0; i < threads::mask_size(numa_mask); ++i)
            {
                if (threads::test(numa_mask, i))
                {
                    hwloc_bitmap_t bitmap = hwloc_bitmap_alloc();
                    hwloc_bitmap_zero(bitmap);
                    hwloc_bitmap_set(bitmap, i);
                    nodesets.push_back(
                        std::make_shared<threads::hpx_hwloc_bitmap_wrapper>(
                            bitmap));
                    LOG_NUMA_MSG(
                        "Node mask " << i << " is " << nodesets.back());
                }
            }
            return nodesets;
        }

        void touch_pages(pointer p, size_t n, numa_binding_helper_ptr helper,
            size_type numa_domain,
            const std::vector<threads::hwloc_bitmap_ptr>& nodesets) const
        {
            const size_type pagesize = threads::get_memory_page_size();
            const size_type pageN = pagesize / sizeof(T);
            const size_type num_pages =
                (n * sizeof(T) + pagesize - 1) / pagesize;
            pointer page_ptr = p;
            HPX_ASSERT(reinterpret_cast<std::intptr_t>(p) % pagesize == 0);

            LOG_NUMA_MSG("touch pages for numa " << numa_domain);
            for (size_type i = 0; i < num_pages; ++i)
            {
                // we pass the base pointer and current page pointer
                size_type dom =
                    helper->operator()(p, page_ptr, pagesize, nodesets.size());
                if (dom == numa_domain)
                {
                    HPX_ASSERT((std::size_t(page_ptr) &
                                (threads::get_memory_page_size()-1)) == 0);
                    // trigger a memory read and rewrite without changing contents
                    volatile T* vaddr = const_cast<volatile T*>(page_ptr);
                    *vaddr = *vaddr;
#ifdef NUMA_BINDING_ALLOCATOR_INIT_MEMORY
#if defined(NUMA_ALLOCATOR_LINUX)
                    int Vmem =
                        sched_getcpu();    // show which cpu is actually being used
#else
                    int Vmem =
                        numa_domain;    // show just the domain we think we're on
#endif
                    pointer elem_ptr = page_ptr;
                    for (size_type j = 0; j < pageN; ++j)
                    {
                        *elem_ptr++ = T(Vmem);
                    }
#endif
                }
                page_ptr += pageN;
            }
        }

        // This is obsolete but kept for possible future use
        void bind_pages(pointer p, size_t n, numa_binding_helper_ptr helper,
            size_type numa_domain,
            const std::vector<threads::hwloc_bitmap_ptr>& nodesets) const
        {
            const size_type pagesize = threads::get_memory_page_size();
            const size_type pageN = pagesize / sizeof(T);
            const size_type num_pages =
                (n * sizeof(T) + pagesize - 1) / pagesize;
            pointer page_ptr = p;
            HPX_ASSERT(reinterpret_cast<std::intptr_t>(p) % pagesize == 0);

            LOG_NUMA_MSG("bind pages for numa " << numa_domain);
            for (size_type i = 0; i < num_pages; ++i)
            {
                // we pass the base pointer and current page pointer
                size_type dom =
                    helper->operator()(p, page_ptr, pagesize, nodesets.size());
                if (dom == numa_domain)
                {
                    threads::topology().set_area_membind_nodeset(
                        page_ptr, pagesize, nodesets[dom]->get_bmp());
                }
                page_ptr += pageN;
            }
        }

    public:
        // return the binding helper cast to a specific type
        template <typename Binder>
        std::shared_ptr<Binder> get_binding_helper_cast() const
        {
            return std::dynamic_pointer_cast<Binder>(binding_helper_);
        }

        std::shared_ptr<numa_binding_helper<T>> binding_helper_;
        threads::hpx_hwloc_membind_policy       policy_;
        unsigned int                            flags_;

    private:
        mutable std::mutex init_mutex;
    };
}}}

#endif
