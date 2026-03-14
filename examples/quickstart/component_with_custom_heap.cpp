//  Copyright (c) 2019-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate how to control the memory
// placement of newly created component instances. Here we use a simple
// free-list allocator to make sure all created components are placed
// consecutively in memory.
// After creating 1000 component instances this example directly accesses those
// in local memory without having to go through the AGAS address resolution.

// make inspect happy: hpxinspect:noinclude:HPX_ASSERT

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/iostream.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Implementation of a free_list allocator, this has no bearings on the example
// component below.
namespace allocator {
    ///////////////////////////////////////////////////////////////////////////
    constexpr std::size_t BLOCK_ALIGNMENT = 8;
    constexpr std::size_t PAGE_SIZE_ = 16384;

    struct alloc_page;

    ///////////////////////////////////////////////////////////////////////////
    struct alloc_block_header
    {
        constexpr HPX_FORCEINLINE alloc_block_header(
            alloc_page* p = nullptr) noexcept
          : page(p)
        {
        }

        ~alloc_block_header() = default;

        alloc_block_header(alloc_block_header const&) = delete;
        alloc_block_header(alloc_block_header&&) = delete;

        alloc_block_header& operator=(alloc_block_header const&) = delete;
        alloc_block_header& operator=(alloc_block_header&&) = delete;

        alloc_page* page;
    };

    template <typename T>
    struct alloc_block : alloc_block_header
    {
        static constexpr std::size_t allocation_size =
            ((sizeof(T) + BLOCK_ALIGNMENT - 1) & ~(BLOCK_ALIGNMENT - 1)) +
            sizeof(alloc_block_header);

        constexpr HPX_FORCEINLINE alloc_block(
            alloc_page* page = nullptr) noexcept
          : alloc_block_header(page)
          , next_free(nullptr)
        {
        }

        ~alloc_block() = default;

        alloc_block(alloc_block const&) = delete;
        alloc_block(alloc_block&&) = delete;

        alloc_block& operator=(alloc_block const&) = delete;
        alloc_block& operator=(alloc_block&&) = delete;

        HPX_FORCEINLINE alloc_block* operator[](std::size_t i) noexcept
        {
            return reinterpret_cast<alloc_block*>(
                reinterpret_cast<char*>(this) + i * allocation_size);
        }
        HPX_FORCEINLINE alloc_block const* operator[](
            std::size_t i) const noexcept
        {
            return reinterpret_cast<alloc_block const*>(
                reinterpret_cast<char const*>(this) + i * allocation_size);
        }

        alloc_block* next_free;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class free_list_allocator
    {
        using mutex_type = hpx::spinlock;

    public:
        free_list_allocator()
          : chain(nullptr)
          , pages(nullptr)
        {
        }

        ~free_list_allocator() = default;

        free_list_allocator(free_list_allocator const&) = delete;
        free_list_allocator(free_list_allocator&&) = delete;

        free_list_allocator& operator=(free_list_allocator const&) = delete;
        free_list_allocator& operator=(free_list_allocator&&) = delete;

        HPX_FORCEINLINE static free_list_allocator& get_allocator();

        constexpr HPX_FORCEINLINE alloc_page const* first_page() const
        {
            return pages;
        }

        void* alloc();
        void free(void* addr);

    private:
        friend struct alloc_page;

        alloc_block<T>* chain;
        alloc_page* pages;
        mutex_type mtx;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct alloc_page
    {
        HPX_FORCEINLINE explicit alloc_page(std::size_t size) noexcept
          : next(nullptr)
          , allocated_blocks(0)
          , block_size(size)
        {
        }

        // FIXME: loop over blocks and call destructor
        ~alloc_page() = default;

        alloc_page(alloc_page const&) = delete;
        alloc_page& operator=(alloc_page const&) = delete;

        alloc_page(alloc_page&&) = delete;
        alloc_page& operator=(alloc_page&&) = delete;

        template <typename T>
        HPX_FORCEINLINE alloc_block<T>* get_block() noexcept
        {
            static_assert(page_size >= alloc_block<T>::allocation_size,
                "size of objects is larger than configured page size");

            // block_size must match allocation size of requested type
            HPX_ASSERT(block_size == alloc_block<T>::allocation_size);

            return reinterpret_cast<alloc_block<T>*>(&data);
        }
        template <typename T>
        HPX_FORCEINLINE alloc_block<T> const* get_block() const noexcept
        {
            static_assert(page_size >= alloc_block<T>::allocation_size,
                "size of objects is larger than configured page size");

            // block_size must match allocation size of requested type
            HPX_ASSERT(block_size == alloc_block<T>::allocation_size);

            return reinterpret_cast<alloc_block<T> const*>(&data);
        }

        template <typename T>
        HPX_FORCEINLINE T& get(std::size_t i) noexcept
        {
            static_assert(page_size >= alloc_block<T>::allocation_size,
                "size of objects is larger than configured page size");

            // block_size must match allocation size of requested type
            HPX_ASSERT(block_size == alloc_block<T>::allocation_size);

            // NOLINTNEXTLINE(bugprone-casting-through-void)
            return *reinterpret_cast<T*>(static_cast<void*>(
                static_cast<alloc_block_header*>((*get_block<T>())[i]) + 1));
        }
        template <typename T>
        HPX_FORCEINLINE T const& get(std::size_t i) const noexcept
        {
            static_assert(page_size >= alloc_block<T>::allocation_size,
                "size of objects is larger than configured page size");

            // block_size must match allocation size of requested type
            HPX_ASSERT(block_size == alloc_block<T>::allocation_size);

            // NOLINTNEXTLINE(bugprone-casting-through-void)
            return *reinterpret_cast<T const*>(static_cast<void const*>(
                static_cast<alloc_block_header const*>((*get_block<T>())[i]) +
                1));
        }

        // for the available page size we account for the members of this
        // class below
        static constexpr std::size_t page_size =
            PAGE_SIZE_ - sizeof(void*) - 2 * sizeof(std::size_t);

        hpx::aligned_storage_t<page_size> data;

        alloc_page* next;
        std::size_t allocated_blocks;
        std::size_t const block_size;
    };

    template <typename T>
    constexpr HPX_FORCEINLINE T& get(alloc_page* page, std::size_t i) noexcept
    {
        return page->template get<T>(i);
    }
    template <typename T>
    constexpr HPX_FORCEINLINE T const& get(
        alloc_page const* page, std::size_t i) noexcept
    {
        return page->template get<T>(i);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    HPX_FORCEINLINE free_list_allocator<T>&
    free_list_allocator<T>::get_allocator()
    {
        static free_list_allocator ctx{};
        return ctx;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    void* free_list_allocator<T>::alloc()
    {
        std::unique_lock<mutex_type> lk(mtx);

        alloc_block<T>* blk = chain;
        if (HPX_LIKELY(blk != nullptr))
        {
            chain = blk->next_free;
            ++blk->page->allocated_blocks;
            return reinterpret_cast<alloc_block_header*>(blk) + 1;
        }

        alloc_page* pg = nullptr;
        alloc_block<T>* new_chain = nullptr;

        {
            hpx::unlock_guard<std::unique_lock<mutex_type>> ul(lk);

            // allocate new page
            pg = reinterpret_cast<alloc_page*>(std::malloc(sizeof(alloc_page)));
            if (nullptr == pg)
            {
                return nullptr;
            }

            hpx::construct_at(pg, alloc_block<T>::allocation_size);

            // FIXME: move block construction into alloc_page constructor
            blk = new_chain = (*pg->template get_block<T>())[1];
            hpx::construct_at(blk, pg);

            std::size_t blocks =
                (alloc_page::page_size / alloc_block<T>::allocation_size) - 2;

            do
            {
                alloc_block<T>* next = (*blk)[1];
                hpx::construct_at(next, pg);

                blk->next_free = next;
                blk = next;
            } while (--blocks != 0);

            blk->next_free = nullptr;

            blk = pg->template get_block<T>();
            hpx::construct_at(blk, pg);

            blk->next_free = nullptr;
        }

        pg->next = pages;
        pages = pg;
        chain = new_chain;

        ++blk->page->allocated_blocks;

        return reinterpret_cast<alloc_block_header*>(blk) + 1;
    }

    template <typename T>
    void free_list_allocator<T>::free(void* addr)
    {
        if (addr == nullptr)
        {
            return;    // ignore nullptr arguments
        }

        std::unique_lock<mutex_type> lk(mtx);

        auto* blk = static_cast<alloc_block<T>*>(
            reinterpret_cast<alloc_block_header*>(addr) - 1);

        --blk->page->allocated_blocks;

        blk->next_free = chain;
        chain = blk;
    }
}    // namespace allocator

///////////////////////////////////////////////////////////////////////////////
// define component type
struct hello_world_server : hpx::components::component_base<hello_world_server>
{
    hello_world_server(std::size_t cnt = 0)
      : count_(cnt)
    {
    }

    void print(std::size_t pagenum, std::size_t item) const
    {
        if (pagenum != std::size_t(-1))
        {
            hpx::cout << "hello world from page: " << pagenum
                      << ", item: " << item << ", number: " << count_ << "\n";
        }
        else
        {
            hpx::cout << "hello world from item: " << item
                      << ", number: " << count_ << "\n";
        }
    }

    HPX_DEFINE_COMPONENT_ACTION(hello_world_server, print, print_action)

    std::size_t count_;
};

// special heap to use for placing components in free_list_heap
template <typename Component>
struct free_list_component_heap
{
    // alloc and free have to be exposed from a component heap
    static void* alloc(std::size_t)
    {
        return allocator::free_list_allocator<Component>::get_allocator()
            .alloc();
    }

    static void free(void* p, std::size_t)
    {
        return allocator::free_list_allocator<Component>::get_allocator().free(
            p);
    }

    // this is an additional function needed just for this example
    static allocator::alloc_page const* get_first_page()
    {
        return allocator::free_list_allocator<Component>::get_allocator()
            .first_page();
    }
};

///////////////////////////////////////////////////////////////////////////////
// associate heap with component above
namespace hpx { namespace traits {
    template <>
    struct component_heap_type<hello_world_server>
    {
        using type = free_list_component_heap<hello_world_server>;
    };
}}    // namespace hpx::traits

// the component macros must come after the component_heap_type specialization
using server_type = hpx::components::component<hello_world_server>;
HPX_REGISTER_COMPONENT(server_type, hello_world_server)

using print_action = hello_world_server::print_action;
HPX_REGISTER_ACTION_DECLARATION(print_action)
HPX_REGISTER_ACTION(print_action)

///////////////////////////////////////////////////////////////////////////////
int main()
{
    // create 100 new component instances
    std::vector<hpx::id_type> ids;
    ids.reserve(1000);

    for (std::size_t i = 0; i != 1000; ++i)
    {
        ids.push_back(hpx::local_new<hello_world_server>(hpx::launch::sync, i));
    }

    // Just for demonstration purposes we now access the components directly
    // without having to go through AGAS to resolve their addresses. This
    // obviously relies on internal knowledge of the used heap.

    // Extract base pointer to the array (pages) managed by the heap above.
    free_list_component_heap<hello_world_server>& heap =
        hpx::components::component_heap<server_type>();

    std::size_t pagenum = 0;
    for (auto* page = heap.get_first_page(); page != nullptr;
        page = page->next, ++pagenum)
    {
        auto blocks = page->allocated_blocks;
        for (std::size_t i = 0; i != blocks; ++i)
        {
            allocator::get<hello_world_server>(page, i).print(pagenum, i);
        }
    }

    // now do the same but using AGAS
    std::size_t i = 0;
    for (auto const& id : ids)
    {
        print_action()(id, std::size_t(-1), i++);
    }

    return 0;
}

#endif
