//  Copyright (c) 2019 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate how to control the memory
// placement of newly created component instances. Here we use a simple
// free-list allocator to make sure all created components are placed
// consecutively in memory.
// After creating 1000 component instances this example directly accesses those
// in local memory without having to go through the AGAS address resolution.

#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
// Implementation of a free_list allocator, this has no bearings on the example
// component below.
namespace free_list_allocator
{
    ///////////////////////////////////////////////////////////////////////////
    HPX_CONSTEXPR std::size_t BLOCK_ALIGNMENT = 8;
    HPX_CONSTEXPR std::size_t PAGE_SIZE = 16384;

    template <typename T>
    struct alloc_page;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct alloc_block_header
    {
        HPX_CONSTEXPR HPX_FORCEINLINE alloc_block_header(
                alloc_page<T>* p = nullptr) noexcept
          : page(p)
        {}

        HPX_FORCEINLINE alloc_page<T>* get_ptr() noexcept
        {
            return page;
        }

        alloc_page<T>* page;
    };

    template <typename T>
    struct alloc_block : alloc_block_header<T>
    {
        HPX_STATIC_CONSTEXPR std::size_t allocation_size =
            ((sizeof(T) + BLOCK_ALIGNMENT - 1) & ~(BLOCK_ALIGNMENT - 1)) +
            sizeof(alloc_block_header<T>);

        HPX_CONSTEXPR HPX_FORCEINLINE alloc_block(
                alloc_page<T>* page = nullptr) noexcept
          : alloc_block_header<T>(page)
          , next_free(nullptr)
        {}

        HPX_CONSTEXPR HPX_FORCEINLINE alloc_block<T>* operator[](
            std::size_t i) noexcept
        {
            return reinterpret_cast<alloc_block<T>*>(
                reinterpret_cast<char*>(this) + i * allocation_size);
        }
        HPX_CONSTEXPR HPX_FORCEINLINE alloc_block<T> const* operator[](
            std::size_t i) const noexcept
        {
            return reinterpret_cast<alloc_block<T> const*>(
                reinterpret_cast<char const*>(this) + i * allocation_size);
        }

        alloc_block<T>* next_free;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class free_list_allocator
    {
        using mutex_type = hpx::lcos::local::spinlock;

    public:
        free_list_allocator()
          : chain(nullptr)
          , pages(nullptr)
        {}

        ~free_list_allocator() = default;

        free_list_allocator(free_list_allocator const&) = delete;
        free_list_allocator(free_list_allocator&&) = delete;

        free_list_allocator& operator=(free_list_allocator const&) = delete;
        free_list_allocator& operator=(free_list_allocator&&) = delete;

        HPX_FORCEINLINE static free_list_allocator& get_allocator();

        HPX_CONSTEXPR HPX_FORCEINLINE alloc_page<T> const* first_page() const
        {
            return pages;
        }

        void* alloc();
        void free(void* addr);

    private:

        friend struct alloc_page<T>;

        alloc_block<T>* chain;
        alloc_page<T>* pages;
        mutex_type mtx;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct alloc_page
    {
        HPX_CONSTEXPR HPX_FORCEINLINE alloc_page() noexcept
          : next(nullptr)
          , allocated_blocks(0)
        {}

        ~alloc_page()
        {
            // FIXME: loop over blocks and call destructor
        }

        alloc_page(alloc_page const&) = delete;
        alloc_page& operator=(alloc_page const&) = delete;

        alloc_page(alloc_page &&) = delete;
        alloc_page& operator=(alloc_page &&) = delete;

        HPX_CONSTEXPR HPX_FORCEINLINE alloc_block<T>* get_block() noexcept
        {
            return reinterpret_cast<alloc_block<T>*>(&data);
        }
        HPX_CONSTEXPR HPX_FORCEINLINE alloc_block<T> const* get_block() const noexcept
        {
            return reinterpret_cast<alloc_block<T> const*>(&data);
        }

        HPX_CONSTEXPR HPX_FORCEINLINE T& operator[](std::size_t i) noexcept
        {
            return *reinterpret_cast<T*>(static_cast<void*>(
                static_cast<alloc_block_header<T>*>((*get_block())[i]) + 1));
        }
        HPX_CONSTEXPR HPX_FORCEINLINE T const& operator[](std::size_t i) const
            noexcept
        {
            return *reinterpret_cast<T const*>(static_cast<void const*>(
                static_cast<alloc_block_header<T> const*>((*get_block())[i]) + 1));
        }

        // for the available page size we account for the members of this
        // class below
        HPX_STATIC_CONSTEXPR std::size_t page_size =
            PAGE_SIZE - 2 * sizeof(void*) - sizeof(std::size_t);

        static_assert(page_size >= alloc_block<T>::allocation_size,
            "size of objects is larger than configured page size");

        typename std::aligned_storage<page_size>::type data;

        alloc_page<T>* next;
        std::size_t allocated_blocks;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    HPX_FORCEINLINE free_list_allocator<T>&
        free_list_allocator<T>::get_allocator()
    {
        static free_list_allocator<T> ctx{};
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
            return reinterpret_cast<alloc_block_header<T>*>(blk) + 1;
        }

        alloc_page<T>* pg = nullptr;
        alloc_block<T>* new_chain = nullptr;

        {
            hpx::util::unlock_guard<std::unique_lock<mutex_type>> ul(lk);

            // allocate new page
            pg = reinterpret_cast<alloc_page<T>*>(
                std::malloc(sizeof(alloc_page<T>)));
            if (nullptr == pg)
            {
                return nullptr;
            }

            new (pg) alloc_page<T>();

            // FIXME: move block construction into alloc_page constructor
            blk = new_chain = (*pg->get_block())[1];
            new (blk) alloc_block<T>(pg);

            std::size_t blocks =
                (alloc_page<T>::page_size / alloc_block<T>::allocation_size) - 2;

            do
            {
                alloc_block<T>* next = (*blk)[1];
                new (next) alloc_block<T>(pg);

                blk->next_free = next;
                blk = next;
            } while (--blocks != 0);

            blk->next_free = nullptr;

            blk = pg->get_block();
            new (blk) alloc_block<T>(pg);

            blk->next_free = nullptr;
        }

        pg->next = pages;
        pages = pg;
        chain = new_chain;

        ++blk->page->allocated_blocks;

        return reinterpret_cast<alloc_block_header<T>*>(blk) + 1;
    }

    template <typename T>
    void free_list_allocator<T>::free(void* addr)
    {
        if (addr == nullptr)
        {
            return;     // ignore nullptr arguments
        }

        std::unique_lock<mutex_type> lk(mtx);

        auto* blk = static_cast<alloc_block<T>*>(
            reinterpret_cast<alloc_block_header<T>*>(addr) - 1);

        alloc_page<T>* page = blk->get_ptr();

        --page->allocated_blocks;

        blk->next_free = chain;
        chain = blk;
    }
}

///////////////////////////////////////////////////////////////////////////////
// define component type
struct hello_world_server
  : hpx::components::component_base<hello_world_server>
{
    hello_world_server(std::size_t cnt = 0)
      : count_(cnt)
    {}

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

    HPX_DEFINE_COMPONENT_ACTION(hello_world_server, print, print_action);

    std::size_t count_;
};

// special heap to use for placing components in free_list_heap
template <typename Component>
struct free_list_component_heap
{
    // alloc and free have to be exposed from a component heap
    static void* alloc(std::size_t)
    {
        return free_list_allocator::free_list_allocator<Component>::
            get_allocator().alloc();
    }

    static void free(void* p, std::size_t)
    {
        return free_list_allocator::free_list_allocator<Component>::
            get_allocator().free(p);
    }

    // this is an additional function needed just for this example
    static free_list_allocator::alloc_page<Component> const* get_first_page()
    {
        return free_list_allocator::free_list_allocator<Component>::
            get_allocator().first_page();
    }
};

///////////////////////////////////////////////////////////////////////////////
// associate heap with component above
namespace hpx { namespace traits
{
    template <>
    struct component_heap_type<hello_world_server>
    {
        using type = free_list_component_heap<hello_world_server>;
    };
}}

// the component macros must come after the component_heap_type specialization
using server_type = hpx::components::component<hello_world_server>;
HPX_REGISTER_COMPONENT(server_type, hello_world_server);

using print_action = hello_world_server::print_action;
HPX_REGISTER_ACTION_DECLARATION(print_action);
HPX_REGISTER_ACTION(print_action);

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
            (*page)[i].print(pagenum, i);
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

