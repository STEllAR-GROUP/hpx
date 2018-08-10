//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This is partially taken from: http://www.garret.ru/threadalloc/readme.html

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/thread_allocator.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <type_traits>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    HPX_CONSTEXPR std::size_t NUM_CHAINS = 12;
    HPX_CONSTEXPR std::size_t BLOCK_ALIGNMENT = 8;
    HPX_CONSTEXPR std::size_t PENDING_THRESHOLD = 100;
    HPX_CONSTEXPR std::size_t PAGE_SIZE = 16384;

    HPX_CONSTEXPR std::int16_t MAX_BLOCK_SIZE = 640;

    HPX_CONSTEXPR std::int8_t block_chain[] =
    {
         0,
         0,  0,  0,  0,                     // 32 bytes
         1,  1,  1,  1,                     // 64 bytes
         2,  2,  2,  2,                     // 96 bytes
         3,  3,  3,  3,                     // 128 bytes
         4,  4,  4,  4,  4,  4,  4,  4,     // 192 bytes
         5,  5,  5,  5,  5,  5,  5,  5,     // 256 bytes
         6,  6,  6,  6,  6,  6,  6,  6,     // 320 bytes
         7,  7,  7,  7,  7,  7,  7,  7,     // 384 bytes
         8,  8,  8,  8,  8,  8,  8,  8,     // 448 bytes
         9,  9,  9,  9,  9,  9,  9,  9,     // 512 bytes
        10, 10, 10, 10, 10, 10, 10, 10,     // 576 bytes
        11, 11, 11, 11, 11, 11, 11, 11,     // 640 bytes
    };

    HPX_CONSTEXPR std::int16_t chain_block_size[NUM_CHAINS] =
    {
        32, 64, 96, 128, 192, 256, 320, 384, 448, 512, 576, 640
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct alloc_page;

        struct alloc_block_header
        {
            alloc_block_header(alloc_page* p = nullptr)
              : size(0)
              , page(p)
            {}

            std::int16_t size;
            alloc_page* page;
        };

        struct alloc_block : alloc_block_header
        {
            alloc_block(alloc_page* page = nullptr)
              : alloc_block_header(page)
              , next_free(nullptr)
            {}

            alloc_block* next(std::size_t block_size)
            {
                return reinterpret_cast<detail::alloc_block*>(
                    reinterpret_cast<char*>(this) + block_size);
            }

            alloc_block* next_free;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    class thread_local_allocator
    {
        using mutex_type = hpx::lcos::local::spinlock;

    public:
        thread_local_allocator()
          : pending_chain(nullptr)
          , pages(nullptr)
          , pending_requests(0)
        {}

        ~thread_local_allocator()
        {
            process_pending_requests(true);
            cleanup();
        }

        thread_local_allocator(thread_local_allocator const&) = delete;
        thread_local_allocator(thread_local_allocator&&) = delete;

        thread_local_allocator& operator=(thread_local_allocator const&) = delete;
        thread_local_allocator& operator=(thread_local_allocator&&) = delete;

        static thread_local_allocator& get_tls_allocator();
        void process_pending_requests(bool forceall = false);

    private:
        void cleanup();

        friend HPX_EXPORT void* thread_alloc(std::size_t size);
        friend HPX_EXPORT void thread_free(void* addr);

        friend struct detail::alloc_page;

        std::array<detail::alloc_block*, NUM_CHAINS> chain;
        detail::alloc_block* pending_chain;
        detail::alloc_page* pages;
        std::atomic<std::size_t> pending_requests;
        mutex_type mtx;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct alloc_page
        {
            alloc_page(thread_local_allocator* alloc)
              : next(alloc->pages)
              , allocator(alloc)
              , allocated_blocks(0)
            {}

            ~alloc_page()
            {
                // FIXME: loop over blocks and call destructor
            }

            alloc_block* get()
            {
                return reinterpret_cast<alloc_block*>(&data);
            }

            alloc_block const* get() const
            {
                return reinterpret_cast<alloc_block const*>(&data);
            }

            // FIXME: should we align this on page boundaries?
            typename std::aligned_storage<PAGE_SIZE>::type data;

            alloc_page *next;
            thread_local_allocator *allocator;
            std::atomic<std::size_t> allocated_blocks;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_local_allocator& thread_local_allocator::get_tls_allocator()
    {
        // FIXME: create feature test for thread_local
        static thread_local thread_local_allocator ctx{};
        return ctx;
    }

    ///////////////////////////////////////////////////////////////////////////
    void thread_local_allocator::process_pending_requests(bool forceall)
    {
        if (forceall ||
            pending_requests.load(std::memory_order_acquire) > PENDING_THRESHOLD)
        {
            detail::alloc_block* blk = nullptr;

            {
                std::unique_lock<mutex_type> lk(mtx);
                blk = pending_chain;
                pending_chain = nullptr;
                pending_requests.store(0, std::memory_order_release);
            }

            while (blk != nullptr)
            {
                detail::alloc_block* next = blk->next_free;
                std::size_t i = block_chain[blk->size / BLOCK_ALIGNMENT];
                --blk->page->allocated_blocks;
                blk->next_free = chain[i];
                chain[i] = blk;
                blk = next;
            }
        }
    }

    void thread_local_allocator::cleanup()
    {
        std::unique_lock<mutex_type> lk(mtx);
        while (true)
        {
            for (detail::alloc_block* blk = pending_chain; blk != nullptr;
                blk = blk->next_free)
            {
                --blk->page->allocated_blocks;
            }
            pending_chain = nullptr;

            {
                hpx::util::unlock_guard<std::unique_lock<mutex_type>> ul(lk);

                detail::alloc_page** ppg = &pages;
                for (detail::alloc_page* pg = pages; pg != nullptr; /**/)
                {
                    detail::alloc_page* next = pg->next;
                    if (!pg->allocated_blocks.load(std::memory_order_acquire))
                    {
                        *ppg = next;
                        delete pg;
                    }
                    else
                    {
                        ppg = &pg->next;
                    }

                    pg = next;
                }
                *ppg = nullptr;
            }

            // if there are still no pending requests, exit
            if (pending_chain == nullptr)
            {
                break;
            }
        }
    }

    void* thread_alloc(std::size_t size)
    {
        size = ((size + BLOCK_ALIGNMENT - 1) & ~(BLOCK_ALIGNMENT - 1)) +
            sizeof(detail::alloc_block_header);

        detail::alloc_block* blk = nullptr;
        if (size > MAX_BLOCK_SIZE)
        {
            blk = reinterpret_cast<detail::alloc_block*>(std::malloc(size));
        }
        else
        {
            thread_local_allocator* allocator =
                &thread_local_allocator::get_tls_allocator();
            allocator->process_pending_requests();

            std::size_t i = block_chain[size / BLOCK_ALIGNMENT];
            blk = allocator->chain[i];
            if (blk == nullptr)
            {
                detail::alloc_page* pg = new detail::alloc_page(allocator);
                allocator->pages = pg;

                std::size_t block_size = chain_block_size[i];
                std::size_t blocks = PAGE_SIZE / block_size - 2;

                // FIXME: move block construction into alloc_page constructor
                allocator->chain[i] = pg->get()->next(block_size);

                blk = allocator->chain[i];
                new (blk) detail::alloc_block;

                do
                {
                    detail::alloc_block* next = blk->next(block_size);
                    new (next) detail::alloc_block;

                    blk->page = pg;
                    blk->next_free = next;

                    blk = next;
                } while (--blocks != 0);

                blk->page = pg;
                blk->next_free = nullptr;

                blk = pg->get();
                new (blk) detail::alloc_block(pg);

                blk->page = pg;
                blk->next_free = nullptr;
            }
            else
            {
                allocator->chain[i] = blk->next_free;
            }

            ++blk->page->allocated_blocks;
        }

        blk->size = static_cast<std::int16_t>(size);
        return reinterpret_cast<detail::alloc_block_header*>(blk) + 1;
    }

    void thread_free(void* addr)
    {
        if (addr == nullptr)
        {
            return;     // ignore nullptr arguments
        }

        detail::alloc_block* blk = reinterpret_cast<detail::alloc_block*>(
            reinterpret_cast<detail::alloc_block_header*>(addr) - 1);

        if (blk->size > MAX_BLOCK_SIZE)
        {
            free(blk);
        }
        else
        {
            detail::alloc_page* page = blk->page;
            thread_local_allocator* allocator = page->allocator;
            thread_local_allocator* current =
                &thread_local_allocator::get_tls_allocator();

            if (allocator != current)
            {
                std::unique_lock<thread_local_allocator::mutex_type> lk(
                    allocator->mtx);

                blk->next_free = allocator->pending_chain;
                allocator->pending_chain = blk;
                ++allocator->pending_requests;
            }
            else
            {
                --page->allocated_blocks;

                std::size_t i = block_chain[blk->size / BLOCK_ALIGNMENT];
                blk->next_free = allocator->chain[i];
                allocator->chain[i] = blk;
            }
        }
    }
}}

