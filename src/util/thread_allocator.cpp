//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This is partially taken from: http://www.garret.ru/threadalloc/readme.html

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/atomic_count.hpp>
#include <hpx/util/thread_allocator.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <memory>
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
        template <typename T>
        class compressed_ptr
        {
        public:
            using compressed_ptr_type = std::uint64_t;
            using tag_type = std::uint16_t;

        private:
            union cast_unit
            {
                compressed_ptr_type value;
                tag_type tag[4];
            };

            HPX_STATIC_CONSTEXPR int tag_index = 3;
            HPX_STATIC_CONSTEXPR compressed_ptr_type ptr_mask =
                0xffffffffffffUL;   // (1L << 48L) - 1

            HPX_CONSTEXPR HPX_FORCEINLINE static T* extract_ptr(
                volatile compressed_ptr_type const& i) noexcept
            {
                return reinterpret_cast<T*>(i & ptr_mask);
            }

            HPX_CONSTEXPR HPX_FORCEINLINE static tag_type extract_tag(
                volatile compressed_ptr_type const& i) noexcept
            {
                cast_unit cu;
                cu.value = i;
                return cu.tag[tag_index];
            }

            HPX_CONSTEXPR HPX_FORCEINLINE static compressed_ptr_type pack_ptr(
                T* ptr, tag_type tag) noexcept
            {
                cast_unit ret;
                ret.value = compressed_ptr_type(ptr);
                ret.tag[tag_index] = tag;
                return ret.value;
            }

        public:
            HPX_CONSTEXPR HPX_FORCEINLINE compressed_ptr() noexcept
              : ptr(pack_ptr(nullptr, 0))
            {}
            HPX_CONSTEXPR HPX_FORCEINLINE explicit compressed_ptr(
                    T* p, tag_type t = 0) noexcept
              : ptr(pack_ptr(p, t))
            {}

            compressed_ptr(compressed_ptr const& p) = delete;
            compressed_ptr& operator=(compressed_ptr const& p) = delete;

            compressed_ptr(compressed_ptr && p) = delete;
            compressed_ptr& operator=(compressed_ptr && p) = delete;

            HPX_CONSTEXPR HPX_FORCEINLINE void set(T* p, tag_type t)
            {
                ptr = pack_ptr(p, t);
            }

            HPX_CONSTEXPR bool operator==(
                volatile compressed_ptr const& p) const noexcept
            {
                return ptr == p.ptr;
            }
            HPX_FORCEINLINE bool operator!=(
                volatile compressed_ptr const& p) const noexcept
            {
                return ptr != p.ptr;
            }

            HPX_CONSTEXPR HPX_FORCEINLINE T* get_ptr() const noexcept
            {
                return extract_ptr(ptr);
            }
            HPX_CONSTEXPR HPX_FORCEINLINE void set_ptr(T* p) noexcept
            {
                ptr = pack_ptr(p, get_tag());
            }

            HPX_CONSTEXPR HPX_FORCEINLINE tag_type get_tag() const noexcept
            {
                return extract_tag(ptr);
            }
            HPX_CONSTEXPR HPX_FORCEINLINE void set_tag(tag_type t) noexcept
            {
                ptr = pack_ptr(get_ptr(), t);
            }

            HPX_CONSTEXPR HPX_FORCEINLINE explicit operator bool() const noexcept
            {
                return get_ptr() != nullptr;
            }

            HPX_CONSTEXPR HPX_FORCEINLINE T& operator*() noexcept
            {
                return get_ptr();
            }
            HPX_CONSTEXPR HPX_FORCEINLINE T const& operator*() const noexcept
            {
                return get_ptr();
            }
            HPX_CONSTEXPR HPX_FORCEINLINE T* operator->() noexcept
            {
                return get_ptr();
            }

        protected:
            compressed_ptr_type ptr;
        };

        ///////////////////////////////////////////////////////////////////////
        struct alloc_page;

        struct alloc_block_header
        {
            HPX_CONSTEXPR HPX_FORCEINLINE alloc_block_header(
                    alloc_page* p = nullptr, std::int16_t size = 0) noexcept
              : page(p, size)
            {}

            HPX_CONSTEXPR HPX_FORCEINLINE alloc_page* get_ptr() noexcept
            {
                return page.get_ptr();
            }

            HPX_CONSTEXPR HPX_FORCEINLINE std::int16_t get_size() const noexcept
            {
                return page.get_tag();
            }
            HPX_CONSTEXPR HPX_FORCEINLINE void set_size(std::int16_t s) noexcept
            {
                page.set_tag(s);
            }

            compressed_ptr<alloc_page> page;
        };

        struct alloc_block : alloc_block_header
        {
            HPX_CONSTEXPR HPX_FORCEINLINE alloc_block(
                    alloc_page* page = nullptr, std::int16_t size = 0) noexcept
              : alloc_block_header(page, size)
              , next_free(nullptr)
            {}

            HPX_CONSTEXPR alloc_block* next(std::size_t block_size) noexcept
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
        util::atomic_count pending_requests;
        mutex_type mtx;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct alloc_page
        {
            alloc_page(thread_local_allocator* alloc) noexcept
              : next(alloc->pages)
              , allocator(alloc)
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

            HPX_CONSTEXPR HPX_FORCEINLINE alloc_block* get() noexcept
            {
                return reinterpret_cast<alloc_block*>(&data);
            }

            HPX_CONSTEXPR HPX_FORCEINLINE alloc_block const* get() const
                noexcept
            {
                return reinterpret_cast<alloc_block const*>(&data);
            }

            // for the available page size we account for the members of this
            // class below
            HPX_STATIC_CONSTEXPR std::size_t page_size =
                PAGE_SIZE - 2 * sizeof(void*) - sizeof(std::size_t);

            typename std::aligned_storage<page_size>::type data;

            alloc_page* next;
            thread_local_allocator* allocator;
            std::size_t allocated_blocks;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_local_allocator& thread_local_allocator::get_tls_allocator()
    {
        static HPX_NATIVE_TLS thread_local_allocator ctx{};
        return ctx;
    }

    ///////////////////////////////////////////////////////////////////////////
    void thread_local_allocator::process_pending_requests(bool forceall)
    {
        if (forceall || pending_requests > PENDING_THRESHOLD)
        {
            detail::alloc_block* blk = nullptr;

            {
                std::unique_lock<mutex_type> lk(mtx);
                blk = pending_chain;
                pending_chain = nullptr;
                pending_requests = 0;
            }

            while (blk != nullptr)
            {
                detail::alloc_block* next = blk->next_free;
                std::size_t i = block_chain[blk->get_size() / BLOCK_ALIGNMENT];
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
            detail::alloc_block* blk = pending_chain;
            pending_chain = nullptr;

            {
                hpx::util::unlock_guard<std::unique_lock<mutex_type>> ul(lk);

                for (/**/; blk != nullptr; blk = blk->next_free)
                {
                    --blk->page->allocated_blocks;
                }

                detail::alloc_page** ppg = &pages;
                for (detail::alloc_page* pg = pages; pg != nullptr; /**/)
                {
                    detail::alloc_page* next = pg->next;
                    if (pg->allocated_blocks == 0)
                    {
                        *ppg = next;

                        pg->detail::alloc_page::~alloc_page();
                        std::free(pg);
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
        if (size <= MAX_BLOCK_SIZE)
        {
            thread_local_allocator* allocator =
                &thread_local_allocator::get_tls_allocator();
            allocator->process_pending_requests();

            std::size_t i = block_chain[size / BLOCK_ALIGNMENT];

            blk = allocator->chain[i];
            if (blk != nullptr)
            {
                allocator->chain[i] = blk->next_free;
                blk->set_size(static_cast<std::int16_t>(size));
            }
            else
            {
                detail::alloc_page* pg = reinterpret_cast<detail::alloc_page*>(
                    std::malloc(sizeof(detail::alloc_page)));
                if (nullptr == pg)
                {
                    return nullptr;
                }
                new (pg) detail::alloc_page(allocator);

                allocator->pages = pg;

                std::size_t block_size = chain_block_size[i];
                std::size_t blocks =
                    detail::alloc_page::page_size / block_size - 2;

                // FIXME: move block construction into alloc_page constructor
                allocator->chain[i] = pg->get()->next(block_size);

                blk = allocator->chain[i];
                new (blk) detail::alloc_block(pg);

                do
                {
                    detail::alloc_block* next = blk->next(block_size);
                    new (next) detail::alloc_block(pg);

                    blk->next_free = next;
                    blk = next;
                } while (--blocks != 0);

                blk->next_free = nullptr;

                blk = pg->get();
                new (blk)
                    detail::alloc_block(pg, static_cast<std::int16_t>(size));

                blk->next_free = nullptr;
            }

            ++blk->page->allocated_blocks;
        }
        else
        {
            blk = reinterpret_cast<detail::alloc_block*>(std::malloc(size));
            if (blk == nullptr)
            {
                return nullptr;
            }
            new (blk) detail::alloc_block(nullptr, MAX_BLOCK_SIZE + 1);
        }

        return static_cast<detail::alloc_block_header*>(blk) + 1;
    }

    void thread_free(void* addr)
    {
        if (addr == nullptr)
        {
            return;     // ignore nullptr arguments
        }

        detail::alloc_block* blk = static_cast<detail::alloc_block*>(
            reinterpret_cast<detail::alloc_block_header*>(addr) - 1);

        if (blk->get_size() > MAX_BLOCK_SIZE)
        {
            blk->detail::alloc_block::~alloc_block();
            std::free(blk);
        }
        else
        {
            detail::alloc_page* page = blk->get_ptr();
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

                std::size_t i = block_chain[blk->get_size() / BLOCK_ALIGNMENT];
                blk->next_free = allocator->chain[i];
                allocator->chain[i] = blk;
            }
        }
    }
}}

