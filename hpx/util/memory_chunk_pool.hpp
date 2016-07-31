//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MEMORY_CHUNK_POOL_HPP
#define HPX_UTIL_MEMORY_CHUNK_POOL_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_chunk_allocator.hpp>
#include <hpx/util/memory_chunk.hpp>
#include <hpx/util/memory_chunk_pool_allocator.hpp>

#include <boost/atomic.hpp>

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

// forward declare pool
namespace hpx { namespace util
{
    template <typename Mutex>
    struct memory_chunk_pool;
}}

// specialize chunk pool allocator traits for this memory_chunk_pool
namespace hpx { namespace traits
{
    // if the chunk pool supplies fixed chunks of memory when the alloc
    // is smaller than some threshold, then the pool must declare
    // std::size_t chunk_size_
    template <typename T, typename M>
    struct is_chunk_allocator<
            util::detail::memory_chunk_pool_allocator<
                T, util::memory_chunk_pool<M>, M
            >
        >
      : std::true_type
    {};
}}

namespace hpx { namespace util
{
    template <typename Mutex = hpx::lcos::local::spinlock>
    struct memory_chunk_pool
    {
    private:
        HPX_NON_COPYABLE(memory_chunk_pool);

    public:
        typedef Mutex mutex_type;
        typedef memory_chunk<mutex_type> memory_chunk_type;

        typedef typename memory_chunk_type::size_type size_type;
        typedef std::multimap<size_type, char *> backup_chunks_type;

        static const std::size_t offset_ = sizeof(memory_chunk_type *);

        memory_chunk_pool(std::size_t chunk_size, std::size_t max_chunks)
          : last_used_chunk_(0)
          , memory_chunks_(max_chunks, memory_chunk_type(chunk_size))
          , chunk_size_(chunk_size)
          , max_chunks_(max_chunks)
          , backup_size_(0)
          , backup_threshold_(max_chunks * chunk_size)
        {
        }

        ~memory_chunk_pool()
        {
            for (typename backup_chunks_type::value_type& v : backup_chunks_)
            {
                char *ptr = v.second - offset_;
#ifdef _POSIX_SOURCE
                free(ptr);
#else
                delete[] ptr;
#endif
            }
        }

        std::pair<char *, size_type> get_chunk_address(char * p, size_type size)
        {
            if(size > chunk_size_)
                return std::make_pair(p, size);

            memory_chunk_type *chunk = nullptr;
            std::memcpy(&chunk, p - offset_, offset_);
            if(chunk)
            {
                return std::make_pair(chunk->data_.get(), chunk_size_);
            }

            return std::make_pair(p, size);
        }

        char *allocate(size_type size)
        {
            char * result = nullptr;

            if(size + offset_ <= chunk_size_)
            {
                std::size_t i = last_used_chunk_;
                HPX_ASSERT(i < memory_chunks_.size());
                std::size_t count = 0;
                while(count != memory_chunks_.size())
                {
                    memory_chunk_type & chunk = memory_chunks_[i];

                    // We encode the chunk address at the first few bytes to
                    // avoid a linear search on deallocation
                    result = chunk.allocate(size + offset_);
                    if(result != 0)
                    {
                        void * chunk_addr = &chunk;
                        std::memcpy(result, &chunk_addr, offset_);
#if defined(HPX_DEBUG)
                        memory_chunk_type *chunk_test = nullptr;
                        std::memcpy(&chunk_test, result, offset_);
                        HPX_ASSERT(chunk_test == &chunk);
#endif
                        last_used_chunk_.store(i);
                        return result + offset_;
                    }
                    i = (i + 1) % memory_chunks_.size();
                    ++count;
                }
            }

            {
                std::lock_guard<mutex_type> l(backup_chunks_mtx_);
                typename backup_chunks_type::iterator it =
                    backup_chunks_.lower_bound(size);

                if(it != backup_chunks_.end())
                {
                    result = it->second;
                    backup_size_ -= it->first;
                    backup_chunks_.erase(it);
                    return result;
                }
            }

#ifdef _POSIX_SOURCE
            int ret = posix_memalign(
                reinterpret_cast<void **>(&result),
                EXEC_PAGESIZE, size + offset_);
            if(ret != 0 && !result)
                throw std::bad_alloc();
#else
            result = new char[size];
#endif
            std::memset(result, 0, offset_);
            return result + offset_;
        }

        void deallocate(char * p, size_type size)
        {
            memory_chunk_type *chunk = nullptr;
            std::memcpy(&chunk, p - offset_, offset_);
            if(chunk)
            {
#if defined(HPX_DEBUG)
                bool valid_chunk = false;
                for (memory_chunk_type& c : memory_chunks_)
                {
                    if(&c == chunk)
                    {
                        valid_chunk = true;
                        break;
                    }
                }
                HPX_ASSERT(valid_chunk);
#endif
                HPX_ASSERT(chunk->contains(p - offset_));
                chunk->deallocate(p - offset_, size + offset_);

                HPX_ASSERT(std::size_t(chunk - &memory_chunks_[0]) <
                    memory_chunks_.size());
                last_used_chunk_.store(chunk - &memory_chunks_[0]);
            }
            else
            {
                std::lock_guard<mutex_type> l(backup_chunks_mtx_);
                if(backup_size_ <= backup_threshold_)
                {
                    backup_size_ += size;
                    backup_chunks_.insert(std::make_pair(size, p));
                }
                else
                {
                    char *ptr = p - offset_;
#ifdef _POSIX_SOURCE
                    free(ptr);
#else
                    delete[] ptr;
#endif
                }
            }
        }

        boost::atomic<std::size_t> last_used_chunk_;
        std::vector<memory_chunk_type> memory_chunks_;
        std::size_t const chunk_size_;
        std::size_t const max_chunks_;

        mutable mutex_type backup_chunks_mtx_;
        backup_chunks_type backup_chunks_;
        std::size_t backup_size_;
        std::size_t const backup_threshold_;
    };

}}

#endif
