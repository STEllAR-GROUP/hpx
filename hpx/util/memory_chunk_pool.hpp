//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MEMORY_CHUNK_POOL_HPP
#define HPX_UTIL_MEMORY_CHUNK_POOL_HPP

#include <hpx/config.hpp>

#include <hpx/util/memory_chunk.hpp>

#include <cstdlib>
#include <vector>

namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T, typename Mutex = hpx::lcos::local::spinlock>
        struct memory_chunk_pool_allocator;
    }

    template <typename Mutex = hpx::lcos::local::spinlock>
    struct memory_chunk_pool : boost::noncopyable
    {
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
            BOOST_FOREACH(typename backup_chunks_type::value_type &v, backup_chunks_)
            {
                char *ptr = v.second - offset_;
#if _POSIX_SOURCE
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

            memory_chunk_type *chunk = 0;
            std::memcpy(&chunk, p - offset_, offset_);
            if(chunk)
            {
                return std::make_pair(chunk->data_.get(), chunk_size_);
            }

            return std::make_pair(p, size);
        }

        char *allocate(size_type size)
        {
            char * result = 0;

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
                        memory_chunk_type *chunk_test = 0;
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
                typename mutex_type::scoped_lock l(backup_chunks_mtx_);
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

#if _POSIX_SOURCE
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
            memory_chunk_type *chunk = 0;
            std::memcpy(&chunk, p - offset_, offset_);
            if(chunk)
            {
#if defined(HPX_DEBUG)
                bool valid_chunk = false;
                BOOST_FOREACH(memory_chunk_type & c, memory_chunks_)
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
                HPX_ASSERT(std::size_t(chunk - &memory_chunks_[0]) < memory_chunks_.size());
                last_used_chunk_.store(chunk - &memory_chunks_[0]);
            }
            else
            {
                typename mutex_type::scoped_lock l(backup_chunks_mtx_);
                if(backup_size_ <= backup_threshold_)
                {
                    backup_size_ += size;
                    backup_chunks_.insert(std::make_pair(size, p));
                }
                else
                {
                    char *ptr = p - offset_;
#if _POSIX_SOURCE
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

    namespace detail
    {
        template <typename T, typename Mutex>
        struct memory_chunk_pool_allocator
        {
            typedef std::size_t size_type;
            typedef std::ptrdiff_t difference_type;
            typedef T* pointer;
            typedef const T* const_pointer;
            typedef T& reference;
            typedef const T& const_reference;
            typedef T value_type;

            template <typename U>
            struct rebind
            {
                typedef memory_chunk_pool_allocator<U, Mutex> other;
            };

            memory_chunk_pool_allocator() throw()
              : memory_pool_(0)
            {}
            memory_chunk_pool_allocator(
                    util::memory_chunk_pool<Mutex> & mp) throw()
              : memory_pool_(&mp)
            {}
            memory_chunk_pool_allocator(
                    memory_chunk_pool_allocator const & other) throw()
              : memory_pool_(other.memory_pool_)
            {}
            template <typename U>
            memory_chunk_pool_allocator(
                    memory_chunk_pool_allocator<U, Mutex> const & other) throw()
              : memory_pool_(other.memory_pool_)
            {}

            pointer address(reference x) const
            {
                return &x;
            }

            const_pointer address(const_reference x) const
            {
                return &x;
            }

            pointer allocate(size_type n, void* /*hint*/ = 0)
            {
                HPX_ASSERT(memory_pool_);
                return reinterpret_cast<T*>(memory_pool_->allocate(sizeof(T) * n));
            }

            void deallocate(pointer p, size_type n)
            {
                HPX_ASSERT(memory_pool_);
                memory_pool_->deallocate(reinterpret_cast<char*>(p), sizeof(T) * n);
            }

            size_type max_size() const throw()
            {
                return (std::numeric_limits<std::size_t>::max)() / sizeof(T);
            }

            void construct(pointer p)
            {
                new (p) T();
            }

            template <typename U>
            void construct(pointer p, U && val)
            {
                new (p) typename util::decay<T>::type(std::forward<U>(val));
            }

            /** Destroy the object referenced by @c p. */
            void destroy(pointer p)
            {
                p->~T();
            }

            bool operator==(memory_chunk_pool_allocator const & other)
            {
                return memory_pool_ == other.memory_pool_;
            }

            bool operator!=(memory_chunk_pool_allocator const & other)
            {
                return memory_pool_ != other.memory_pool_;
            }

            util::memory_chunk_pool<Mutex> * memory_pool_;
        };
    }
}}

#endif
