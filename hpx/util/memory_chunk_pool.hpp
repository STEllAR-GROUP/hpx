//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MEMORY_CHUNK_POOL_HPP
#define HPX_UTIL_MEMORY_CHUNK_POOL_HPP

#include <hpx/config.hpp>

#include <hpx/util/memory_chunk.hpp>

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
        typedef std::multimap<size_type, char *> large_chunks_type;

        memory_chunk_pool(std::size_t chunk_size, std::size_t max_chunks)
          : memory_chunks_(max_chunks, memory_chunk_type(chunk_size))
          , chunk_size_(chunk_size)
          , max_chunks_(max_chunks)
        {
        }

        ~memory_chunk_pool()
        {
            BOOST_FOREACH(typename large_chunks_type::value_type &v, large_chunks_)
            {
#if POSIX_VERSION_
                free(v.second);
#else
                delete[] v.second;
#endif
            }
        }

        std::pair<char *, size_type> get_chunk_address(char * p, size_type size)
        {
            if(size > chunk_size_)
                return std::make_pair(p, size);

            typename mutex_type::scoped_lock l(chunks_mtx_);
            BOOST_FOREACH(memory_chunk_type & chunk, memory_chunks_)
            {
                if(chunk.contains(p))
                    return std::make_pair(chunk.data_.get(), chunk_size_);
            }
            return std::make_pair(p, size);
        }

        char *allocate(size_type size)
        {
            char * result = 0;

            if(size <= chunk_size_)
            {
                typename mutex_type::scoped_lock l(chunks_mtx_);
                BOOST_FOREACH(memory_chunk_type & chunk, memory_chunks_)
                {
                    result = chunk.allocate(size);
                    if(result != 0)
                        return result;
                }
            }

            {
                typename mutex_type::scoped_lock l(large_chunks_mtx_);
                typename large_chunks_type::iterator it =
                    large_chunks_.find(size);

                if(it != large_chunks_.end())
                {
                    result = it->second;
                    large_chunks_.erase(it);
                    return result;
                }
            }

#if POSIX_VERSION_
            int ret = posix_memalign(
                reinterpret_cast<void **>(&result),
                EXEC_PAGESIZE, size);
            if(ret != 0)
                throw std::bad_alloc();
#else
            result = new char[size];
#endif
            return result;
        }

        void deallocate(char * p, size_type size)
        {
            if(size <= chunk_size_)
            {
                typename mutex_type::scoped_lock l(chunks_mtx_);
                BOOST_FOREACH(memory_chunk_type & chunk, memory_chunks_)
                {
                    if(chunk.deallocate(p, size))
                    {
                        return;
                    }
                }
            }

            {
                typename mutex_type::scoped_lock l(large_chunks_mtx_);
                large_chunks_.insert(std::make_pair(size, p));
            }
        }

        mutable mutex_type chunks_mtx_;
        std::vector<memory_chunk_type> memory_chunks_;
        std::size_t const chunk_size_;
        std::size_t const max_chunks_;

        mutable mutex_type large_chunks_mtx_;
        large_chunks_type large_chunks_;
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

            template <typename T>
            void construct(pointer p, T && val)
            {
                new (p) typename util::decay<T>::type(std::forward<T>(val));
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
