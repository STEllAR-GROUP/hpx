//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MEMORY_CHUNK_HPP
#define HPX_UTIL_MEMORY_CHUNK_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>

// include unist.d conditionally to check for POSIX version. Not all OSs have the
// unistd header...
#if defined(HPX_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#if defined(_POSIX_VERSION)
#include <hpx/runtime/threads/coroutines/detail/posix_utility.hpp>
#endif

#include <boost/thread/locks.hpp>

namespace hpx { namespace util
{
    template <typename Mutex>
    struct memory_chunk
    {
        typedef char data_type;
        typedef char * iterator;
        typedef std::size_t size_type;
        typedef std::multimap<size_type, iterator> free_list_type;
        typedef Mutex mutex_type;

        static void deleter(char * p)
        {
#ifdef _POSIX_SOURCE
            free(p);
#else
            delete[] p;
#endif
        }

        explicit memory_chunk(std::size_t chunk_size)
          : chunk_size_(chunk_size)
          , allocated_(0)
          , current_(0)
        {}

        void charge()
        {
            HPX_ASSERT(allocated_ == 0);
            HPX_ASSERT(free_list_.empty());
            HPX_ASSERT(!data_);
#ifdef _POSIX_SOURCE
            char * ptr = 0;
            int ret = posix_memalign(
                reinterpret_cast<void **>(&ptr), EXEC_PAGESIZE,
                chunk_size_);
            if(ret != 0)
                throw std::bad_alloc();
#else
            char * ptr = new char[chunk_size_];
#endif
            data_.reset(ptr, deleter);
            current_ = ptr;
        }

        memory_chunk(memory_chunk const & other)
          : data_(other.data_)
          , chunk_size_(other.chunk_size_)
          , allocated_(other.allocated_)
          , current_(other.current_)
          , free_list_(other.free_list_)
        {
        }

        ~memory_chunk()
        {
            if(data_)
            {
                /*
                std::cout << "allocated: " << allocated_ << "\n";
                std::cout << "free_list size: " << free_list_.size() << "\n";
                */
            }
        }

        bool full() const
        {
            boost::lock_guard<mutex_type> l(mtx_);
            check_invariants_locked();
            if(allocated_ == chunk_size_)
            {
                return free_list_.empty();
            }
            return false;
        }

        bool contains(char * p) const
        {
            boost::lock_guard<mutex_type> l(mtx_);
            return contains_locked(p);
        }

        bool contains_locked(char * p) const
        {
            if(!data_) return false;

            if(p >= data_.get() && p < data_.get() + chunk_size_)
                return true;

            return false;
        }

        void check_invariants(char * p = 0, std::size_t size = 0) const
        {
#ifdef HPX_DEBUG
            boost::lock_guard<mutex_type> l(mtx_);
            check_invariants_locked(p, size);
#endif
        }

        void check_invariants_locked(char * p = 0, std::size_t size = 0) const
        {
            HPX_ASSERT(allocated_ <= chunk_size_);
#ifdef HPX_DEBUG
            if(data_)
            {
                HPX_ASSERT(data_.get() <= current_);
                HPX_ASSERT(current_ <= data_.get() + chunk_size_);
                for (free_list_type::value_type const & v : free_list_)
                {
                    HPX_ASSERT(v.first <= chunk_size_);
                    HPX_ASSERT(v.second != current_);
                    HPX_ASSERT(v.second >= data_.get());
                    HPX_ASSERT(v.second < data_.get() + chunk_size_);
                    if(p)
                    {
                        HPX_ASSERT(p < v.second || p >= v.second + v.first);
                    }
                }
                if(p)
                {
                    HPX_ASSERT(p != current_);
                    HPX_ASSERT(size <= chunk_size_);
                    HPX_ASSERT(p >= data_.get());
                    HPX_ASSERT(p < data_.get() + chunk_size_);
                }
            }
#endif
        }

        char *allocate(size_type size)
        {
            boost::lock_guard<mutex_type> l(mtx_);
            check_invariants_locked();

            if(!data_) charge();

            iterator result = 0;

            HPX_ASSERT(size <= chunk_size_);

            typedef free_list_type::iterator free_iterator;
            free_iterator it = free_list_.lower_bound(size);
            if(it != free_list_.end())
            {
                result = it->second;
                HPX_ASSERT(it->first >= size);
                if(it->first == size)
                {
                    free_list_.erase(it);
                    check_invariants_locked(result, size);
                    return result;
                }
                size_type new_size = it->first - size;
                iterator new_free_block = it->second + size;
                free_list_.erase(it);
                if(new_free_block + new_size == current_)
                {
                    current_ = new_free_block;
                    allocated_ -= new_size;
                }
                else
                {
                    free_list_.insert(std::make_pair(new_size, new_free_block));
                }
                check_invariants_locked(result, size);
                return result;
            }

            if(size + allocated_ <= chunk_size_)
            {
                result = current_;
                current_ += size;
                allocated_ += size;
                check_invariants_locked(result, size);
                return result;
            }
            check_invariants_locked(0, size);
            return 0;
        }

        bool deallocate(char * p, size_type size)
        {
            boost::lock_guard<mutex_type> l(mtx_);
            HPX_ASSERT(data_);
            if(!contains_locked(p))
                return false;

            HPX_ASSERT(p != current_);

            check_invariants_locked(p, size);

            if(current_ - size == p)
            {
                allocated_ -= size;
                current_ -= size;
                check_invariants_locked(0, size);
                return true;
            }

            iterator p_end = p + size;
            for(free_list_type::iterator jt = free_list_.begin();
                jt != free_list_.end(); ++jt)
            {
                // check if the chunk to be deleted is left to the chunk in
                // the free list
                iterator chunk_begin = jt->second;
                if(p_end == chunk_begin)
                {
                    size_type new_size = size + jt->first;
                    free_list_.erase(jt);
                    if(p + new_size == current_)
                    {
                        current_ -= new_size;
                        allocated_ -= new_size;
                        check_invariants_locked(0, size);
                        return true;
                    }
                    free_list_.insert(std::make_pair(new_size, p));
                    check_invariants_locked(0, size);
                    return true;
                }

                // check if the chunk to be deleted is right to the chunk in
                // the free list
                iterator chunk_end = chunk_begin + jt->first;
                if(p == chunk_end)
                {
                    size_type new_size = size + jt->first;
                    iterator new_free = jt->second;
                    free_list_.erase(jt);
                    if(new_free + new_size == current_)
                    {
                        current_ -= new_size;
                        allocated_ -= new_size;
                        check_invariants_locked(0, size);
                        return true;
                    }
                    free_list_.insert(std::make_pair(new_size, new_free));
                    check_invariants_locked(0, size);
                    return true;
                }
                // Since we can't have overlapping regions, we don't need more
                // checks ...
            }
            free_list_.insert(std::make_pair(size, p));
            check_invariants_locked(0, size);
            return true;
        }

        mutable mutex_type mtx_;
        boost::shared_ptr<data_type> data_;
        size_type const chunk_size_;
        size_type allocated_;
        iterator current_;
        free_list_type free_list_;
    };
}}

#endif
