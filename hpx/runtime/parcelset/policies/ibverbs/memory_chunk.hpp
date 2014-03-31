//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#ifndef HPX_PARCELSET_POLICIES_IBVERBS_MEMORY_CHUNK_HPP
#define HPX_PARCELSET_POLICIES_IBVERBS_MEMORY_CHUNK_HPP

#include <hpx/config.hpp>

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/helper.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{

    struct memory_chunk
    {
        typedef char data_type;
        typedef char * iterator;
        typedef std::size_t size_type;
        typedef std::multimap<size_type, iterator> free_list_type;
        typedef std::map<ibv_pd *, ibverbs_mr> mr_map_type;
        typedef hpx::lcos::local::spinlock mutex_type;

        static void deleter(char * p)
        {
            free(p);
        }

        memory_chunk()
          : chunk_size_(0)
          , allocated_(0)
        {}

        void charge(std::size_t chunk_size)
        {
            HPX_ASSERT(chunk_size_ == 0);
            HPX_ASSERT(allocated_ == 0);
            HPX_ASSERT(free_list_.empty());
            HPX_ASSERT(!data_);
            chunk_size_ = chunk_size;
            allocated_ = 0;
            int ret;
            char * ptr;
            ret = posix_memalign(reinterpret_cast<void **>(&ptr), EXEC_PAGESIZE, chunk_size_);
            if(ret != 0)
                throw std::bad_alloc();
            data_.reset(ptr, deleter);
            current_ = ptr;
        }

        memory_chunk(memory_chunk const & other)
          : data_(other.data_)
          , chunk_size_(other.chunk_size_)
          , allocated_(static_cast<std::size_t>(other.allocated_))
          , current_(other.current_)
          , free_list_(other.free_list_)
        {
        }

        ~memory_chunk()
        {
            if(data_)
            {
                std::cout << "allocated: " << allocated_ << "\n";
                std::cout << "free_list size: " << free_list_.size() << "\n";
            }
        }

        bool full() const
        {
            while(true)
            {
                check_invariants();
                std::size_t allocated = allocated_;
                if(allocated == chunk_size_)
                {
                    mutex_type::scoped_lock l(free_list_mtx_);
                    return free_list_.empty();
                }
                if(allocated == allocated_) break;
            }
            return false;
        }

        bool contains(char * p) const
        {
            check_invariants();
            std::ptrdiff_t distance = p - data_.get();
            if((distance < 0)
            || (static_cast<std::size_t>(distance) > chunk_size_))
                return false;

            return true;
        }

        void check_invariants(char * p = 0, std::size_t size = 0) const
        {
#ifdef HPX_DEBUG
            mutex_type::scoped_lock l1(mtx_);
            mutex_type::scoped_lock l2(free_list_mtx_);
            check_invariants_locked(p, size);
#endif
        }

        void check_invariants_locked(char * p = 0, std::size_t size = 0) const
        {
            HPX_ASSERT(allocated_ <= chunk_size_);
            HPX_ASSERT(data_.get() <= current_);
            HPX_ASSERT(current_ <= data_.get() + chunk_size_);
#ifdef HPX_DEBUG
            BOOST_FOREACH(free_list_type::value_type const & v, free_list_)
            {
                HPX_ASSERT(v.first <= chunk_size_);
                HPX_ASSERT(v.second >= data_.get());
                HPX_ASSERT(data_.get() < v.second + v.first);
                HPX_ASSERT(data_.get() + chunk_size_ >= v.second + v.first);
            }
            if(p)
            {
                HPX_ASSERT(p != current_);
                HPX_ASSERT(size <= chunk_size_);
                HPX_ASSERT(p >= data_.get());
                HPX_ASSERT(data_.get() < p + size);
                HPX_ASSERT(data_.get() + chunk_size_ >= p + size);
            }
#endif
        }

        char *allocate(size_type size)
        {
            iterator result;
            check_invariants();

            while(true)
            {
                std::size_t allocated = allocated_;
                if(size + allocated <= chunk_size_)
                {
                    mutex_type::scoped_lock l(mtx_);
                    if(size + allocated_ > chunk_size_) break;
                    result = current_;
                    current_ += size;
                    allocated_ += size;
                    check_invariants_locked(result, size);
                    return result;
                }
                if(allocated == allocated_) break;
            }

            {
                mutex_type::scoped_lock l(free_list_mtx_);
                typedef free_list_type::iterator free_iterator;
                free_iterator it = free_list_.lower_bound(size);
                if(it != free_list_.end())
                {
                    result = it->second;
                    if(it->first == size)
                    {
                        free_list_.erase(it);
                    }
                    else
                    {
                        size_type new_size = it->first - size;
                        iterator new_free = it->second + size;
                        free_list_.erase(it);
                        {
                            mutex_type::scoped_lock l(mtx_);
                            if(new_free + new_size == current_)
                            {
                                current_ = new_free;
                                allocated_ -= new_size;
                            }
                            check_invariants_locked(result, size);
                            return result;
                        }
                        free_list_.insert(std::make_pair(new_size, new_free));
                    }
                    check_invariants_locked(result, size);
                    return result;
                }
            }

            return 0;
        }

        void deallocate(char * p, size_type size)
        {
            HPX_ASSERT(contains(p));
            HPX_ASSERT(p != current_);
            check_invariants(p, size);
            if(current_ - size == p)
            {
                mutex_type::scoped_lock l(mtx_);
                allocated_ -= size;
                current_ -= size;
                check_invariants_locked();
                return;
            }

            iterator it = p;
            iterator it_end = it + size;
            {
                mutex_type::scoped_lock l(free_list_mtx_);
                for(free_list_type::iterator jt = free_list_.begin(); jt != free_list_.end(); ++jt)
                {
                    iterator chunk_begin = jt->second;
                    // check if the chunk to be deleted is left to the chunk in the free list
                    if(it_end == chunk_begin)
                    {
                        size_type new_size = size + jt->first;
                        free_list_.erase(jt);
                        {
                            mutex_type::scoped_lock l(mtx_);
                            if(chunk_begin + new_size == current_)
                            {
                                current_ -= new_size;
                                allocated_ -= new_size;
                            }
                            check_invariants_locked();
                            return;
                        }
                        free_list_.insert(std::make_pair(new_size, chunk_begin));
                        check_invariants_locked();
                        return;
                    }
                    iterator chunk_end = chunk_begin + jt->first;
                    // check if the chunk to be deleted is right to the chunk in the free list
                    if(it == chunk_end)
                    {
                        size_type new_size = size + jt->first;
                        free_list_.erase(jt);
                        {
                            mutex_type::scoped_lock l(mtx_);
                            if(it + new_size == current_)
                            {
                                current_ -= new_size;
                                allocated_ -= new_size;
                            }
                        }
                        free_list_.insert(std::make_pair(new_size, it));
                        check_invariants_locked();
                        return;
                    }
                    // Since we can't have overlapping regions, we don't need more checks ...
                }
                free_list_.insert(std::make_pair(size, it));
            }
            check_invariants_locked();
        }

        ibv_mr register_chunk(ibv_pd *pd)
        {
            mutex_type::scoped_lock l(mr_map_mtx_);
            check_invariants();
            mr_map_type::iterator it = mr_map_.find(pd);
            if(it == mr_map_.end())
            {
                ibverbs_mr mr
                    = ibverbs_mr(
                        pd
                      , data_.get()
                      , chunk_size_
                      , IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                    );
                mr_map_.insert(std::make_pair(pd, mr));
                HPX_ASSERT(mr.mr_->addr == data_.get());
                HPX_ASSERT(mr.mr_->length == chunk_size_);
                return *mr.mr_;
            }
            return *it->second.mr_;
        }

        ibv_mr get_mr(ibv_pd *pd)
        {
#ifdef HPX_DEBUG
            {
                mutex_type::scoped_lock l(mr_map_mtx_);
                HPX_ASSERT(mr_map_.find(pd) != mr_map_.end());
            }
#endif
            return register_chunk(pd);
        }

        mutable mutex_type mtx_;
        mutable mutex_type mr_map_mtx_;
        mr_map_type mr_map_;

        boost::shared_ptr<data_type> data_;
        size_type chunk_size_;
        boost::atomic<size_type> allocated_;
        iterator current_;
        mutable mutex_type free_list_mtx_;
        free_list_type free_list_;
    };

}}}}

#endif
