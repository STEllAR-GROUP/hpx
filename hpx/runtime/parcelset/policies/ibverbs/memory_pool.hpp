//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_IBVERBS_MEMORY_POOL_HPP
#define HPX_PARCELSET_POLICIES_IBVERBS_MEMORY_POOL_HPP

#include <hpx/config.hpp>

#include <hpx/runtime/parcelset/policies/ibverbs/memory_chunk.hpp>

#include <vector>

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    struct memory_pool : boost::noncopyable
    {
        typedef memory_chunk::size_type size_type;
        typedef std::multimap<size_type, char *> large_chunks_type;
        typedef hpx::lcos::local::spinlock mutex_type;

        memory_pool(std::size_t chunk_size, std::size_t max_chunks)
          : memory_chunks_(max_chunks)
          , chunk_size_(chunk_size)
          , charged_chunks_(1)
          , max_chunks_(max_chunks)
        {
            memory_chunks_[0].charge(chunk_size_);
        }

        ~memory_pool()
        {
            BOOST_FOREACH(large_chunks_type::value_type &v, large_chunks_)
            {
                free(v.second);
            }
            std::cout << "charged chunks: " << charged_chunks_ << "\n";
            std::cout << "large chunks: " << large_chunks_.size() << "\n";
        }

        bool contains(char * p) const
        {
            std::size_t start = 0;
            while(true)
            {
                std::size_t charged_chunks = charged_chunks_;
                for(std::size_t i = start; i < charged_chunks; ++i)
                {
                    if(memory_chunks_[i].contains(p))
                    {
                        return true;
                    }
                }
                if(charged_chunks == charged_chunks_) break;
                else start = charged_chunks;
            }
            return false;
        }

        char *allocate(size_type size)
        {
            {
                char * result = 0;
                if(size <= chunk_size_)
                {
                    std::size_t start = 0;
                    while(true)
                    {
                        std::size_t charged_chunks = charged_chunks_;
                        for(std::size_t i = start; i < charged_chunks; ++i)
                        {
                            if(!memory_chunks_[i].full())
                            {
                                result = memory_chunks_[i].allocate(size);
                                if(result != 0) return result;
                            }
                        }
                        if(charged_chunks == charged_chunks_) break;
                        else start = charged_chunks;
                    }
                    if(charged_chunks_ < max_chunks_)
                    {
                        std::size_t chunk_idx = charged_chunks_++;
                        memory_chunks_[chunk_idx].charge(chunk_size_);
                        memory_chunk & chunk = memory_chunks_[chunk_idx];

                        result = chunk.allocate(size);
                        {
                            mutex_type::scoped_lock l(pds_mtx_);
                            BOOST_FOREACH(ibv_pd *pd, pds_)
                            {
                                chunk.register_chunk(pd);
                            }
                        }

                        HPX_ASSERT(result);
                        return result;
                    }
                }
                {
                    mutex_type::scoped_lock l(large_chunks_mtx_);
                    large_chunks_type::iterator it = large_chunks_.find(size);
                    if(it != large_chunks_.end())
                    {
                        result = it->second;
                        large_chunks_.erase(it);
                        return result;
                    }
                }
            }
            int ret = 0;
            char * ptr;
            ret = posix_memalign(reinterpret_cast<void **>(&ptr), EXEC_PAGESIZE, size);
            if(ret != 0)
                throw std::bad_alloc();
            return ptr;
        }

        void deallocate(char * p, size_type size)
        {
            if(size <= chunk_size_)
            {
                std::size_t start = 0;
                while(true)
                {
                    std::size_t charged_chunks = charged_chunks_;
                    for(std::size_t i = start; i < charged_chunks; ++i)
                    {
                        if(memory_chunks_[i].contains(p))
                        {
                            memory_chunks_[i].deallocate(p, size);
                            return;
                        }
                    }
                    if(charged_chunks == charged_chunks_) break;
                    else start = charged_chunks;
                }
                HPX_ASSERT(false);
            }
            {
                mutex_type::scoped_lock l(large_chunks_mtx_);
                large_chunks_.insert(std::make_pair(size, p));
            }
        }

        void register_chunk(ibv_pd *pd)
        {
            {
                mutex_type::scoped_lock l(pds_mtx_);
                HPX_ASSERT(std::find(pds_.begin(), pds_.end(), pd) == pds_.end());
                pds_.push_back(pd);
            }
            std::size_t start = 0;
            while(true)
            {
                std::size_t charged_chunks = charged_chunks_;
                for(std::size_t i = start; i < charged_chunks; ++i)
                {
                    memory_chunks_[i].register_chunk(pd);
                }
                if(charged_chunks == charged_chunks_) break;
                else start = charged_chunks;
            }
        }

        ibv_mr get_mr(ibv_pd *pd, char * buffer, std::size_t length)
        {
            ibv_mr result;// = {0};
            result.addr = 0;
#ifdef HPX_DEBUG
            {
                mutex_type::scoped_lock l(pds_mtx_);
                HPX_ASSERT(std::find(pds_.begin(), pds_.end(), pd) != pds_.end());
            }
#endif
            if(length <= chunk_size_)
            {
                std::size_t start = 0;
                while(true)
                {
                    std::size_t charged_chunks = charged_chunks_;
                    for(std::size_t i = start; i < charged_chunks; ++i)
                    {
                        if(memory_chunks_[i].contains(buffer))
                        {
                            result = memory_chunks_[i].get_mr(pd);
                            result.addr = buffer;
                            result.length = length;
                            return result;
                        }
                    }
                    if(charged_chunks == charged_chunks_) break;
                    else start = charged_chunks;
                }
            }
            return result;
        }

        std::vector<memory_chunk> memory_chunks_;
        std::size_t chunk_size_;
        boost::atomic<std::size_t> charged_chunks_;
        std::size_t max_chunks_;

        mutable mutex_type large_chunks_mtx_;
        large_chunks_type large_chunks_;

        mutable mutex_type pds_mtx_;
        std::vector<ibv_pd*> pds_;
    };
}}}}

#endif
