//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2012 Thomas Heller
//
//  Parts of this code were taken from the Boost.Regex library
//  Copyright (c) 2004 John Maddock
//
//  Parts of this code were taking from this article:
//  http://timday.bitbucket.org/lru.html
//  Copyright (c) 2010-2011 Tim Day
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_IPC_DATA_BUFFER_CACHE_DEC_07_2012_0807AM)
#define HPX_IPC_DATA_BUFFER_CACHE_DEC_07_2012_0807AM

#include <hpx/config/defines.hpp>
#if defined(HPX_HAVE_PARCELPORT_IPC)

#include <hpx/hpx_fwd.hpp>
#include <hpx/plugins/parcelport/ipc/data_buffer.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/tuple.hpp>

#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace policies { namespace ipc
{
    ///////////////////////////////////////////////////////////////////////////
    // This class implements an LRU cache to hold connections.
    class data_buffer_cache
    {
        HPX_NON_COPYABLE(data_buffer_cache);

    public:
        typedef boost::recursive_mutex mutex_type;

        typedef data_buffer connection_type;
        typedef connection_type value_type;
        typedef std::size_t key_type;
        typedef std::list<key_type> key_tracker_type;
        typedef util::tuple<
            value_type, key_tracker_type::iterator
        > cache_value_type;
        typedef std::multimap<key_type, cache_value_type> cache_type;
        typedef cache_type::size_type size_type;

        data_buffer_cache(size_type max_cache_size)
          : max_cache_size_(max_cache_size < 2 ? 2 : max_cache_size)
        {}

        bool get(key_type const& size, connection_type& conn)
        {
            std::lock_guard<mutex_type> lock(mtx_);

            // Check if a matching entry exists ...
            cache_type::iterator const it = cache_.lower_bound(size);

            // If it does ...
            if (it != cache_.end()) {
                // remove the entry from the cache
                conn = util::get<0>(it->second);
                key_tracker_.erase(util::get<1>(it->second));
                cache_.erase(it);

                check_invariants();
                return true;
            }

            // If we get here then the item is not in the cache.
            check_invariants();
            return false;
        }

        // add the given data_buffer to the cache, evict old entries if needed
        void add(key_type const& size, connection_type const& conn)
        {
            std::lock_guard<mutex_type> lock(mtx_);

            if (key_tracker_.empty()) {
                HPX_ASSERT(cache_.empty());
            }
            else {
                // eviction strategy implemented here ...

                // If we reached maximum capacity, evict one entry ...
                // Find the least recently used key entry
                key_tracker_type::iterator it = key_tracker_.begin();

                while (cache_.size() >= max_cache_size_)
                {
                    // find it ...
                    cache_type::iterator const kt = cache_.find(*it);
                    HPX_ASSERT(kt != cache_.end());

                    // ... remove it
                    cache_.erase(kt);
                    key_tracker_.erase(it);

                    it = key_tracker_.begin();
                    HPX_ASSERT(it != key_tracker_.end());
                }
            }

            // Add a new entry ...
            key_tracker_type::iterator it =
                key_tracker_.insert(key_tracker_.end(), size);
            cache_.insert(std::make_pair(size, util::make_tuple(conn, it)));

            check_invariants();
        }

        bool full() const
        {
            std::lock_guard<mutex_type> lock(mtx_);
            return (cache_.size() >= max_cache_size_);
        }

        void clear()
        {
            std::lock_guard<mutex_type> lock(mtx_);
            key_tracker_.clear();
            cache_.clear();

            check_invariants();
        }

    protected:
        // verify class invariants
        void check_invariants() const
        {
            // the list of key trackers should have the right size
            HPX_ASSERT(key_tracker_.size() == cache_.size());
        }

    private:
        mutable mutex_type mtx_;
        size_type const max_cache_size_;
        key_tracker_type key_tracker_;
        cache_type cache_;
    };
}}}}

#endif

#endif
