//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2012 Thomas Heller
//  Copyright (c)      2012 Bryce Adelstein-Lelbach
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

#if !defined(HPX_UTIL_CONNECTION_CACHE_MAY_20_0104PM)
#define HPX_UTIL_CONNECTION_CACHE_MAY_20_0104PM

#include <map>
#include <list>
#include <stdexcept>
#include <string>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>

#include <boost/assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/cstdint.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    /// This class implements an LRU cache to hold connections. It includes
    /// entries checked out from the cache in its cache size.
    // TODO: investigate usage of boost.cache.
    template <typename Connection, typename Key>
    class connection_cache
    {
    public:
        typedef boost::recursive_mutex mutex_type;

        typedef boost::shared_ptr<Connection> connection_type;
        typedef std::deque<connection_type> value_type;
        typedef Key key_type;
        typedef std::list<key_type> key_tracker_type;
        typedef boost::tuple<
            value_type, std::size_t, typename key_tracker_type::iterator
        > cache_value_type;
        typedef std::map<key_type, cache_value_type> cache_type;
        typedef typename cache_type::size_type size_type;

        connection_cache(
            size_type max_connections
          , size_type max_connections_per_locality
        )
          : max_connections_(max_connections < 2 ? 2 : max_connections)
          , max_connections_per_locality_(max_connections_per_locality)
          , connections_(0)
          , shutting_down_(false)
          , insertions_(0)
          , evictions_(0)
          , hits_(0)
          , misses_(0)
        {
            if (max_connections_per_locality_ > max_connections_)
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "connection_cache::connection_cache",
                    "the maximum number of connections per locality cannot "
                    "excede the overall maximum number of connections");
            }
        }

        void shutdown()
        {
            shutting_down_ = true;
        }

        /// Try to get a connection to \a l from the cache.
        ///
        /// \returns A usable connection to \a l if a connection could be
        ///          found, otherwise a default constructed connection.
        ///
        /// \note    The connection must be returned to the cache by calling
        ///          \a reclaim().
        connection_type get(key_type const& l)
        {
            mutex_type::scoped_lock lock(mtx_);

            // Check if this key already exists in the cache.
            typename cache_type::iterator const it = cache_.find(l);

            // Check if this key already exists in the cache.
            if (it != cache_.end())
            {
                // Key exists in cache.

                // Update LRU meta data.
                key_tracker_.splice(
                    key_tracker_.end()
                  , key_tracker_
                  , boost::get<2>(it->second)
                );

                // If connections to the locality are available in the cache,
                // remove the oldest one and return it.
                if (!boost::get<0>(it->second).empty())
                {
                    connection_type result = boost::get<0>(it->second).front();
                    boost::get<0>(it->second).pop_front();

                    ++hits_;
                    check_invariants();
                    return result;
                }
            }

            // If we get here then the item is not in the cache.
            ++misses_;
            check_invariants();
            return connection_type();
        }

        /// Try to get a connection to \a l from the cache, or reserve space for
        /// a new connection to \a l. This function may evict entries from the
        /// cache.
        ///
        /// \returns If a connection was found in the cache, its value is
        ///          assigned to \a conn and this function returns true. If a
        ///          connection was not found but space was reserved, \a conn is
        ///          set such that conn.get() == 0, and this function returns
        ///          true. If a connection could not be found and space could
        ///          not be returned, \a conn is unmodified and this function
        ///          returns false.
        ///
        /// \note    The connection must be returned to the cache by calling
        ///          \a reclaim().
        bool get_or_reserve(key_type const& l, connection_type& conn)
        {
            mutex_type::scoped_lock lock(mtx_);

            typename cache_type::iterator const it = cache_.find(l);

            // Check if this key already exists in the cache.
            if (it != cache_.end())
            {
                // Key exists in cache.

                // Update LRU meta data.
                key_tracker_.splice(
                    key_tracker_.end()
                  , key_tracker_
                  , boost::get<2>(it->second)
                );

                // If connections to the locality are available in the cache,
                // remove the oldest one and return it.
                if (!boost::get<0>(it->second).empty())
                {
                    conn = boost::get<0>(it->second).front();
                    boost::get<0>(it->second).pop_front();

                    ++hits_;
                    check_invariants();
                    return true;
                }

                // Otherwise, if we have less connections for this locality
                // than the maximum, try to reserve space in the cache for a new
                // connection.
                if (boost::get<1>(it->second) < max_connections_per_locality_)
                {
                    // See if we have enough space or can make space available.

                    // Note that if we don't have any space and there are no
                    // outstanding connections for this locality, we grow the
                    // cache size beyond its limit (hoping that it will be
                    // reduced in size next time some connection is handed back
                    // to the cache).

                    if (!free_space() && boost::get<1>(it->second) != 0)
                    {
                        // If we can't find or make space, give up.
                        ++misses_;
                        check_invariants();
                        return false;
                    }

                    // Make sure the input connection shared_ptr doesn't hold
                    // anything.
                    conn.reset();

                    // Increase the per-locality and overall connection counts.
                    ++boost::get<1>(it->second);
                    ++connections_;

                    ++insertions_;
                    check_invariants();
                    return true;
                }

                // We've reached the maximum number of connections for this
                // locality, and none of them are checked into the cache, so
                // we have to give up.
                ++misses_;
                check_invariants();
                return false;
            }

            // Key (locality) isn't in cache.

            // See if we have enough space or can make space available.

            // Note that we ignore the outcome of free_space() here as we have 
            // to guarantee to have space for the new connection as there are 
            // no connections outstanding for this locality. If free_space 
            // fails we grow the cache size beyond its limit (hoping that it 
            // will be reduced in size next time some connection is handed back
            // to the cache).
            free_space();

            // Update LRU meta data.
            typename key_tracker_type::iterator kt =
                key_tracker_.insert(key_tracker_.end(), l);

            cache_.insert(
                std::make_pair(l, boost::make_tuple(value_type(), 1, kt)));

            // Make sure the input connection shared_ptr doesn't hold anything.
            conn.reset();

            // Increase the overall connection counts.
            ++connections_;

            ++insertions_;
            check_invariants();
            return true;
        }

        /// Returns a connection for \a l to the cache.
        ///
        /// \note The cache must already be aware of the connection, through
        ///       a prior call to \a get() or \a get_or_reserve().
        void reclaim(key_type const& l, connection_type const& conn)
        {
            mutex_type::scoped_lock lock(mtx_);

            // Search for an entry for this key.
            typename cache_type::iterator const ct = cache_.find(l);

            if (ct != cache_.end()) {
                // Update LRU meta data.
                key_tracker_.splice(
                    key_tracker_.end()
                  , key_tracker_
                  , boost::get<2>(ct->second)
                    );

                // Add the connection to the entry.
                boost::get<0>(ct->second).push_back(conn);

                // FIXME: Again, this should probably throw instead of asserting,
                // as invariants could be invalidated here due to caller error.
                check_invariants();
            }
            else {
                // Key should already exist in the cache. FIXME: This should
                // probably throw as could easily be triggered by caller error.
                BOOST_ASSERT(shutting_down_);
            }
        }

        /// Returns true if the overall connection count is equal to or larger
        /// than the maximum number of overall connections, and false otherwise.
        bool full() const
        {
            mutex_type::scoped_lock lock(mtx_);
            return (connections_ >= max_connections_);
        }

        /// Returns true if the connection count for \a l is equal to or larger
        /// than the maximum connection count per locality, and false otherwise.
        bool full(key_type const& l) const
        {
            mutex_type::scoped_lock lock(mtx_);

            if (!cache_.count(l))
                return false || (connections_ >= max_connections_);

            typename cache_type::const_iterator ct = cache_.find(l);
            BOOST_ASSERT(ct != cache_.end());
            return (boost::get<1>(ct->second) >= max_connections_per_locality_)
                || (connections_ >= max_connections_);
        }

        /// Destroys all connections in the cache, and resets all counts.
        ///
        /// \note Calling this function while connections are still checked out
        ///       of the cache is a bad idea, and will violate this class'
        ///       invariants.
        void clear()
        {
            mutex_type::scoped_lock lock(mtx_);
            key_tracker_.clear();
            cache_.clear();
            connections_ = 0;

            insertions_ = 0;
            evictions_ = 0;
            hits_ = 0;
            misses_ = 0;

            // FIXME: This should probably throw instead of asserting, as it
            // can be triggered by caller error.
            check_invariants();
        }

        /// Destroys all connections for the give locality in the cache, reset
        /// all associated counts.
        ///
        /// \note Calling this function while connections are still checked out
        ///       of the cache is a bad idea, and will violate this classes
        ///       invariants.
        void clear(key_type const& l)
        {
            mutex_type::scoped_lock lock(mtx_);

            // Check if this key already exists in the cache.
            typename cache_type::iterator const it = cache_.find(l);
            if (it != cache_.end())
            {
                // Remove from LRU meta data.
                key_tracker_.erase(boost::get<2>(it->second));

                // correct counter to avoid assertions later on
                connections_ -= boost::get<1>(it->second);
                evictions_ += boost::get<1>(it->second);

                // Erase entry if key exists in the cache.
                cache_.erase(it);
            }

            // FIXME: This should probably throw instead of asserting, as it
            // can be triggered by caller error.
            check_invariants();
        }

        // access statistics
        boost::int64_t get_cache_insertions() const
        {
            mutex_type::scoped_lock lock(mtx_);
            return insertions_;
        }

        boost::int64_t get_cache_evictions() const
        {
            mutex_type::scoped_lock lock(mtx_);
            return evictions_;
        }

        boost::int64_t get_cache_hits() const
        {
            mutex_type::scoped_lock lock(mtx_);
            return hits_;
        }

        boost::int64_t get_cache_misses() const
        {
            mutex_type::scoped_lock lock(mtx_);
            return misses_;
        }

    private:
        /// Verify class invariants
        void check_invariants() const
        {
#if defined(HPX_DEBUG)
            typedef typename cache_type::const_iterator const_iterator;

            size_type in_cache_count = 0, total_count = 0;
            const_iterator end = cache_.end();
            for (const_iterator ct = cache_.begin(); ct != end; ++ct)
            {
                cache_value_type const& val = ct->second;

                // The separate item counter has to properly count all the
                // existing the elements, not only those in the cache entry.
                BOOST_ASSERT(boost::get<0>(val).size() <= boost::get<1>(val));

                // The overall number of connections in each entry (for each
                // locality) should not be larger than the allowed number.
                BOOST_ASSERT(boost::get<1>(val) <= max_connections_per_locality_);

                // Count all connections (both those in the cache and those
                // checked out of the cache).
                in_cache_count += boost::get<0>(val).size();
                total_count += boost::get<1>(val);
            }

            // Overall connection count should be larger than or equal to the
            // number of entries in the cache.
            BOOST_ASSERT(in_cache_count <= connections_);

            // Overall connection count should be equal to the sum of connection
            // counts for all localities.
            BOOST_ASSERT(total_count == connections_);

            // The list of key trackers should have the same size as the cache.
            BOOST_ASSERT(key_tracker_.size() == cache_.size());
#endif
        }

        /// Evict the least recently used removable entry from the cache if the
        /// cache is full.
        ///
        /// \returns Returns true if an entry was evicted or if the cache is not
        ///          full, and false if nothing could be evicted.
        bool free_space()
        {
            // If the cache isn't full, just return true.
            if (connections_ < max_connections_)
                return true;

            // Find the least recently used key.
            typename key_tracker_type::iterator kt = key_tracker_.begin();

            while (connections_ >= max_connections_)
            {
                // Find the least recent used keys data.
                const typename cache_type::iterator ct = cache_.find(*kt);
                BOOST_ASSERT(ct != cache_.end());

                // If the entry is empty, ignore it and try the next least
                // recently used entry.
                if (boost::get<0>(ct->second).empty())
                {
                    // Remove the key if its connection count is zero.
                    if (0 == boost::get<1>(ct->second)) {
                        cache_.erase(ct);
                        key_tracker_.erase(kt);
                        kt = key_tracker_.begin();
                    }
                    else {
                        // REVIEW: Should we reorder key_tracker_ to speed up
                        // the eviction?
                        ++kt;
                    }

                    // If we've gone through key_tracker_ and haven't found
                    // anything evictable, then all the entries must be
                    // currently checked out.
                    if (key_tracker_.end() == kt)
                        return false;

                    continue;
                }

                // Remove the oldest connection.
                boost::get<0>(ct->second).pop_front();

                // Adjust the overall and per-locality connection count.
                --boost::get<1>(ct->second);
                --connections_;

                ++evictions_;
            }

            return true;
        }

        mutable mutex_type mtx_;
        size_type const max_connections_;
        size_type const max_connections_per_locality_;
        key_tracker_type key_tracker_;
        cache_type cache_;
        size_type connections_;
        bool shutting_down_;

        // statistics support
        boost::int64_t insertions_;
        boost::int64_t evictions_;
        boost::int64_t hits_;
        boost::int64_t misses_;
    };
}}

#endif
