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

#if !defined(HPX_UTIL_CONNECTION_CACHE_MAY_20_0104PM)
#define HPX_UTIL_CONNECTION_CACHE_MAY_20_0104PM

#include <map>
#include <list>
#include <stdexcept>
#include <string>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>

#include <boost/asio/ip/tcp.hpp>
#include <boost/assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // This class implements an LRU cache to hold connections.
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
            size_type max_cache_size
          , size_type max_connections_per_locality
        )
          : max_cache_size_(max_cache_size < 2 ? 2 : max_cache_size)
          , max_connections_per_locality_(max_connections_per_locality)
          , cache_size_(0)
        {
            if(max_connections_per_locality_ > max_cache_size)
            {
                HPX_THROW_EXCEPTION(bad_parameter, "connection_cache ctor",
                    "max_connections_per_locality_ > max_cache_size");
            }
        }

        connection_type get(key_type const& l)
        {
            mutex_type::scoped_lock lock(mtx_);

            // Check if entry already exists ...
            const typename cache_type::iterator it = cache_.find(l);

            // If it does ...
            if(it != cache_.end())
            {
                // .. update LRU meta data
                key_tracker_.splice(
                    key_tracker_.end()
                  , key_tracker_
                  , boost::get<2>(it->second)
                );

                // remove the oldest from that prefix and return it.
                if(!boost::get<0>(it->second).empty())
                {
                    connection_type result = boost::get<0>(it->second).front();
                    boost::get<0>(it->second).pop_front();
                    --cache_size_;

                    // The following line has to be removed in order to fix
                    // the behavior of the connection cache. Sadly, this breaks
                    // the AGAS booststrap.
                    --boost::get<1>(it->second);

                    check_invariants();
                    return result;
                }
            }

            // If we get here then the item is not in the cache.
            check_invariants();
            return connection_type();
        }

        // returns true if insertion was successful, which means maximum cache
        // size has not been exceeded
        void add(key_type const& l, connection_type const& conn)
        {
            mutex_type::scoped_lock lock(mtx_);

            const typename cache_type::iterator jt = cache_.find(l);

            if (jt != cache_.end() && boost::get<1>(jt->second) >= max_connections_per_locality_)
            {
                key_tracker_.splice(
                    key_tracker_.end()
                  , key_tracker_
                  , boost::get<2>(jt->second)
                );

                check_invariants();
                return;
            }

            if(key_tracker_.empty())
            {
                BOOST_ASSERT(cache_.size() == 0);
            }
            // eviction strategy implemented here ...
            else
            {
                // If we reached maximum capacity, evict one entry ...
                // Find the least recently used key entry
                typename key_tracker_type::iterator it = key_tracker_.begin();

                while (cache_size_ >= max_cache_size_)
                {
                    // find it ...
                    const typename cache_type::iterator kt = cache_.find(*it);
                    BOOST_ASSERT(kt != cache_.end());

                    // remove it if it is empty, doesn't affect cache size
                    if(boost::get<0>(kt->second).empty())
                    {
                        cache_.erase(kt);
                        key_tracker_.erase(it);
                        it = key_tracker_.begin();
                        BOOST_ASSERT(it != key_tracker_.end());
                        continue;
                    }

                    // just remove the oldest connection.
                    boost::get<0>(kt->second).pop_front();
                    --boost::get<1>(kt->second);
                    --cache_size_;
                    break;
                }
            }

            // If we reach here, we can safely add a new entry ...
            if (jt == cache_.end())
            {
                // cache doesn't hold the key yet ... insert
                typename key_tracker_type::iterator it =
                    key_tracker_.insert(key_tracker_.end(), l);
                cache_.insert(
                    std::make_pair(
                        l
                      , boost::make_tuple(value_type(1, conn), 1, it)
                    )
                );
            }
            else
            {
                BOOST_ASSERT(!key_tracker_.empty());

                // add it to our vector of connections for this prefix
                boost::get<0>(jt->second).push_back(conn);
                ++boost::get<1>(jt->second);

                // updating LRU info ... this key has just been used.
                key_tracker_.splice(
                    key_tracker_.end()
                  , key_tracker_
                  , boost::get<2>(jt->second)
                );
            }
            ++cache_size_;

            check_invariants();
        }

        bool full() const
        {
            mutex_type::scoped_lock lock(mtx_);
            return (cache_size_ >= max_cache_size_);
        }

        bool full(key_type const & l) const
        {
            mutex_type::scoped_lock lock(mtx_);

            if (!cache_.count(l))
                return false;

            typename cache_type::const_iterator jt = cache_.find(l);
            BOOST_ASSERT(jt != cache_.end());
            return boost::get<1>(jt->second) >= max_connections_per_locality_;
        }

        bool contains(key_type const & l) const
        {
            mutex_type::scoped_lock lock(mtx_);

            if (!cache_.count(l))
                return false;

            typename cache_type::const_iterator jt = cache_.find(l);
            BOOST_ASSERT(jt != cache_.end());
            return boost::get<0>(jt->second).size();
        }

        void clear()
        {
            mutex_type::scoped_lock lock(mtx_);
            key_tracker_.clear();
            cache_.clear();
            cache_size_ = 0;

            check_invariants();
        }

    protected:
        // verify class invariants
        void check_invariants() const
        {
#if defined(HPX_DEBUG)
            typedef typename cache_type::const_iterator const_iterator;

            size_type count = 0;
            const_iterator end = cache_.end();
            for (const_iterator it = cache_.begin(); it != end; ++it)
            {
                cache_value_type const& val = it->second;

                // The separate item counter has to properly count all the
                // existing the elements, not only those in the cache entry.
                BOOST_ASSERT(boost::get<0>(val).size() <= boost::get<1>(val));

                // The overall number of connections in each entry (for each
                // locality) should  not be larger than the allowed number.
                BOOST_ASSERT(boost::get<1>(val) <= max_connections_per_locality_);

                // count overall cache entries
                count += boost::get<0>(val).size();
            }

            // Overall number of entries should match separate counter.
            BOOST_ASSERT(count == cache_size_);

            // check that we do not hold too many elements
            BOOST_ASSERT(cache_size_ <= max_cache_size_);

            // the list of key trackers should have the right size
            BOOST_ASSERT(key_tracker_.size() == cache_.size());
#endif
        }

    private:
        mutable mutex_type mtx_;
        size_type const max_cache_size_;
        size_type const max_connections_per_locality_;
        key_tracker_type key_tracker_;
        cache_type cache_;
        size_type cache_size_;
    };
}}

#endif
