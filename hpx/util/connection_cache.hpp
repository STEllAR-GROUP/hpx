//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Regex library
//  Copyright (c) 2004 John Maddock
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
    template <typename Connection, typename Key>
    class connection_cache
    {
    public:
        typedef boost::recursive_mutex mutex_type;
        typedef boost::shared_ptr<Connection> connection_type;
        typedef Key key_type;

        typedef std::pair<connection_type, std::pair<key_type const*, int> > value_type;
        typedef std::list<value_type> list_type;
        typedef typename list_type::iterator list_iterator;
        typedef typename list_type::size_type size_type;

        typedef std::multimap<key_type, std::pair<list_iterator, int> > map_type;
        typedef typename map_type::iterator map_iterator;
        typedef typename map_type::size_type map_size_type;

        connection_cache(size_type max_cache_size, char const* const logdest)
          : max_cache_size_(max_cache_size < 2 ? 2 : max_cache_size),
            logdest_(logdest), count_(0)
        {}

        connection_type get(key_type const& l)
        {
            mutex_type::scoped_lock lock(mtx_);

//            LHPX_(debug, logdest_) << "connection_cache: requesting: " << l;

            // see if the object is already in the cache:
            std::pair<map_iterator, map_iterator> mpos = index_.equal_range(l);
            if (mpos.first != mpos.second) {
            // We have a cached item, return it
//                LHPX_(debug, logdest_)
//                    << "connection_cache: reusing existing connection for: "
//                    << l;

                connection_type result(mpos.first->second.first->first);
                cont_.erase(mpos.first->second.first);
                index_.erase(mpos.first);
                return result;
            }

//            LHPX_(debug, logdest_)
//                << "connection_cache: no existing connection for: " << l;

            // if we get here then the item is not in the cache
            return connection_type();
        }

        void add(key_type const& l, connection_type conn)
        {
            mutex_type::scoped_lock lock(mtx_);

//            LHPX_(debug, logdest_)
//                << "connection_cache: returning connection to cache: " << l;

            // Add it to the list, and index it
            cont_.push_back(value_type(conn,
                std::make_pair((key_type const*)NULL, ++count_)));
            map_iterator it = index_.insert(
                std::make_pair(l, std::make_pair(--(cont_.end()), count_)));
            if (it == index_.end())
            {
                HPX_THROW_EXCEPTION(out_of_memory, "connection_cache::add",
                    "couldn't insert new item into connection cache");
            }
            cont_.back().second.first = &(it->first);

            map_size_type s = index_.size();
            if (s > max_cache_size_) {
            // We have too many items in the list, so we need to start popping them
            // off the back of the list
//                LHPX_(debug, logdest_)
//                    << "connection_cache: cache full, removing least recently "
//                       "used entries";

                list_iterator pos = cont_.begin();
                list_iterator last = cont_.end();
                while (pos != last && s > max_cache_size_) {
                    // now remove the items from our containers
                    list_iterator condemmed(pos);
                    ++pos;

//                    LHPX_(debug, logdest_)
//                        << "connection_cache: removing entry for: "
//                        << *(condemmed->second.first);

                    int generational_count = condemmed->second.second;
                    std::pair<map_iterator, map_iterator> mpos =
                        index_.equal_range(*(condemmed->second.first));
                    BOOST_ASSERT(mpos.first != mpos.second);

#if defined(HPX_DEBUG)
                    bool found = false;
#endif
                    for(/**/; mpos.first != mpos.second; ++mpos.first)
                    {
                        if (mpos.first->second.second == generational_count)
                        {
                            index_.erase(mpos.first);
#if defined(HPX_DEBUG)
                            found = true;
#endif
                            break;
                        }
                    }
#if defined(HPX_DEBUG)
                    BOOST_ASSERT(found);
#endif

                    cont_.erase(condemmed);
                    --s;
                }
            }
        }

        void clear()
        {
            mutex_type::scoped_lock lock(mtx_);
            index_.clear();
            cont_.clear();
        }

    private:
        mutex_type mtx_;
        size_type max_cache_size_;
        char const* const logdest_;
        list_type cont_;
        map_type index_;
        int count_;
    };

}}

#endif
