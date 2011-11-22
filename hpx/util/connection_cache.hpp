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

        typedef std::map<key_type, std::vector<connection_type> > map_type;
        typedef typename map_type::iterator iterator;
        typedef typename map_type::size_type size_type;

        connection_cache(size_type max_cache_size)
          : max_cache_size_(max_cache_size < 2 ? 2 : max_cache_size)
        {}

        connection_type get(key_type const& l)
        {
            mutex_type::scoped_lock lock(mtx_);

            if (cache_.count(l) && cache_[l].size())
            {
                connection_type result = cache_[l].back();
                cache_[l].pop_back();
                --cache_size_;
                return result;
            }

            // if we get here then the item is not in the cache
            return connection_type();
        }

        void add(key_type const& l, connection_type const& conn)
        {
            mutex_type::scoped_lock lock(mtx_);

            if (cache_size_ < max_cache_size_)
            {
                cache_[l].push_back(conn);
                ++cache_size_;
            }
        }

        void clear()
        {
            mutex_type::scoped_lock lock(mtx_);
            cache_.clear();
            cache_size_ = 0; 
        }

    private:
        mutex_type mtx_;
        size_type const max_cache_size_;
        map_type cache_;
        size_type cache_size_;
    };

}}

#endif
