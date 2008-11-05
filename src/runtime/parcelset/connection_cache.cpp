//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Regex library
//  Copyright (c) 2004 John Maddock
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/connection_cache.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/util/logging.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    connection_cache::connection_type 
    connection_cache::get (key_type const& l)
    {
        boost::mutex::scoped_lock lock(mtx_);

        LPT_(debug) << "connection_cache: requesting: " << l;

        // see if the object is already in the cache:
        std::pair<map_iterator, map_iterator> mpos = index_.equal_range(l);
        if (mpos.first != mpos.second) {
        // We have a cached item, return it
            LPT_(debug) << "connection_cache: reusing existing connection for: " 
                        << l;

            connection_type result(mpos.first->second->first);
            cont_.erase(mpos.first->second);
            index_.erase(mpos.first);
            return result;
        }

        LPT_(debug) << "connection_cache: no existing connection for: " << l;

        // if we get here then the item is not in the cache
        return connection_type();
    }
    
    ///////////////////////////////////////////////////////////////////////////
    void
    connection_cache::add (key_type const& l, connection_type conn)
    {
        boost::mutex::scoped_lock lock(mtx_);

        LPT_(debug) << "connection_cache: adding new connection for: " << l;

        // Add it to the list, and index it
        cont_.push_back(value_type(conn, NULL));
        index_.insert(std::make_pair(l, --(cont_.end())));
        cont_.back().second = &(index_.find(l)->first);

        map_size_type s = index_.size();
        if (s > max_cache_size_) {
        // We have too many items in the list, so we need to start popping them 
        // off the back of the list
            LPT_(debug) << "connection_cache: cache full, removing least "
                           "recently used entries";
            list_iterator pos = cont_.begin();
            list_iterator last = cont_.end();
            while (pos != last && s > max_cache_size_) {
                // now remove the items from our containers
                list_iterator condemmed(pos);
                ++pos;

                LPT_(debug) << "connection_cache: removing entry for: " 
                            << *(condemmed->second);

                index_.erase(*(condemmed->second));
                cont_.erase(condemmed); 
                --s;
            }
        }
    }

}}
