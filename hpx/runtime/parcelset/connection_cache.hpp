//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Regex library
//  Copyright (c) 2004 John Maddock
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_CONNECTION_CACHE_MAY_20_0104PM)
#define HPX_PARCELSET_CONNECTION_CACHE_MAY_20_0104PM

#include <map>
#include <list>
#include <stdexcept>
#include <string>

#include <boost/asio.hpp>
#include <boost/assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

namespace hpx { namespace parcelset
{
    class parcelport_connection;
    
    ///////////////////////////////////////////////////////////////////////////
    class connection_cache
    {
    public:
        typedef boost::shared_ptr<parcelport_connection> connection_type;
        typedef boost::asio::ip::tcp::endpoint key_type;
        
        typedef std::pair<connection_type, key_type const*> value_type;
        typedef std::list<value_type> list_type;
        typedef list_type::iterator list_iterator;
        typedef list_type::size_type size_type;

        typedef std::map<key_type, list_iterator> map_type;
        typedef map_type::iterator map_iterator;
        typedef map_type::size_type map_size_type;

        connection_cache(size_type max_cache_size)
          : max_cache_size_(max_cache_size < 2 ? 2 : max_cache_size)
        {}
        
        connection_type get (key_type const& endp);
        void add (key_type const& endp, connection_type conn);
        
    private:
        boost::mutex mtx_;
        size_type max_cache_size_;
        list_type cont_;
        map_type index_;
    };

}}

#endif
