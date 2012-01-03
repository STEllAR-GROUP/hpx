//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// The color_map data structure and the corresponding access functions below
// have been taken from the Boost Graph Library. We have modified those to make
// them usable in multi-threaded environments.
//
// Original copyrights:
// Copyright 1997, 1998, 1999, 2000 University of Notre Dame.
// Authors: Andrew Lumsdaine, Lie-Quan Lee, Jeremy G. Siek

#if !defined(HPX_EXAMPLE_BFS_CONCURRENT_BGL_COLORMAP_JAN_02_2012_0736PM)
#define HPX_EXAMPLE_BFS_CONCURRENT_BGL_COLORMAP_JAN_02_2012_0736PM

#include <boost/graph/two_bit_color_map.hpp>
#include <boost/atomic.hpp>
#include <boost/shared_array.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace concurrent_bgl
{
    ///////////////////////////////////////////////////////////////////////////
    inline void init_atomic(boost::atomic<unsigned char>& d)
    {
        d.store(0);
    }

    template <typename IndexMap = boost::identity_property_map>
    struct color_map
    {
        std::size_t n;
        IndexMap index;
        boost::shared_array<boost::atomic<unsigned char> > data;

        typedef typename boost::property_traits<IndexMap>::key_type key_type;
        typedef boost::two_bit_color_type value_type;
        typedef void reference;
        typedef boost::read_write_property_map_tag category;

        explicit color_map(std::size_t n, IndexMap const& index = IndexMap())
          : n(n), index(index), data(new boost::atomic<unsigned char>[n])
        {
            // Fill to white
            std::for_each(data.get(), data.get() + n, init_atomic);
        }
    };

    template <typename IndexMap>
    inline boost::two_bit_color_type get(color_map<IndexMap> const& pm,
        typename boost::property_traits<IndexMap>::key_type key,
        boost::memory_order order = boost::memory_order_seq_cst)
    {
        typename boost::property_traits<IndexMap>::value_type i = get(pm.index, key);
        BOOST_ASSERT ((std::size_t)i < pm.n);
        return boost::two_bit_color_type(pm.data[i].load(order));
    }

    template <typename IndexMap>
    inline void put(color_map<IndexMap> const& pm,
        typename boost::property_traits<IndexMap>::key_type key,
        boost::two_bit_color_type value,
        boost::memory_order order = boost::memory_order_seq_cst)
    {
        typename boost::property_traits<IndexMap>::value_type i = get(pm.index, key);
        pm.data[i].store(value, order);
    }

    template <typename IndexMap>
    inline bool cas(color_map<IndexMap> const& pm,
        typename boost::property_traits<IndexMap>::key_type key,
        boost::two_bit_color_type oldvalue,
        boost::two_bit_color_type value,
        boost::memory_order order = boost::memory_order_seq_cst)
    {
        typename boost::property_traits<IndexMap>::value_type i = get(pm.index, key);
        unsigned char old_value = oldvalue;
        return pm.data[i].compare_exchange_strong(old_value, value, order);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename IndexMap>
    inline color_map<IndexMap>
    make_color_map(std::size_t n, IndexMap const& index_map)
    {
        return color_map<IndexMap>(n, index_map);
    }
}

#endif

