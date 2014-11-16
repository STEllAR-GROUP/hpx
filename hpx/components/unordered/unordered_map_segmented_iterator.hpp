//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http:// ww.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UNORDERED_MAP_SEGMENTED_ITERATOR_NOV_11_2014_0854PM)
#define HPX_UNORDERED_MAP_SEGMENTED_ITERATOR_NOV_11_2014_0854PM

/// \file hpx/components/unordered/unordered_map_segmented_iterator.hpp
///
/// This file contains the implementation of iterators for hpx::unordered_map.

 // The idea for these iterators is taken from
 // http://afstern.org/matt/segmented.pdf.

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/util.hpp>

#include <hpx/traits/segemented_iterator_traits.hpp>
#include <hpx/components/unordered/partition_unordered_map_component.hpp>

#include <cstdint>
#include <iterator>

#include <boost/integer.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/filter_iterator.hpp>

#include <boost/serialization/serialization.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Key, typename T, typename Hash = std::hash<Key>,
        typename KeyEqual = std::equal_to<Key> >
    class partition_unordered_map;

    template <typename Key, typename T, typename Hash = std::hash<Key>,
        typename KeyEqual = std::equal_to<Key> >
    class unordered_map;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Key, typename T, typename Hash, typename KeyEqual>
        struct unordered_map_value_proxy
        {
            typedef hpx::unordered_map<Key, T, Hash, KeyEqual>
                unordered_map_type;

            unordered_map_value_proxy(unordered_map_type& m, Key const& key)
              : m_(m), key_(key)
            {}
            unordered_map_value_proxy(unordered_map_type& m, Key && key)
              : m_(m), key_(std::move(key))
            {}

            operator T() const
            {
                return m_.get_value_sync(key_);
            }

            template <typename T_>
            unordered_map_value_proxy& operator=(T_ && value)
            {
                m_.set_value_sync(key_, std::forward<T_>(value));
                return *this;
            }

            unordered_map_type& m_;
            Key key_;
        };
    }
}

#endif
