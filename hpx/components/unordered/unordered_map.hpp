//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/vector/vector.hpp

#if !defined(HPX_UNORDERED_MAP_NOV_11_2014_0852PM)
#define HPX_UNORDERED_MAP_NOV_11_2014_0852PM

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/components.hpp>

#include <hpx/components/unordered/unordered_map_segmented_iterator.hpp>
#include <hpx/components/unordered/partition_unordered_map_component.hpp>

#include <cstdint>
#include <memory>

#include <boost/cstdint.hpp>

/// The hpx::unordered_map and its API's are defined here.
///
/// The hpx::unordered_map is a segmented data structure which is a collection
/// of one or more hpx::partition_unordered_maps. The hpx::unordered_map stores
/// the global IDs of each hpx::partition_unordered_maps.

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// This is the unordered_map class which define hpx::unordered_map
    /// functionality.
    ///
    ///  This contains the client side implementation of the hpx::unordered_map.
    ///  This class defines the synchronous and asynchronous API's for each of
    ///  the exposed functionalities.
    ///
    template <typename Key, typename T, typename Hash, typename KeyEqual>
    class unordered_map
    {
    public:
        typedef std::allocator<T> allocator_type;

        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        typedef T value_type;
        typedef T reference;
        typedef T const const_reference;

#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 40700
        typedef typename std::allocator_traits<allocator_type>::pointer pointer;
        typedef typename std::allocator_traits<allocator_type>::const_pointer
            const_pointer;
#else
        typedef T* pointer;
        typedef T const* const_pointer;
#endif

    public:
        unordered_map()
        {}

        unordered_map(unordered_map const& rhs)
        {}
        unordered_map(unordered_map && rhs)
        {}

        ~unordered_map()
        {}

        unordered_map& operator=(unordered_map const& rhs)
        {
            return *this;
        }
        unordered_map& operator=(unordered_map && rhs)
        {
            return *this;
        }

        /// \brief Array subscript operator. This does not throw any exception.
        ///
        /// \param pos Position of the element in the vector [Note the first
        ///            position in the partition is 0]
        ///
        /// \return Returns the value of the element at position represented by
        ///         \a pos.
        ///
        /// \note The non-const version of is operator returns a proxy object
        ///       instead of a real reference to the element.
        ///
        detail::unordered_map_value_proxy<Key, T, Hash, KeyEqual>
        operator[](Key const& pos)
        {
            return detail::unordered_map_value_proxy<
                    Key, T, Hash, KeyEqual
                >(*this, pos);
        }
        T operator[](Key const& pos) const
        {
            return get_value_sync(pos);
        }

    };
}

#endif
