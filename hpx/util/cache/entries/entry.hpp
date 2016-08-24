//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CACHE_ENTRY_NOV_17_2008_1032AM)
#define HPX_UTIL_CACHE_ENTRY_NOV_17_2008_1032AM

#include <hpx/config.hpp>

#include <boost/operators.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace cache { namespace entries
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Value, typename Derived = void>
    class entry;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Value, typename Derived>
        struct derived
        {
            typedef Derived type;
        };

        template <typename Value>
        struct derived<Value, void>
        {
            typedef entry<Value> type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \class entry entry.hpp hpx/util/cache/entries/entry.hpp
    ///
    /// \tparam Value     The data type to be stored in a cache. It has to be
    ///                   default constructible, copy constructible and
    ///                   less_than_comparable.
    /// \tparam Derived   The (optional) type for which this type is used as a
    ///                   base class.
    ///
    template <typename Value, typename Derived>
    class entry
      : boost::less_than_comparable<
            typename detail::derived<Value, Derived>::type
        >
    {
    public:
        typedef Value value_type;

    public:
        /// \brief Any cache entry has to be default constructible
        entry()
        {}

        /// \brief Construct a new instance of a cache entry holding the given
        ///        value.
        explicit entry(value_type const& val)
          : value_(val)
        {}

        /// \brief    The function \a touch is called by a cache holding this
        ///           instance whenever it has been requested (touched).
        ///
        /// \note     It is possible to change the entry in a way influencing
        ///           the sort criteria mandated by the UpdatePolicy. In this
        ///           case the function should return \a true to indicate this
        ///           to the cache, forcing to reorder the cache entries.
        /// \note     This function is part of the CacheEntry concept
        ///
        /// \return   This function should return true if the cache needs to
        ///           update it's internal heap. Usually this is needed if the
        ///           entry has been changed by touch() in a way influencing
        ///           the sort order as mandated by the cache's UpdatePolicy
        bool touch()
        {
            return false;
        }

        /// \brief    The function \a insert is called by a cache whenever it
        ///           is about to be inserted into the cache.
        ///
        /// \note     This function is part of the CacheEntry concept
        ///
        /// \returns  This function should return \a true if the entry should
        ///           be added to the cache, otherwise it should return
        ///           \a false.
        bool insert()
        {
            return true;
        }

        /// \brief    The function \a remove is called by a cache holding this
        ///           instance whenever it is about to be removed from the
        ///           cache.
        ///
        /// \note     This function is part of the CacheEntry concept
        ///
        /// \returns  The return value can be used to avoid removing this
        ///           instance from the cache. If the value is \a true it is
        ///           ok to remove the entry, other wise it will stay in the
        ///           cache.
        bool remove()
        {
            return true;
        }

        /// \brief    Return the 'size' of this entry. By default the size of
        ///           each entry is just one (1), which is sensible if the
        ///           cache has a limit (capacity) measured in number of
        ///           entries.
        std::size_t get_size() const
        {
            return 1;
        }

        /// \brief    Forwarding operator< allowing to compare entries in stead
        ///           of the values.
        friend bool operator< (entry const& lhs, entry const& rhs)
        {
            return lhs.value_ < rhs.value_;
        }

        /// \brief Get a reference to the stored data value
        ///
        /// \note This function is part of the CacheEntry concept
        value_type& get() { return value_; }
        value_type const& get() const { return value_; }

    private:
        value_type value_;
    };
}}}}

#endif
