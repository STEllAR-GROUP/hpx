//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CACHE_SIZE_ENTRY_NOV_19_2008_0800M)
#define HPX_UTIL_CACHE_SIZE_ENTRY_NOV_19_2008_0800M

#include <hpx/config.hpp>
#include <hpx/util/cache/entries/entry.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace cache { namespace entries
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Value, typename Derived = void>
    class size_entry;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Value, typename Derived>
        struct size_derived
        {
            typedef Derived type;
        };

        template <typename Value>
        struct size_derived<Value, void>
        {
            typedef size_entry<Value> type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \class size_entry size_entry.hpp hpx/util/cache/entries/size_entry.hpp
    ///
    /// The \a size_entry type can be used to store values in a cache which
    /// have a size associated (such as files, etc.).
    /// Using this type as the cache's entry type makes sure that the entries
    /// with the biggest size are discarded from the cache first.
    ///
    /// \note The \a size_entry conforms to the CacheEntry concept.
    /// \note This type can be used to model a 'discard smallest first' cache
    ///       policy if it is used with a std::greater as the caches'
    ///       UpdatePolicy (instead of the default std::less).
    ///
    /// \tparam Value     The data type to be stored in a cache. It has to be
    ///                   default constructible, copy constructible and
    ///                   less_than_comparable.
    /// \tparam Derived   The (optional) type for which this type is used as a
    ///                   base class.
    ///
    template <typename Value, typename Derived>
    class size_entry
      : public entry<Value, typename detail::size_derived<Value, Derived>::type>
    {
    private:
        typedef typename detail::size_derived<Value, Derived>::type derived_type;
        typedef entry<Value, derived_type> base_type;

    public:
        /// \brief Any cache entry has to be default constructible
        size_entry()
          : size_(0)
        {}

        /// \brief Construct a new instance of a cache entry holding the given
        ///        value.
        explicit size_entry(Value const& val, std::size_t size)
          : base_type(val), size_(size)
        {}

        /// \brief    Return the 'size' of this entry.
        std::size_t get_size() const
        {
            return size_;
        }

        /// \brief Compare the 'age' of two entries. An entry is 'older' than
        ///        another entry if it has a bigger size.
        friend bool operator< (size_entry const& lhs, size_entry const& rhs)
        {
            return lhs.get_size() > rhs.get_size();
        }

    private:
        std::size_t size_;      // the 'size' of the entry
    };
}}}}

#endif
