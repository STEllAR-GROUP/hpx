//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CACHE_LFU_ENTRY_NOV_19_2008_0121PM)
#define HPX_UTIL_CACHE_LFU_ENTRY_NOV_19_2008_0121PM

#include <hpx/config.hpp>
#include <hpx/util/cache/entries/entry.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace cache { namespace entries
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class lfu_entry lfu_entry.hpp hpx/util/cache/entries/lfu_entry.hpp
    ///
    /// The \a lfu_entry type can be used to store arbitrary values in a cache.
    /// Using this type as the cache's entry type makes sure that the least
    /// frequently used entries are discarded from the cache first.
    ///
    /// \note The \a lfu_entry conforms to the CacheEntry concept.
    /// \note This type can be used to model a 'most frequently used' cache
    ///       policy if it is used with a std::greater as the caches'
    ///       UpdatePolicy (instead of the default std::less).
    ///
    /// \tparam Value     The data type to be stored in a cache. It has to be
    ///                   default constructible, copy constructible and
    ///                   less_than_comparable.
    ///
    template <typename Value>
    class lfu_entry : public entry<Value, lfu_entry<Value> >
    {
    private:
        typedef entry<Value, lfu_entry<Value> > base_type;

    public:
        /// \brief Any cache entry has to be default constructible
        lfu_entry()
          : ref_count_(0)
        {}

        /// \brief Construct a new instance of a cache entry holding the given
        ///        value.
        explicit lfu_entry(Value const& val)
          : base_type(val), ref_count_(0)
        {}

        /// \brief    The function \a touch is called by a cache holding this
        ///           instance whenever it has been requested (touched).
        ///
        /// In the case of the LFU entry we store the reference count tracking
        /// the number of times this entry has been requested. This which will
        /// be used to compare the age of an entry during the invocation of the
        /// operator<().
        ///
        /// \returns  This function should return true if the cache needs to
        ///           update it's internal heap. Usually this is needed if the
        ///           entry has been changed by touch() in a way influencing
        ///           the sort order as mandated by the cache's UpdatePolicy
        bool touch()
        {
            ++ref_count_;
            return true;
        }

        unsigned long const& get_access_count() const
        {
            return ref_count_;
        }

        /// \brief Compare the 'age' of two entries. An entry is 'older' than
        ///        another entry if it has been accessed less frequently (LFU).
        friend bool operator< (lfu_entry const& lhs, lfu_entry const& rhs)
        {
            return lhs.get_access_count() < rhs.get_access_count();
        }

    private:
        unsigned long ref_count_;
    };
}}}}

#endif
