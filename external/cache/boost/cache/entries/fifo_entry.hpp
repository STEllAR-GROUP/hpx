//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BOOST_CACHE_FIFO_ENTRY_NOV_19_2008_0748PM)
#define BOOST_CACHE_FIFO_ENTRY_NOV_19_2008_0748PM

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/cache/entries/entry.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace cache { namespace entries
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class fifo_entry fifo_entry.hpp boost/cache/entries/fifo_entry.hpp
    ///
    /// The \a fifo_entry type can be used to store arbitrary values in a cache.
    /// Using this type as the cache's entry type makes sure that the least
    /// recently inserted entries are discarded from the cache first.
    ///
    /// \note The \a fifo_entry conforms to the CacheEntry concept.
    /// \note This type can be used to model a 'last in first out' cache
    ///       policy if it is used with a std::greater as the caches'
    ///       UpdatePolicy (instead of the default std::less).
    ///
    /// \tparam Value     The data type to be stored in a cache. It has to be
    ///                   default constructible, copy constructible and
    ///                   less_than_comparable.
    ///
    template <typename Value>
    class fifo_entry : public entry<Value, fifo_entry<Value> >
    {
    private:
        typedef entry<Value, fifo_entry<Value> > base_type;

    public:
        /// \brief Any cache entry has to be default constructible
        fifo_entry()
        {}

        /// \brief Construct a new instance of a cache entry holding the given
        ///        value.
        explicit fifo_entry(Value const& val)
          : base_type(val)
        {}

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
            insertion_time_ = boost::posix_time::microsec_clock::local_time();
            return true;
        }

        boost::posix_time::ptime const& get_creation_time() const
        {
            return insertion_time_;
        }

        /// \brief Compare the 'age' of two entries. An entry is 'older' than
        ///        another entry if it has been created earlier (FIFO).
        friend bool operator< (fifo_entry const& lhs, fifo_entry const& rhs)
        {
            return lhs.get_creation_time() < rhs.get_creation_time();
        }

    private:
        boost::posix_time::ptime insertion_time_;
    };

}}}

#endif
