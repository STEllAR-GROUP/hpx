//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BOOST_CACHE_NO_STATISTICS_NOV_20_2008_1229PM)
#define BOOST_CACHE_NO_STATISTICS_NOV_20_2008_1229PM

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace cache { namespace statistics
{
    ///////////////////////////////////////////////////////////////////////////
    class no_statistics
    {
    public:
        /// \brief  The function \a got_hit will be called by a cache instance
        ///         whenever a entry got touched.
        void got_hit() {}

        /// \brief  The function \a got_miss will be called by a cache instance
        ///         whenever a requested entry has not been found in the cache.
        void got_miss() {}

        /// \brief  The function \a got_insertion will be called by a cache
        ///         instance whenever a new entry has been inserted.
        void got_insertion() {}

        /// \brief  The function \a got_eviction will be called by a cache
        ///         instance whenever an entry has been removed from the cache
        ///         because a new inserted entry let the cache grow beyond its
        ///         capacity.
        void got_eviction() {}

        /// \brief Reset all statistics
        void clear() {}
    };

}}}

#endif

