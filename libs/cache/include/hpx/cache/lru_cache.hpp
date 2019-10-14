//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_CACHE_LRU_CACHE_HPP
#define HPX_UTIL_CACHE_LRU_CACHE_HPP

#include <hpx/config.hpp>
#include <hpx/cache/statistics/no_statistics.hpp>

#include <algorithm>
#include <cstddef>
#include <list>
#include <map>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace cache {
    ///////////////////////////////////////////////////////////////////////////
    /// \class lru_cache lru_cache.hpp hpx/cache/lru_cache.hpp
    ///
    /// \brief The \a lru_cache implements the basic functionality needed for
    ///        a local (non-distributed) LRU cache.
    ///
    /// \tparam Key           The type of the keys to use to identify the
    ///                       entries stored in the cache
    /// \tparam Entry         The type of the items to be held in the cache.
    /// \tparam Statistics    A (optional) type allowing to collect some basic
    ///                       statistics about the operation of the cache
    ///                       instance. The type must conform to the
    ///                       CacheStatistics concept. The default value is
    ///                       the type \a statistics#no_statistics which does
    ///                       not collect any numbers, but provides empty stubs
    ///                       allowing the code to compile.
    template <typename Key, typename Entry,
        typename Statistics = statistics::no_statistics>
    class lru_cache
    {
    public:
        typedef Key key_type;
        typedef Entry entry_type;
        typedef Statistics statistics_type;
        typedef std::pair<key_type, entry_type> entry_pair;
        typedef std::list<entry_pair> storage_type;
        typedef std::map<Key, typename storage_type::iterator> map_type;
        typedef std::size_t size_type;

    private:
        typedef typename statistics_type::update_on_exit update_on_exit;

    public:
        ///////////////////////////////////////////////////////////////////////
        /// \brief Construct an instance of a lru_cache.
        ///
        /// \param max_size   [in] The maximal size this cache is allowed to
        ///                   reach any time. The default is zero (no size
        ///                   limitation). The unit of this value is usually
        ///                   determined by the unit of the values returned by
        ///                   the entry's \a get_size function.
        ///
        lru_cache(size_type max_size = 0)
          : max_size_(max_size)
          , current_size_(0)
        {
        }

        lru_cache(lru_cache&& other)
          : max_size_(other.max_size_)
          , current_size_(0)
          , storage_(std::move(other.storage_))
          , map_(std::move(other.map_))
          , statistics_(std::move(other.statistics_))
        {
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Return current size of the cache.
        ///
        /// \returns The current size of this cache instance.
        size_type size() const
        {
            return current_size_;
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Access the maximum size the cache is allowed to grow to.
        ///
        /// \note       The unit of this value is usually determined by the
        ///             unit of the return values of the entry's function
        ///             \a entry#get_size.
        ///
        /// \returns    The maximum size this cache instance is currently
        ///             allowed to reach. If this number is zero the cache has
        ///             no limitation with regard to a maximum size.
        size_type capacity() const
        {
            return max_size_;
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Change the maximum size this cache can grow to
        ///
        /// \param max_size    [in] The new maximum size this cache will be
        ///             allowed to grow to.
        ///
        void reserve(size_type max_size)
        {
            if (max_size > max_size_)
            {
                max_size_ = max_size;
                return;
            }

            max_size_ = max_size;
            while (current_size_ > max_size_)
            {
                evict();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Check whether the cache currently holds an entry identified
        ///        by the given key
        ///
        /// \param k      [in] The key for the entry which should be looked up
        ///               in the cache.
        ///
        /// \note         This function does not call the entry's function
        ///               \a entry#touch. It just checks if the cache contains
        ///               an entry corresponding to the given key.
        ///
        /// \returns      This function returns \a true if the cache holds the
        ///               referenced entry, otherwise it returns \a false.
        bool holds_key(key_type const& key)
        {
            return map_.find(key) != map_.end();
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Get a specific entry identified by the given key.
        ///
        /// \param key     [in] The key for the entry which should be retrieved
        ///               from the cache.
        /// \param entry  [out] If the entry indexed by the key is found in the
        ///               cache this value on successful return will be a copy
        ///               of the corresponding entry.
        ///
        /// \note         The function will "touch" the entry and mark it as recently
        ///               used if the key was found in the cache.
        ///
        /// \returns      This function returns \a true if the cache holds the
        ///               referenced entry, otherwise it returns \a false.
        bool get_entry(
            key_type const& key, key_type& realkey, entry_type& entry)
        {
            update_on_exit update(statistics_, statistics::method_get_entry);

            auto it = map_.find(key);

            if (it == map_.end())
            {
                // Got miss
                statistics_.got_miss();    // update statistics
                return false;
            }

            touch(it->second);

            // update statistics
            statistics_.got_hit();

            // got hit
            realkey = it->first;
            entry = it->second->second;
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Get a specific entry identified by the given key.
        ///
        /// \param key     [in] The key for the entry which should be retrieved
        ///               from the cache.
        /// \param entry  [out] If the entry indexed by the key is found in the
        ///               cache this value on successful return will be a copy
        ///               of the corresponding entry.
        ///
        /// \note         The function will "touch" the entry and mark it as recently
        ///               used if the key was found in the cache.
        ///
        /// \returns      This function returns \a true if the cache holds the
        ///               referenced entry, otherwise it returns \a false.
        bool get_entry(key_type const& key, entry_type& entry)
        {
            key_type tmp;
            return get_entry(key, tmp, entry);
        }

        /// \brief Insert a new entry into this cache
        ///
        /// \param key    [in] The key for the entry which should be added to
        ///               the cache.
        /// \param entry  [in] The entry which should be added to the cache.
        ///
        /// \note         This function assumes that the entry is not in the
        ///               cache already. Inserting an already existing entry
        ///               is considered undefined behavior
        bool insert(key_type const& key, entry_type const& entry)
        {
            update_on_exit update(statistics_, statistics::method_insert_entry);
            if (map_.find(key) != map_.end())
            {
                return false;
            }

            insert_nonexist(key, entry);
            return true;
        }

        void insert_nonexist(key_type const& key, entry_type const& entry)
        {
            // insert ...
            storage_.push_front(entry_pair(key, entry));
            map_[key] = storage_.begin();
            ++current_size_;

            // update statistics
            statistics_.got_insertion();

            // Do we need to evict a cache entry?
            if (current_size_ > max_size_)
            {
                // evict an entry
                evict();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Update an existing element in this cache
        ///
        /// \param key    [in] The key for the value which should be updated in
        ///               the cache.
        /// \param entry  [in] The entry which should be used as a replacement
        ///               for the existing value in the cache. Any existing
        ///               cache entry is not changed except for its value.
        ///
        /// \note         The function will "touch" the entry and mark it as recently
        ///               used if the key was found in the cache.
        /// \note         The difference to the other overload of the \a insert
        ///               function is that this overload replaces the cached
        ///               value only, while the other overload replaces the
        ///               whole cache entry, updating the cache entry
        ///               properties.
        void update(key_type const& key, entry_type const& entry)
        {
            update_on_exit update(statistics_, statistics::method_update_entry);

            // Is it already in the cache?
            auto it = map_.find(key);
            if (it == map_.end())
            {
                statistics_.got_miss();    // update statistics
                // got miss
                update_on_exit update(
                    statistics_, statistics::method_insert_entry);
                insert_nonexist(key, entry);
                return;
            }

            // got hit!
            it->second->second = entry;
            touch(it->second);
            // update statistics
            statistics_.got_hit();
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Update an existing element in this cache
        ///
        /// \param key    [in] The key for the value which should be updated in
        ///               the cache.
        /// \param entry  [in] The value which should be used as a replacement
        ///               for the existing value in the cache. Any existing
        ///               cache entry is not changed except for its value.
        /// \param f      [in] A callable taking two arguments, \a k and the
        ///               key found in the cache (in that order). If \a f
        ///               returns true, then the update will continue. If \a f
        ///               returns false, then the update will not succeed.
        ///
        /// \note         The function will "touch" the entry and mark it as recently
        ///               used if the key was found in the cache.
        /// \note         The difference to the other overload of the \a insert
        ///               function is that this overload replaces the cached
        ///               value only, while the other overload replaces the
        ///               whole cache entry, updating the cache entry
        ///               properties.
        ///
        /// \returns      This function returns \a true if the entry has been
        ///               successfully updated, otherwise it returns \a false.
        ///               If the entry currently is not held by the cache it is
        ///               added and the return value reflects the outcome of
        ///               the corresponding insert operation.
        template <typename F>
        bool update_if(key_type const& key, entry_type const& entry, F&& f)
        {
            update_on_exit update(statistics_, statistics::method_update_entry);
            // Is it already in the cache?
            auto it = map_.find(key);
            if (it == map_.end())
            {
                // got miss
                statistics_.got_miss();    // update statistics
                update_on_exit update(
                    statistics_, statistics::method_insert_entry);
                insert_nonexist(key, entry);
                return true;
            }

            if (f(key, it->first))
                return false;

            // got hit!
            touch(it->second);
            it->second->second = entry;

            // update statistics
            statistics_.got_hit();

            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Remove stored entries from the cache for which the supplied
        ///        function object returns true.
        ///
        /// \param ep     [in] This parameter has to be a (unary) function
        ///               object. It is invoked for each of the entries
        ///               currently held in the cache. An entry is considered
        ///               for removal from the cache whenever the value
        ///               returned from this invocation is \a true.
        ///
        /// \returns      This function returns the overall size of the removed
        ///               entries (which is the sum of the values returned by
        ///               the \a entry#get_size functions of the removed
        ///               entries).
        template <typename Func>
        size_type erase(Func const& ep)
        {
            update_on_exit update(statistics_, statistics::method_erase_entry);

            size_type erased = 0;
            for (auto it = map_.begin(); it != map_.end();)
            {
                auto jt = it->second;
                if (ep(*jt))
                {
                    ++erased;

                    storage_.erase(jt);
                    it = map_.erase(it);

                    // update statistics
                    statistics_.got_eviction();
                }
                else
                {
                    ++it;
                }
            }

            return erased;
        }

        /// \brief Remove all stored entries from the cache
        ///
        /// \returns      This function returns the overall size of the removed
        ///               entries (which is the sum of the values returned by
        ///               the \a entry#get_size functions of the removed
        ///               entries).
        size_type erase()
        {
            std::size_t current_size = current_size_;
            clear();
            return current_size;
        }

        /// \brief Clear the cache
        ///
        /// Unconditionally removes all stored entries from the cache.
        size_type clear()
        {
            size_type erased = current_size_;
            current_size_ = 0;
            map_.clear();
            storage_.clear();
            return erased;
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Allow to access the embedded statistics instance
        ///
        /// \returns      This function returns a reference to the statistics
        ///               instance embedded inside this cache
        statistics_type const& get_statistics() const
        {
            return statistics_;
        }

        statistics_type& get_statistics()
        {
            return statistics_;
        }

    private:
        void touch(typename storage_type::iterator it)
        {
            storage_.splice(storage_.begin(), storage_, it);
        }

        void evict()
        {
            statistics_.got_eviction();
            map_.erase(storage_.back().first);
            storage_.pop_back();
            --current_size_;
        }

        size_type max_size_;
        size_type current_size_;

        storage_type storage_;
        map_type map_;

        statistics_type statistics_;
    };
}}}    // namespace hpx::util::cache

#endif
