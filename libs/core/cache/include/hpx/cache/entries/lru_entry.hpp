//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/cache/entries/entry.hpp>

#include <chrono>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util::cache::entries {

    ///////////////////////////////////////////////////////////////////////////
    /// \class lru_entry lru_entry.hpp hpx/cache/entries/lru_entry.hpp
    ///
    /// The \a lru_entry type can be used to store arbitrary values in a cache.
    /// Using this type as the cache's entry type makes sure that the least
    /// recently used entries are discarded from the cache first.
    ///
    /// \note The \a lru_entry conforms to the CacheEntry concept.
    /// \note This type can be used to model a 'most recently used' cache
    ///       policy if it is used with a std::greater as the caches'
    ///       UpdatePolicy (instead of the default std::less).
    ///
    /// \tparam Value     The data type to be stored in a cache. It has to be
    ///                   default constructible, copy constructible and
    ///                   less_than_comparable.
    ///
    template <typename Value>
    class lru_entry : public entry<Value, lru_entry<Value>>
    {
    private:
        using base_type = entry<Value, lru_entry<Value>>;
        using time_point = std::chrono::steady_clock::time_point;

    public:
        /// \brief Any cache entry has to be default constructible
        lru_entry()
          : access_time_(std::chrono::steady_clock::now())
        {
        }

        /// \brief Construct a new instance of a cache entry holding the given
        ///        value.
        explicit lru_entry(Value const& val) noexcept(
            std::is_nothrow_constructible_v<base_type, Value const&>)
          : base_type(val)
          , access_time_(std::chrono::steady_clock::now())
        {
        }

        /// \brief Construct a new instance of a cache entry holding the given
        ///        value.
        explicit lru_entry(Value&& val) noexcept
          : base_type(HPX_MOVE(val))
          , access_time_(std::chrono::steady_clock::now())
        {
        }

        /// \brief    The function \a touch is called by a cache holding this
        ///           instance whenever it has been requested (touched).
        ///
        /// In the case of the LRU entry we store the time of the last access
        /// which will be used to compare the age of an entry during the
        /// invocation of the operator<().
        ///
        /// \returns  This function should return true if the cache needs to
        ///           update it's internal heap. Usually this is needed if the
        ///           entry has been changed by touch() in a way influencing
        ///           the sort order as mandated by the cache's UpdatePolicy
        bool touch()
        {
            access_time_ = std::chrono::steady_clock::now();
            return true;
        }

        /// \brief Returns the last access time of the entry.
        [[nodiscard]] constexpr time_point const& get_access_time()
            const noexcept
        {
            return access_time_;
        }

        /// \brief Compare the 'age' of two entries. An entry is 'older' than
        ///        another entry if it has been accessed less recently (LRU).
        friend bool
        operator<(lru_entry const& lhs, lru_entry const& rhs) noexcept(
            noexcept(std::declval<time_point const&>() <
                std::declval<time_point const&>()))
        {
            return lhs.get_access_time() > rhs.get_access_time();
        }

    private:
        time_point access_time_;
    };
}    // namespace hpx::util::cache::entries
