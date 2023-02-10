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
    /// \class fifo_entry fifo_entry.hpp hpx/cache/entries/fifo_entry.hpp
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
    class fifo_entry : public entry<Value, fifo_entry<Value>>
    {
    private:
        using base_type = entry<Value, fifo_entry<Value>>;
        using time_point = std::chrono::steady_clock::time_point;

    public:
        /// \brief Any cache entry has to be default constructible
        fifo_entry() = default;

        /// \brief Construct a new instance of a cache entry holding the given
        ///        value.
        explicit fifo_entry(Value const& val) noexcept(
            std::is_nothrow_constructible_v<base_type, Value const&>)
          : base_type(val)
        {
        }

        /// \brief Construct a new instance of a cache entry holding the given
        ///        value.
        explicit fifo_entry(Value&& val) noexcept
          : base_type(HPX_MOVE(val))
        {
        }

        /// \brief    The function \a insert is called by a cache whenever it
        ///           is about to be inserted into the cache.
        ///
        /// \note     This function is part of the CacheEntry concept
        ///
        /// \returns  This function should return \a true if the entry should
        ///           be added to the cache, otherwise it should return
        ///           \a false.
        constexpr bool insert()
        {
            insertion_time_ = std::chrono::steady_clock::now();
            return true;
        }

        [[nodiscard]] constexpr time_point const& get_creation_time()
            const noexcept
        {
            return insertion_time_;
        }

        /// \brief Compare the 'age' of two entries. An entry is 'older' than
        ///        another entry if it has been created earlier (FIFO).
        friend bool
        operator<(fifo_entry const& lhs, fifo_entry const& rhs) noexcept(
            noexcept(std::declval<time_point const&>() <
                std::declval<time_point const&>()))
        {
            return lhs.get_creation_time() < rhs.get_creation_time();
        }

    private:
        time_point insertion_time_;
    };
}    // namespace hpx::util::cache::entries
