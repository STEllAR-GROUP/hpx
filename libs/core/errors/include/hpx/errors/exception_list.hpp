//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file exception_list.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/thread_support/spinlock.hpp>

#include <cstddef>
#include <exception>
#include <list>
#include <mutex>
#include <string>
#include <system_error>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    /// The class exception_list is a container of exception_ptr objects
    /// parallel algorithms may use to communicate uncaught exceptions
    /// encountered during parallel execution to the caller of the algorithm.
    ///
    /// The type exception_list::const_iterator fulfills the requirements of
    /// a forward iterator.
    ///
    class HPX_CORE_EXPORT exception_list : public hpx::exception
    {
    private:
        using mutex_type = hpx::util::detail::spinlock;
        using exception_list_type = std::list<std::exception_ptr>;
        
        exception_list_type exceptions_;
        mutable mutex_type mtx_;

        // Adds an exception to the list without acquiring a lock
        void add_no_lock(std::exception_ptr const& e)
        {
            exceptions_.push_back(e);
        }

    public:
        using iterator = exception_list_type::const_iterator;

        ~exception_list() noexcept override = default;

        exception_list() = default;

        explicit exception_list(std::exception_ptr const& e)
        {
            add(e);
        }

        explicit exception_list(exception_list_type&& l)
        {
            std::lock_guard<mutex_type> lock(mtx_);
            exceptions_ = std::move(l);
        }

        exception_list(exception_list const& l) : hpx::exception(l)
        {
            std::lock_guard<mutex_type> lock(l.mtx_);
            exceptions_ = l.exceptions_;
        }

        exception_list(exception_list&& l) noexcept : hpx::exception(std::move(l))
        {
            std::lock_guard<mutex_type> l_lock(l.mtx_);
            exceptions_ = std::move(l.exceptions_);
        }

        exception_list& operator=(exception_list const& l)
        {
            if (this != &l) {
                std::lock_guard<mutex_type> this_lock(mtx_);
                std::lock_guard<mutex_type> l_lock(l.mtx_);
                exceptions_ = l.exceptions_;
            }
            return *this;
        }

        exception_list& operator=(exception_list&& l) noexcept
        {
            if (this != &l) {
                std::lock_guard<mutex_type> this_lock(mtx_);
                std::lock_guard<mutex_type> l_lock(l.mtx_);
                exceptions_ = std::move(l.exceptions_);
            }
            return *this;
        }

        // Adds an exception to the list in a thread-safe manner
        void add(std::exception_ptr const& e)
        {
            std::lock_guard<mutex_type> lock(mtx_);
            add_no_lock(e);
        }

        /// Returns the number of exception_ptr objects in the exception_list.
        /// Complexity: Constant time.
        [[nodiscard]] std::size_t size() const noexcept
        {
            std::lock_guard<mutex_type> lock(mtx_);
            return exceptions_.size();
        }

        /// Returns an iterator referring to the first exception_ptr object.
        [[nodiscard]] exception_list_type::const_iterator begin() const noexcept
        {
            std::lock_guard<mutex_type> lock(mtx_);
            return exceptions_.begin();
        }

        /// Returns a past-the-end iterator for the exception_list.
        [[nodiscard]] exception_list_type::const_iterator end() const noexcept
        {
            std::lock_guard<mutex_type> lock(mtx_);
            return exceptions_.end();
        }

        [[nodiscard]] std::error_code get_error_code() const
        {
            return std::error_code(); // Placeholder implementation
        }

        [[nodiscard]] std::string get_message() const
        {
            return "Exception occurred"; // Placeholder implementation
        }
    };
} // namespace hpx

#include <hpx/config/warnings_suffix.hpp>
