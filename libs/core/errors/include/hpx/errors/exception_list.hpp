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
        using mutex_type = hpx::spinlock; // Use hpx::spinlock for thread safety
        using exception_list_type = std::list<std::exception_ptr>;
        
        exception_list_type exceptions_; // Holds the list of exceptions
        mutable mutex_type mtx_;         // Mutex for thread-safe access

        // Internal function to add an exception without locking
        void add_no_lock(std::exception_ptr const& e)
        {
            exceptions_.push_back(e);
        }

    public:
        /// Bidirectional iterator type for the exception list
        using iterator = exception_list_type::const_iterator;

        ~exception_list() noexcept override = default;

        // Default constructor
        exception_list() = default;

        // Constructor that adds an initial exception
        explicit exception_list(std::exception_ptr const& e)
        {
            add(e);
        }

        // Move constructor that transfers ownership of exceptions
        explicit exception_list(exception_list_type&& l)
        {
            std::lock_guard<mutex_type> lock(mtx_);
            exceptions_ = std::move(l);
        }

        // Copy constructor with thread-safe access
        exception_list(exception_list const& l)
        {
            std::lock_guard<mutex_type> lock(l.mtx_);
            exceptions_ = l.exceptions_;
        }

        // Move constructor with thread-safe access
        exception_list(exception_list&& l) noexcept
        {
            std::lock_guard<mutex_type> lock(l.mtx_);
            exceptions_ = std::move(l.exceptions_);
        }

        // Copy assignment operator with thread-safe access
        exception_list& operator=(exception_list const& l)
        {
            if (this != &l) { // Avoid self-assignment
                std::lock_guard<mutex_type> this_lock(mtx_);
                std::lock_guard<mutex_type> l_lock(l.mtx_);
                exceptions_ = l.exceptions_;
            }
            return *this;
        }

        // Move assignment operator with thread-safe access
        exception_list& operator=(exception_list&& l) noexcept
        {
            if (this != &l) { // Avoid self-assignment
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

        // Placeholder for retrieving an error code
        [[nodiscard]] std::error_code get_error_code() const
        {
            return std::error_code(); // Placeholder implementation
        }

        // Placeholder for retrieving an exception message
        [[nodiscard]] std::string get_message() const
        {
            return "Exception occurred"; // Placeholder implementation
        }
    };
} // namespace hpx

#include <hpx/config/warnings_suffix.hpp>
