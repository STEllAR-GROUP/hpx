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
    /// encountered during parallel execution to the caller of the algorithm
    ///
    /// The type exception_list::const_iterator fulfills the requirements of
    /// a forward iterator.
    ///
    class HPX_CORE_EXPORT exception_list : public hpx::exception
    {
    private:
        /// \cond NOINTERNAL

        // TODO: Does this need to be hpx::spinlock?
        // typedef hpx::spinlock mutex_type;
        // TODO: Add correct initialization of hpx::util::detail spinlock.
        using mutex_type = hpx::util::detail::spinlock;

        using exception_list_type = std::list<std::exception_ptr>;
        exception_list_type exceptions_;
        mutable mutex_type mtx_;

        void add_no_lock(std::exception_ptr const& e);
        /// \endcond

    public:
        /// bidirectional iterator
        using iterator = exception_list_type::const_iterator;

        /// \cond NOINTERNAL
        // \throws nothing
        ~exception_list() noexcept override = default;

        exception_list();
        explicit exception_list(std::exception_ptr const& e);
        explicit exception_list(exception_list_type&& l);

        exception_list(exception_list const& l);
        exception_list(exception_list&& l) noexcept;

        exception_list& operator=(exception_list const& l);
        exception_list& operator=(exception_list&& l) noexcept;

        ///
        void add(std::exception_ptr const& e);
        /// \endcond

        /// The number of exception_ptr objects contained within the
        /// exception_list.
        ///
        /// \note Complexity: Constant time.
        [[nodiscard]] std::size_t size() const noexcept
        {
            std::lock_guard<mutex_type> l(mtx_);
            return exceptions_.size();
        }

        /// An iterator referring to the first exception_ptr object contained
        /// within the exception_list.
        [[nodiscard]] exception_list_type::const_iterator begin() const noexcept
        {
            std::lock_guard<mutex_type> l(mtx_);
            return exceptions_.begin();
        }

        /// An iterator which is the past-the-end value for the exception_list.
        [[nodiscard]] exception_list_type::const_iterator end() const noexcept
        {
            std::lock_guard<mutex_type> l(mtx_);
            return exceptions_.end();
        }

        /// \cond NOINTERNAL
        [[nodiscard]] std::error_code get_error_code() const;

        [[nodiscard]] std::string get_message() const;
        /// \endcond
    };
}    // namespace hpx

#include <hpx/config/warnings_suffix.hpp>
