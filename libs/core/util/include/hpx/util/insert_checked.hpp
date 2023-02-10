////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <utility>

namespace hpx::util {

    ////////////////////////////////////////////////////////////////////////////////
    /// \brief  Helper function for writing predicates that test whether an std::map
    ///         insertion succeeded. This inline template function negates the need
    ///         to explicitly write the sometimes lengthy std::pair<Iterator, bool>
    ///         type.
    ///
    /// \param r  [in] The return value of a std::map insert operation.
    ///
    /// \returns  This function returns \b r.second.
    template <typename Iterator>
    constexpr bool insert_checked(std::pair<Iterator, bool> const& r) noexcept
    {
        return r.second;
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Helper function for writing predicates that test whether an std::map
    /// insertion succeeded. This inline template function negates the need to
    /// explicitly write the sometimes lengthy std::pair<Iterator, bool> type.
    ///
    /// \param r  [in] The return value of a std::map insert operation.
    ///
    /// \param r  [out] A reference to an Iterator, which is set to \b r.first.
    /// \param it [out] on exit, will hold the iterator referring to the
    ///        inserted element
    ///
    /// \returns  This function returns \b r.second.
    template <typename Iterator>
    bool insert_checked(std::pair<Iterator, bool> const& r, Iterator& it)
    {
        it = r.first;
        return r.second;
    }
}    // namespace hpx::util
