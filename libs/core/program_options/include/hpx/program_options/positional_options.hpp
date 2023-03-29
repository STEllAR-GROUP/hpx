//  Copyright Vladimir Prus 2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/program_options/config.hpp>

#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::program_options {

    /** Describes positional options.

        The class allows to guess option names for positional options, which
        are specified on the command line and are identified by the position.
        The class uses the information provided by the user to associate a name
        with every positional option, or tell that no name is known.

        The primary assumption is that only the relative order of the
        positional options themselves matters, and that any interleaving
        ordinary options don't affect interpretation of positional options.

        The user initializes the class by specifying that first N positional
        options should be given the name X1, following M options should be given
        the name X2 and so on.
    */
    class HPX_CORE_EXPORT positional_options_description
    {
    public:
        positional_options_description() = default;

        /** Species that up to 'max_count' next positional options
            should be given the 'name'. The value of '-1' means 'unlimited'.
            No calls to 'add' can be made after call with 'max_value' equal to
            '-1'.
        */
        positional_options_description& add(char const* name, int max_count);

        /** Returns the maximum number of positional options that can
            be present. Can return (numeric_limits<unsigned>::max)() to
            indicate unlimited number. */
        unsigned max_total_count() const noexcept;

        /** Returns the name that should be associated with positional
            options at 'position'.
            Precondition: position < max_total_count()
        */
        std::string const& name_for_position(unsigned position) const noexcept;

    private:
        // List of names corresponding to the positions. If the number of
        // positions is unlimited, then the last name is stored in
        // m_trailing;
        std::vector<std::string> m_names;
        std::string m_trailing;
    };
}    // namespace hpx::program_options

#include <hpx/config/warnings_suffix.hpp>
