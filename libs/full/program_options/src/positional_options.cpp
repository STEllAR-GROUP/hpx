// Copyright Vladimir Prus 2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/program_options/config.hpp>

#if !defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
#include <hpx/assert.hpp>
#include <hpx/program_options/positional_options.hpp>

#include <cstddef>
#include <limits>
#include <string>

namespace hpx { namespace program_options {

    positional_options_description::positional_options_description() {}

    positional_options_description& positional_options_description::add(
        const char* name, int max_count)
    {
        HPX_ASSERT(max_count != -1 || m_trailing.empty());

        if (max_count == -1)
            m_trailing = name;
        else
        {
            m_names.resize(
                m_names.size() + static_cast<std::size_t>(max_count), name);
        }
        return *this;
    }

    unsigned positional_options_description::max_total_count() const
    {
        return m_trailing.empty() ? static_cast<unsigned>(m_names.size()) :
                                    (std::numeric_limits<unsigned>::max)();
    }

    const std::string& positional_options_description::name_for_position(
        unsigned position) const
    {
        HPX_ASSERT(position < max_total_count());

        if (static_cast<std::size_t>(position) < m_names.size())
            return m_names[static_cast<std::size_t>(position)];

        return m_trailing;
    }

}}    // namespace hpx::program_options

#endif
