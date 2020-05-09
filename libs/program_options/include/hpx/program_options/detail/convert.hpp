// Copyright Vladimir Prus 2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/program_options/config.hpp>

#if defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
// hpxinspect:nodeprecatedinclude:boost/program_options/detail/convert.hpp

#include <boost/program_options/detail/convert.hpp>

namespace hpx { namespace program_options {

    using boost::from_8_bit;
    using boost::from_local_8_bit;
    using boost::from_utf8;
    using boost::to_8_bit;
    using boost::to_local_8_bit;
    using boost::to_utf8;
    using boost::program_options::to_internal;

}}    // namespace hpx::program_options

#else

#include <cstddef>
#include <cwchar>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpx { namespace program_options {

    /** Converts from local 8 bit encoding into wchar_t string using
        the specified locale facet. */
    HPX_EXPORT std::wstring from_8_bit(const std::string& s,
        const std::codecvt<wchar_t, char, std::mbstate_t>& cvt);

    /** Converts from wchar_t string into local 8 bit encoding into using
        the specified locale facet. */
    HPX_EXPORT std::string to_8_bit(const std::wstring& s,
        const std::codecvt<wchar_t, char, std::mbstate_t>& cvt);

    /** Converts 's', which is assumed to be in UTF8 encoding, into wide
        string. */
    HPX_EXPORT std::wstring from_utf8(const std::string& s);

    /** Converts wide string 's' into string in UTF8 encoding. */
    HPX_EXPORT std::string to_utf8(const std::wstring& s);

    /** Converts wide string 's' into local 8 bit encoding determined by
        the current locale. */
    HPX_EXPORT std::string to_local_8_bit(const std::wstring& s);

    /** Converts 's', which is assumed to be in local 8 bit encoding, into wide
        string. */
    HPX_EXPORT std::wstring from_local_8_bit(const std::string& s);

    /** Convert the input string into internal encoding used by
            program_options. Presence of this function allows to avoid
            specializing all methods which access input on wchar_t.
        */
    HPX_EXPORT std::string to_internal(const std::string&);
    /** @overload */
    HPX_EXPORT std::string to_internal(const std::wstring&);

    template <class T>
    std::vector<std::string> to_internal(const std::vector<T>& s)
    {
        std::vector<std::string> result;
        for (std::size_t i = 0; i < s.size(); ++i)
            result.push_back(to_internal(s[i]));
        return result;
    }

}}    // namespace hpx::program_options

#endif
