// Copyright Vladimir Prus 2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/program_options/config.hpp>

#if !defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
#include <hpx/program_options/detail/convert.hpp>
#include <hpx/program_options/detail/utf8_codecvt_facet.hpp>

#include <fstream>
#include <functional>
#include <iostream>
#include <locale.h>
#include <locale>
#include <stdexcept>
#include <string>

namespace hpx { namespace program_options { namespace detail {

    using namespace std;

    /* Internal function to actually perform conversion.
       The logic in from_8_bit and to_8_bit function is exactly
       the same, except that one calls 'in' method of codecvt and another
       calls the 'out' method, and that syntax difference makes straightforward
       template implementation impossible.

       This functions takes a 'fun' argument, which should have the same
       parameters and return type and the in/out methods. The actual converting
       function will pass functional objects created with std::bind.
       Experiments show that the performance loss is less than 10%.
    */
    template <class ToChar, class FromChar, class Fun>
    std::basic_string<ToChar> convert(
        const std::basic_string<FromChar>& s, Fun fun)

    {
        std::basic_string<ToChar> result;

        std::mbstate_t state = std::mbstate_t();

        const FromChar* from = s.data();
        const FromChar* from_end = s.data() + s.size();
        // The interface of cvt is not really iterator-like, and it's
        // not possible the tell the required output size without the conversion.
        // All we can is convert data by pieces.
        while (from != from_end)
        {
            // std::basic_string does not provide non-const pointers to the data,
            // so converting directly into string is not possible.
            ToChar buffer[32];

            ToChar* to_next = buffer;
            // Need variable because std::bind doesn't work with rvalues.
            ToChar* to_end = buffer + 32;
            std::codecvt_base::result r =
                fun(state, from, from_end, from, buffer, to_end, to_next);

            if (r == std::codecvt_base::error)
                throw std::logic_error("character conversion failed");
            // 'partial' is not an error, it just means not all source
            // characters were converted. However, we need to check that at
            // least one new target character was produced. If not, it means
            // the source data is incomplete, and since we don't have extra
            // data to add to source, it's error.
            if (to_next == buffer)
                throw std::logic_error("character conversion failed");

            // Add converted characters
            result.append(buffer, to_next);
        }

        return result;
    }
}}}    // namespace hpx::program_options::detail

namespace hpx { namespace program_options {

    std::wstring from_8_bit(const std::string& s,
        const std::codecvt<wchar_t, char, std::mbstate_t>& cvt)
    {
        using namespace std::placeholders;
        return detail::convert<wchar_t>(s,
            std::bind(&std::codecvt<wchar_t, char, std::mbstate_t>::in, &cvt,
                _1, _2, _3, _4, _5, _6, _7));
    }

    std::string to_8_bit(const std::wstring& s,
        const std::codecvt<wchar_t, char, std::mbstate_t>& cvt)
    {
        using namespace std::placeholders;
        return detail::convert<char>(s,
            std::bind(&std::codecvt<wchar_t, char, std::mbstate_t>::out, &cvt,
                _1, _2, _3, _4, _5, _6, _7));
    }

    namespace {
        hpx::program_options::detail::utf8_codecvt_facet utf8_facet;
    }

    std::wstring from_utf8(const std::string& s)
    {
        return from_8_bit(s, utf8_facet);
    }

    std::string to_utf8(const std::wstring& s)
    {
        return to_8_bit(s, utf8_facet);
    }

    std::wstring from_local_8_bit(const std::string& s)
    {
        using facet_type = std::codecvt<wchar_t, char, std::mbstate_t>;
        return from_8_bit(s, std::use_facet<facet_type>(std::locale()));
    }

    std::string to_local_8_bit(const std::wstring& s)
    {
        using facet_type = std::codecvt<wchar_t, char, std::mbstate_t>;
        return to_8_bit(s, std::use_facet<facet_type>(std::locale()));
    }

    std::string to_internal(const std::string& s)
    {
        return s;
    }

    std::string to_internal(const std::wstring& s)
    {
        return to_utf8(s);
    }
}}    // namespace hpx::program_options

#endif
