/*=============================================================================
    Copyright (c) 2001-2011 Joel de Guzman
    Copyright (c) 2001-2022 Hartmut Kaiser
    Copyright (c)      2010 Bryce Lelbach

//  SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/

#pragma once

#include <hpx/config.hpp>

#include <boost/spirit/home/x3/core/parser.hpp>
#include <boost/spirit/home/x3/core/skip_over.hpp>
#include <boost/spirit/home/x3/support/traits/move_to.hpp>
#include <boost/spirit/home/x3/support/unused.hpp>

#include <string>
#include <type_traits>

namespace hpx::threads::detail {

    template <typename Char, typename Iterator>
    inline bool partial_string_parse(
        Char const* str, Iterator& first, Iterator const& last) noexcept
    {
        Iterator i = first;
        Char ch = *str;

        for (; !!ch; ++i)
        {
            if (i == last || (ch != *i))
            {
                if (i == first)
                    return false;
                break;
            }
            ch = *++str;
        }

        first = i;
        return true;
    }

    template <typename String, typename Iterator>
    inline bool partial_string_parse(
        String const& str, Iterator& first, Iterator const& last) noexcept
    {
        Iterator i = first;
        typename String::const_iterator stri = str.begin();
        typename String::const_iterator str_last = str.end();

        for (; stri != str_last; ++stri, ++i)
        {
            if (i == last || (*stri != *i))
            {
                if (i == first)
                    return false;
                break;
            }
        }

        first = i;
        return true;
    }

    template <typename Char, typename Iterator>
    inline bool partial_string_parse(Char const* uc_i, Char const* lc_i,
        Iterator& first, Iterator const& last) noexcept
    {
        Iterator i = first;

        for (; *uc_i && *lc_i; ++uc_i, ++lc_i, ++i)
        {
            if (i == last || ((*uc_i != *i) && (*lc_i != *i)))
            {
                if (i == first)
                    return false;
                break;
            }
        }

        first = i;
        return true;
    }

    template <typename String, typename Iterator>
    inline bool partial_string_parse(String const& ucstr, String const& lcstr,
        Iterator& first, Iterator const& last) noexcept
    {
        typename String::const_iterator uc_i = ucstr.begin();
        typename String::const_iterator uc_last = ucstr.end();
        typename String::const_iterator lc_i = lcstr.begin();
        Iterator i = first;

        for (; uc_i != uc_last; ++uc_i, ++lc_i, ++i)
        {
            if (i == last || ((*uc_i != *i) && (*lc_i != *i)))
            {
                if (i == first)
                    return false;
                break;
            }
        }

        first = i;
        return true;
    }

    template <typename String, typename Attribute>
    struct partlit_parser
      : boost::spirit::x3::parser<partlit_parser<String, Attribute>>
    {
        using attribute_type = Attribute;

        static constexpr bool const has_attribute =
            !std::is_same_v<boost::spirit::x3::unused_type, attribute_type>;

        constexpr partlit_parser(String const& str, Attribute const& value)
          : str{str}
          , value{value}
        {
        }

        template <typename Iterator, typename Context, typename Attribute_>
        bool parse(Iterator& first, Iterator const& last,
            Context const& context, boost::spirit::x3::unused_type,
            Attribute_& attr_) const
        {
            boost::spirit::x3::skip_over(first, last, context);
            if (partial_string_parse(str, first, last))
            {
                // "move" from const (still no copy_to as of 1.75.0)
                boost::spirit::x3::traits::move_to(value, attr_);
                return true;
            }
            return false;
        }

        String str;
        Attribute value;
    };

    struct partlit_gen
    {
        template <typename Char, typename Attribute>
        constexpr partlit_parser<Char const*, Attribute> operator()(
            Char const* str, Attribute const& value) const
        {
            return {str, value};
        }

        template <typename Char>
        constexpr partlit_parser<Char const*, boost::spirit::x3::unused_type>
        operator()(Char const* str) const
        {
            return {str, boost::spirit::x3::unused};
        }
    };

    inline constexpr partlit_gen const partlit{};
}    // namespace hpx::threads::detail
