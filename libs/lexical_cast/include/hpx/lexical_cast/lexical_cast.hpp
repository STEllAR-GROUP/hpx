// Copyright Kevlin Henney, 2000-2005.
// Copyright Alexander Nasonov, 2006-2010.
// Copyright Antony Polukhin, 2011-2019.
// Copyright Agustin Berge, 2019.
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// what:  lexical_cast custom keyword cast
// who:   contributed by Kevlin Henney,
//        enhanced with contributions from Terje Slettebo,
//        with additional fixes and suggestions from Gennaro Prota,
//        Beman Dawes, Dave Abrahams, Daryle Walker, Peter Dimov,
//        Alexander Nasonov, Antony Polukhin, Justin Viiret, Michael Hofmann,
//        Cheng Yang, Matthew Bradbury, David W. Birdsall, Pavel Korzh and other Boosters
// when:  November 2000, March 2003, June 2005, June 2006, March 2011 - 2014

#ifndef HPX_LEXICAL_CAST_LEXICAL_CAST_INCLUDED
#define HPX_LEXICAL_CAST_LEXICAL_CAST_INCLUDED

#include <hpx/config.hpp>

#include <cstddef>

#include <hpx/lexical_cast/bad_lexical_cast.hpp>
#include <hpx/lexical_cast/detail/cstring_wrapper.hpp>
#include <hpx/lexical_cast/try_lexical_convert.hpp>

namespace hpx { namespace util {

    template <typename Target, typename Source>
    inline Target lexical_cast(const Source& arg)
    {
        Target result = Target();

        if (!detail::try_lexical_convert(arg, result))
        {
            detail::throw_bad_cast<Source, Target>();
        }

        return result;
    }

    template <typename Target>
    inline Target lexical_cast(const char* chars, std::size_t count)
    {
        return util::lexical_cast<Target>(
            detail::cstring_wrapper<char>{chars, count});
    }

}}    // namespace hpx::util

#endif    // HPX_LEXICAL_CAST_LEXICAL_CAST_INCLUDED
