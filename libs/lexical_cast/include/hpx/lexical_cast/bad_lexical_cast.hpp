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

#ifndef HPX_LEXICAL_CAST_BAD_LEXICAL_CAST_HPP
#define HPX_LEXICAL_CAST_BAD_LEXICAL_CAST_HPP

#include <hpx/config.hpp>

#include <boost/throw_exception.hpp>
#include <exception>
#include <typeinfo>

namespace hpx { namespace util {
    // exception used to indicate runtime lexical_cast failure
    class bad_lexical_cast : public std::bad_cast
    {
    public:
        bad_lexical_cast() noexcept
          : source(&typeid(void))
          , target(&typeid(void))
        {
        }

        virtual const char* what() const noexcept
        {
            return "bad lexical cast: "
                   "source type value could not be interpreted as target";
        }

        virtual ~bad_lexical_cast() noexcept {}

        bad_lexical_cast(const std::type_info& source_type_arg,
            const std::type_info& target_type_arg) noexcept
          : source(&source_type_arg)
          , target(&target_type_arg)
        {
        }

        const std::type_info& source_type() const noexcept
        {
            return *source;
        }

        const std::type_info& target_type() const noexcept
        {
            return *target;
        }

    private:
        const std::type_info* source;
        const std::type_info* target;
    };

    namespace detail {
        template <class S, class T>
        inline void throw_bad_cast()
        {
            boost::throw_exception(bad_lexical_cast(typeid(S), typeid(T)));
        }
    }    // namespace detail

}}    // namespace hpx::util

#endif    // HPX_LEXICAL_CAST_BAD_LEXICAL_CAST_HPP
