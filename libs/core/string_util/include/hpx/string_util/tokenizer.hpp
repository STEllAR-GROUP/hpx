//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// (c) Copyright Jeremy Siek and John R. Bandela 2001.

// See http://www.boost.org/libs/tokenizer for documentation

// Revision History:
// 03 Jul 2003   John Bandela
//      Converted to new iterator adapter
// 02 Feb 2002   Jeremy Siek
//      Removed tabs and a little cleanup.

#pragma once

#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/string_util/token_iterator.hpp>

#include <string>
#include <type_traits>

namespace hpx::string_util {

    //===========================================================================
    // A container-view of a tokenized "sequence"
    template <typename TokenizerFunc = char_separator<char>,
        typename Iterator = std::string::const_iterator,
        typename Type = std::string>
    class tokenizer
    {
    private:
        using TGen = token_iterator_generator<TokenizerFunc, Iterator, Type>;

        // It seems that MSVC does not like the unqualified use of iterator,
        // Thus we use iter internally when it is used unqualified and
        // the users of this class will always qualify iterator.
        using iter = typename TGen::type;

    public:
        using iterator = iter;
        using const_iterator = iter;
        using value_type = Type;
        using reference = value_type&;
        using const_reference = value_type const&;
        using pointer = value_type*;
        using const_pointer = pointer const;
        using size_type = void;
        using difference_type = void;

        template <typename F = TokenizerFunc>
        tokenizer(Iterator first, Iterator last, F&& f = F())
          : first_(first)
          , last_(last)
          , f_(HPX_FORWARD(F, f))
        {
        }

        template <typename Container,
            typename = std::enable_if_t<traits::is_range_v<Container>>>
        explicit tokenizer(Container const& c)
          : first_(c.begin())
          , last_(c.end())
          , f_()
        {
        }

        template <typename F, typename Container,
            typename = std::enable_if_t<traits::is_range_v<Container>>>
        tokenizer(Container const& c, F&& f)
          : first_(c.begin())
          , last_(c.end())
          , f_(HPX_FORWARD(F, f))
        {
        }

        void assign(Iterator first, Iterator last)
        {
            first_ = first;
            last_ = last;
        }

        template <typename F>
        void assign(Iterator first, Iterator last, F&& f)
        {
            assign(first, last);
            f_ = HPX_FORWARD(F, f);
        }

        template <typename Container,
            typename = std::enable_if_t<traits::is_range_v<Container>>>
        void assign(Container const& c)
        {
            assign(c.begin(), c.end());
        }

        template <typename F, typename Container,
            typename = std::enable_if_t<traits::is_range_v<Container>>>
        void assign(Container const& c, F&& f)
        {
            assign(c.begin(), c.end(), HPX_FORWARD(F, f));
        }

        iter begin() const
        {
            return iter(f_, first_, last_);
        }

        iter end() const
        {
            return iter(f_, last_, last_);
        }

    private:
        Iterator first_;
        Iterator last_;
        TokenizerFunc f_;
    };

    // different clang-format versions disagree
    // clang-format off
    template <typename Iterator, typename F>
    tokenizer(Iterator, Iterator, F&&) -> tokenizer<std::decay_t<F>, Iterator,
        std::basic_string<typename std::iterator_traits<Iterator>::value_type>>;

    template <typename Container, typename F>
    tokenizer(Container const&, F&&)
        -> tokenizer<std::decay_t<F>, traits::range_iterator_t<Container const>,
            std::basic_string<typename std::iterator_traits<
                traits::range_iterator_t<Container>>::value_type>>;
    // clang-format on
}    // namespace hpx::string_util
