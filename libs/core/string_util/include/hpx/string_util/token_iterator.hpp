//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Copyright John R. Bandela 2001

// See http://www.boost.org/libs/tokenizer for documentation.

// Revision History:
// 16 Jul 2003   John Bandela
//      Allowed conversions from convertible base iterators
// 03 Jul 2003   John Bandela
//      Converted to new iterator adapter

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/iterator_support/detail/minimum_category.hpp>
#include <hpx/iterator_support/iterator_facade.hpp>
#include <hpx/string_util/token_functions.hpp>

#include <string>
#include <type_traits>
#include <utility>

namespace hpx::string_util {

    template <typename TokenizerFunc, typename Iterator, typename Type>
    class token_iterator
      : public util::iterator_facade<
            token_iterator<TokenizerFunc, Iterator, Type>, Type,
            typename util::detail::minimum_category<std::forward_iterator_tag,
                typename std::iterator_traits<Iterator>::iterator_category>::
                type,
            Type const&>
    {
    private:
        friend class util::iterator_core_access;

        TokenizerFunc f_;
        Iterator begin_;
        Iterator end_;
        bool valid_ = false;
        Type tok_;

        void increment()
        {
            HPX_ASSERT(valid_);
            valid_ = f_(begin_, end_, tok_);
        }

        Type const& dereference() const noexcept
        {
            HPX_ASSERT(valid_);
            return tok_;
        }

        template <typename Other>
        [[nodiscard]] bool equal(Other const& a) const noexcept
        {
            return (a.valid_ && valid_) ?
                ((a.begin_ == begin_) && (a.end_ == end_)) :
                (a.valid_ == valid_);
        }

        void initialize()
        {
            if (valid_)
                return;

            f_.reset();
            valid_ = (begin_ != end_) ? f_(begin_, end_, tok_) : false;
        }

    public:
        token_iterator() = default;

        template <typename F>
        token_iterator(F&& f, Iterator begin, Iterator e = Iterator())
          : f_(HPX_FORWARD(F, f))
          , begin_(begin)
          , end_(e)
          , tok_()
        {
            initialize();
        }

        explicit token_iterator(Iterator begin, Iterator e = Iterator())
          : f_()
          , begin_(begin)
          , end_(e)
          , tok_()
        {
            initialize();
        }

        template <typename OtherIter,
            typename =
                std::enable_if_t<std::is_convertible_v<OtherIter, Iterator>>>
        explicit token_iterator(
            token_iterator<TokenizerFunc, OtherIter, Type> const& t)
          : f_(t.tokenizer_function())
          , begin_(t.base())
          , end_(t.end())
          , valid_(!t.at_end())
          , tok_(t.current_token())
        {
        }

        [[nodiscard]] Iterator base() const
        {
            return begin_;
        }

        [[nodiscard]] Iterator end() const
        {
            return end_;
        }

        [[nodiscard]] TokenizerFunc const& tokenizer_function() const noexcept
        {
            return f_;
        }

        [[nodiscard]] Type current_token() const
        {
            return tok_;
        }

        [[nodiscard]] bool at_end() const noexcept
        {
            return !valid_;
        }
    };

    template <typename TokenizerFunc = char_separator<char>,
        typename Iterator = std::string::const_iterator,
        typename Type = std::string>
    struct token_iterator_generator
    {
        using type = token_iterator<TokenizerFunc, Iterator, Type>;
    };

    // Type has to be first because it needs to be explicitly specified as there
    // is no way the function can deduce it.
    template <typename Type, typename Iterator, typename TokenizerFunc>
    typename token_iterator_generator<std::decay_t<TokenizerFunc>, Iterator,
        Type>::type
    make_token_iterator(Iterator begin, Iterator end, TokenizerFunc&& fun)
    {
        using iterator_type =
            typename token_iterator_generator<std::decay_t<TokenizerFunc>,
                Iterator, Type>::type;
        return iterator_type(HPX_FORWARD(TokenizerFunc, fun), begin, end);
    }
}    // namespace hpx::string_util
