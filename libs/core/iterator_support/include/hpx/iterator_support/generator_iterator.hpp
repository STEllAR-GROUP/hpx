//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  This code is based on boost::iterators::generator_iterator
//  Copyright Jens Maurer 2001.

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/iterator_facade.hpp>

#include <functional>
#include <iterator>

namespace hpx { namespace util {

    template <typename Generator>
    class generator_iterator
      : public iterator_facade<generator_iterator<Generator>,
            typename Generator::result_type, std::forward_iterator_tag,
            typename Generator::result_type const&>
    {
        using base_type = iterator_facade<generator_iterator<Generator>,
            typename Generator::result_type, std::forward_iterator_tag,
            typename Generator::result_type const&>;

    public:
        generator_iterator() = default;

        generator_iterator(Generator* g)
          : m_g(g)
          , m_value((*m_g)())
        {
        }

        void increment()
        {
            m_value = (*m_g)();
        }

        typename Generator::result_type const& dereference() const
        {
            return m_value;
        }

        bool equal(generator_iterator const& y) const
        {
            return m_g == y.m_g && m_value == y.m_value;
        }

    private:
        Generator* m_g;
        typename Generator::result_type m_value;
    };

    template <typename Generator>
    inline generator_iterator<Generator> make_generator_iterator(Generator& gen)
    {
        using result_type = generator_iterator<Generator>;
        return result_type(&gen);
    }

#if defined(HPX_HAVE_CXX17_DEDUCTION_GUIDES)
    // clang-format produces inconsistent result between different versions
    // clang-format off
    template <typename Generator>
    generator_iterator(Generator*)->generator_iterator<Generator>;
    // clang-format on
#endif
}}    // namespace hpx::util
