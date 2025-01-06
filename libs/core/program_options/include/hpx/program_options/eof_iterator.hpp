//  Copyright Vladimir Prus 2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/program_options/config.hpp>
#include <hpx/iterator_support/iterator_facade.hpp>

#include <iterator>

namespace hpx { namespace program_options {

    /** The 'eof_iterator' class is useful for constructing forward iterators
        in cases where iterator extract data from some source and it's easy
        to detect 'eof' \-- i.e. the situation where there's no data. One
        apparent example is reading lines from a file.

        Implementing such iterators using 'iterator_facade' directly would
        require to create class with three core operation, a couple of
        constructors. When using 'eof_iterator', the derived class should define
        only one method to get new value, plus a couple of constructors.

        The basic idea is that iterator has 'eof' bit. Two iterators are equal
        only if both have their 'eof' bits set. The 'get' method either obtains
        the new value or sets the 'eof' bit.

        Specifically, derived class should define:

        1. A default constructor, which creates iterator with 'eof' bit set. The
        constructor body should call 'found_eof' method defined here.
        2. Some other constructor. It should initialize some 'data pointer' used
        in iterator operation and then call 'get'.
        3. The 'get' method. It should operate this way:
            - look at some 'data pointer' to see if new element is available;
              if not, it should call 'found_eof'.
            - extract new element and store it at location returned by the 'value'
               method.
            - advance the data pointer.

        Essentially, the 'get' method has the functionality of both 'increment'
        and 'dereference'. It's very good for the cases where data extraction
        implicitly moves data pointer, like for stream operation.
    */
    template <class Derived, class ValueType>
    class eof_iterator
      : public util::iterator_facade<Derived, ValueType const,
            std::forward_iterator_tag>
    {
    public:
        eof_iterator() = default;

    protected:    // interface for derived
        /** Returns the reference which should be used by derived
            class to store the next value. */
        ValueType& value() noexcept
        {
            return m_value;
        }

        /** Should be called by derived class to indicate that it can't
            produce next element. */
        void found_eof() noexcept
        {
            m_at_eof = true;
        }

    private:    // iterator core operations
        friend class hpx::util::iterator_core_access;

        void increment()
        {
            static_cast<Derived&>(*this).get();
        }

        [[nodiscard]] bool equal(eof_iterator const& other) const noexcept
        {
            return m_at_eof && other.m_at_eof;
        }

        ValueType const& dereference() const noexcept
        {
            return m_value;
        }

        bool m_at_eof = false;
        ValueType m_value;
    };
}}    // namespace hpx::program_options
