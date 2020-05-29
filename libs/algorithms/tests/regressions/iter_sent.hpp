//  Copyright (c) 2019 Austin McCartney
//  Copyright (c) 2019 Hartmut Kaiser
//  Copyright (c) 2019 Piotr Mikolajczyk
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstddef>
#include <iterator>

template <typename ValueType>
struct Sentinel
{
    explicit Sentinel(ValueType stop_value)
      : stop(stop_value)
    {
    }

    ValueType get_stop() const
    {
        return this->stop;
    }

private:
    ValueType stop;
};

template <typename Value>
struct Iterator
{
    using difference_type = std::ptrdiff_t;
    using value_type = Value;
    using iterator_category = std::forward_iterator_tag;
    using pointer = Value const*;
    using reference = Value const&;

    explicit Iterator(Value initialState)
      : state(initialState)
    {
    }

    virtual Value operator*() const
    {
        return this->state;
    }

    virtual Value operator->() const = delete;

    Iterator& operator++()
    {
        ++(this->state);
        return *this;
    }

    Iterator operator++(int)
    {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    Iterator& operator--()
    {
        --(this->state);
        return *this;
    }

    Iterator operator--(int)
    {
        auto copy = *this;
        --(*this);
        return copy;
    }

    virtual Value operator[](difference_type n) const
    {
        return this->state + n;
    }

    Iterator& operator+=(difference_type n)
    {
        this->state += n;
        return *this;
    }

    Iterator operator+(difference_type n) const
    {
        Iterator copy = *this;
        return copy += n;
    }

    Iterator& operator-=(difference_type n)
    {
        this->state -= n;
        return *this;
    }

    Iterator operator-(difference_type n) const
    {
        Iterator copy = *this;
        return copy -= n;
    }

    bool operator==(const Iterator& that) const
    {
        return this->state == that.state;
    }

    friend bool operator==(Iterator i, Sentinel<Value> s)
    {
        return i.state == s.get_stop();
    }
    friend bool operator==(Sentinel<Value> s, Iterator i)
    {
        return i.state == s.get_stop();
    }

    bool operator!=(const Iterator& that) const
    {
        return this->state != that.state;
    }

    friend bool operator!=(Iterator i, Sentinel<Value> s)
    {
        return i.state != s.get_stop();
    }
    friend bool operator!=(Sentinel<Value> s, Iterator i)
    {
        return i.state != s.get_stop();
    }

    bool operator<(const Iterator& that) const
    {
        return this->state < that.state;
    }

    bool operator<=(const Iterator& that) const
    {
        return this->state <= that.state;
    }

    bool operator>(const Iterator& that) const
    {
        return this->state > that.state;
    }

    bool operator>=(const Iterator& that) const
    {
        return this->state >= that.state;
    }

protected:
    Value state;
};
