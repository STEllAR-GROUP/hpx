//  Copyright (c) 2019 Austin McCartney
//  Copyright (c) 2019 Hartmut Kaiser
//  Copyright (c) 2019 Piotr Mikolajczyk
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TEST_ITER_SENT_APR_14_2019_1040PM)
#define HPX_TEST_ITER_SENT_APR_14_2019_1040PM

#include <iterator>
#include <cstddef>

template<typename Value>
struct Sentinel
{
};

template<typename Value, Value stopValue>
struct Iterator
{
    using difference_type = std::ptrdiff_t;
    using value_type = Value;
    using iterator_category = std::forward_iterator_tag;
    using pointer = Value const*;
    using reference = Value const&;

    explicit Iterator(Value initialState): state(initialState) {}

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

    friend bool operator==(Iterator i, Sentinel<Value>)
    {
        return i.state == stopValue;
    }
    friend bool operator==(Sentinel<Value>, Iterator i)
    {
        return i.state == stopValue;
    }

    bool operator!=(const Iterator& that) const
    {
        return this->state != that.state;
    }

    friend bool operator!=(Iterator i, Sentinel<Value>)
    {
        return i.state != stopValue;
    }
    friend bool operator!=(Sentinel<Value>, Iterator i)
    {
        return i.state != stopValue;
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


#endif
