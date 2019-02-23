//  Copyright (c) 2019 Austin McCartney
//  Copyright (c) 2019 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #3641: Trouble with using ranges-v3 and hpx::parallel::reduce
// #3646: Parallel algorithms should accept iterator/sentinel pairs

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_reduce.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>

struct Sentinel
{
};

struct Iterator
{
    using difference_type = std::ptrdiff_t;
    using value_type = std::int64_t;
    using iterator_category = std::input_iterator_tag;
    using pointer = std::int64_t const*;
    using reference = std::int64_t const&;

    std::int64_t state;

    std::int64_t operator*() const
    {
        return this->state;
    }
    std::int64_t operator->() const = delete;

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

    std::int64_t operator[](difference_type n) const
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

    friend bool operator==(Iterator i, Sentinel)
    {
        return i.state == 100;
    }
    friend bool operator==(Sentinel, Iterator i)
    {
        return i.state == 100;
    }

    bool operator!=(const Iterator& that) const
    {
        return this->state != that.state;
    }

    friend bool operator!=(Iterator i, Sentinel)
    {
        return i.state != 100;
    }
    friend bool operator!=(Sentinel, Iterator i)
    {
        return i.state != 100;
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
};

int main()
{
    std::int64_t result = hpx::parallel::reduce(hpx::parallel::execution::seq,
        Iterator{0}, Sentinel{}, std::int64_t(0));

    HPX_TEST_EQ(result, std::int64_t(4950));

    result = hpx::parallel::reduce(hpx::parallel::execution::par,
        Iterator{0}, Sentinel{}, std::int64_t(0));

    HPX_TEST_EQ(result, std::int64_t(4950));

    return hpx::util::report_errors();
}
