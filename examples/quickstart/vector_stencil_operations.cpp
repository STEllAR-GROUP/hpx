//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/iostreams.hpp>

#include <hpx/util/transform_iterator.hpp>

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct identity
{
    template <typename F>
    struct result;

    template <typename This, typename Iterator>
    struct result<This(Iterator)>
    {
        typedef T& type;
    };

    template <typename Iterator>
    T& operator()(Iterator it) const
    {
        return *it;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorBase, typename IteratorValue = IteratorBase>
struct previous
{
    typedef typename std::iterator_traits<IteratorBase>::reference reference;

    template <typename T>
    struct result;

    template <typename This, typename Iterator>
    struct result<This(Iterator)>
    {
        typedef typename std::iterator_traits<Iterator>::reference type;
    };

    previous() {}

    // at position 'begin' it will dereference 'value', otherwise 'it-1'
    previous(IteratorBase const& begin, IteratorValue const& value)
      : begin_(begin), value_(value)
    {}

    template <typename Iterator>
    typename std::iterator_traits<Iterator>::reference
    operator()(Iterator const& it) const
    {
        if (it == begin_)
            return *value_;
        return *(it - 1);
    }

    IteratorBase base() const { return begin_; }
    IteratorValue value() const { return value_; }

private:
    IteratorBase begin_;
    IteratorValue value_;
};

template <typename IteratorBase, typename IteratorValue>
inline previous<IteratorBase, IteratorValue>
make_previous_transformer(IteratorBase const& base, IteratorValue const& value)
{
    return previous<IteratorBase, IteratorValue>(base, value);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorBase, typename IteratorValue = IteratorBase>
struct next
{
    typedef typename std::iterator_traits<IteratorBase>::reference reference;

    template <typename T>
    struct result;

    template <typename This, typename Iterator>
    struct result<This(Iterator)>
    {
        typedef typename std::iterator_traits<Iterator>::reference type;
    };

    next() {}

    // at position 'end' it will dereference 'value', otherwise 'it+1'
    next(IteratorBase const& end, IteratorValue const& value)
      : end_(end), value_(value)
    {}

    template <typename Iterator>
    typename std::iterator_traits<Iterator>::reference
    operator()(Iterator const& it) const
    {
        if (it == end_)
            return *value_;
        return *(it + 1);
    }

    IteratorBase base() const { return end_; }
    IteratorValue value() const { return value_; }

private:
    IteratorBase end_;
    IteratorValue value_;
};

template <typename IteratorBase, typename IteratorValue>
inline next<IteratorBase, IteratorValue>
make_next_transformer(IteratorBase const& base, IteratorValue const& value)
{
    return next<IteratorBase, IteratorValue>(base, value);
}

///////////////////////////////////////////////////////////////////////////////
// print element to the left of current
template <typename Container>
void print_left_element(Container const& values)
{
    typedef typename Container::value_type value_type;

    auto transformer = make_previous_transformer(
        std::begin(values), &values.back());

    hpx::parallel::for_each(
        hpx::parallel::seq,
        hpx::util::make_transform_iterator(std::begin(values), transformer),
        hpx::util::make_transform_iterator(std::end(values), transformer),
        [](value_type d)
        {
            hpx::cout << d << " ";
        });
    hpx::cout << std::endl;
}

// print element to the right of current
template <typename Container>
void print_right_element(Container const& values)
{
    typedef typename Container::value_type value_type;

    auto transformer = make_next_transformer(
        std::end(values)-1, &values.front());

    hpx::parallel::for_each(
        hpx::parallel::seq,
        hpx::util::make_transform_iterator(std::begin(values), transformer),
        hpx::util::make_transform_iterator(std::end(values), transformer),
        [](value_type d)
        {
            hpx::cout << d << " ";
        });
    hpx::cout << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // demonstrate use of 'previous' and 'next' transformers
    {
        std::vector<double> values(10);
        std::iota(std::begin(values), std::end(values), 1.0);

        print_left_element(values);
        print_right_element(values);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    return hpx::init(argc, argv);
}
