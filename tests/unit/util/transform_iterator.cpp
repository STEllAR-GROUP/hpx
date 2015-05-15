//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/util/transform_iterator.hpp>

#include <sstream>
#include <iterator>
#include <vector>
#include <algorithm>

#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace test
{
    template <typename Iterator>
    BOOST_FORCEINLINE
    Iterator previous(Iterator it)
    {
        return --it;
    }

    template <typename Iterator>
    BOOST_FORCEINLINE
    Iterator next(Iterator it)
    {
        return ++it;
    }

    template <typename IteratorBase, typename IteratorValue>
    struct previous_transformer
    {
        template <typename T>
        struct result;

        template <typename This, typename Iterator>
        struct result<This(Iterator)>
        {
            typedef typename std::iterator_traits<Iterator>::reference type;
        };

        previous_transformer() {}

        // at position 'begin' it will dereference 'value', otherwise 'it-1'
        previous_transformer(IteratorBase const& begin, IteratorValue const& value)
            : begin_(begin), value_(value)
        {}

        template <typename Iterator>
        typename std::iterator_traits<Iterator>::reference
        operator()(Iterator const& it) const
        {
            if (it == begin_)
                return *value_;
            return *test::previous(it);
        }

    private:
        IteratorBase begin_;
        IteratorValue value_;
    };

    template <typename IteratorBase, typename IteratorValue>
    inline previous_transformer<IteratorBase, IteratorValue>
    make_previous_transformer(IteratorBase const& base, IteratorValue const& value)
    {
        return previous_transformer<IteratorBase, IteratorValue>(base, value);
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename IteratorBase, typename IteratorValue>
    struct next_transformer
    {
        template <typename T>
        struct result;

        template <typename This, typename Iterator>
        struct result<This(Iterator)>
        {
            typedef typename std::iterator_traits<Iterator>::reference type;
        };

        next_transformer() {}

        // at position 'end' it will dereference 'value', otherwise 'it+1'
        next_transformer(IteratorBase const& end, IteratorValue const& value)
            : end_(end), value_(value)
        {}

        template <typename Iterator>
        typename std::iterator_traits<Iterator>::reference
        operator()(Iterator const& it) const
        {
            if (it == end_)
                return *value_;
            return *test::next(it);
        }

    private:
        IteratorBase end_;
        IteratorValue value_;
    };

    template <typename IteratorBase, typename IteratorValue>
    inline next_transformer<IteratorBase, IteratorValue>
    make_next_transformer(IteratorBase const& base, IteratorValue const& value)
    {
        return next_transformer<IteratorBase, IteratorValue>(base, value);
    }
}

///////////////////////////////////////////////////////////////////////////////
// dereference element to the left of current
void test_left_element_full()
{
    // demonstrate use of 'previous' and 'next' transformers
    std::vector<int> values(10);
    std::iota(boost::begin(values), boost::end(values), 0);

    auto transformer = test::make_previous_transformer(
        boost::begin(values), &values.back());

    std::ostringstream str;

    std::for_each(
        hpx::util::make_transform_iterator(boost::begin(values), transformer),
        hpx::util::make_transform_iterator(boost::end(values), transformer),
        [&str](int d)
        {
            str << d << " ";
        });

    HPX_TEST_EQ(str.str(), std::string("9 0 1 2 3 4 5 6 7 8 "));
}

// dereference element to the right of current
void test_right_element_full()
{
    // demonstrate use of 'previous' and 'next' transformers
    std::vector<int> values(10);
    std::iota(boost::begin(values), boost::end(values), 0);

    auto transformer = test::make_next_transformer(
        boost::end(values)-1, &values.front());

    std::ostringstream str;

    std::for_each(
        hpx::util::make_transform_iterator(boost::begin(values), transformer),
        hpx::util::make_transform_iterator(boost::end(values), transformer),
        [&str](int d)
        {
            str << d << " ";
        });

    HPX_TEST_EQ(str.str(), std::string("1 2 3 4 5 6 7 8 9 0 "));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_left_element_full();
    test_right_element_full();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
