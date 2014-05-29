//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_TEST_ITERATOR_MAY_29_2014_0110PM)
#define HPX_STL_TEST_ITERATOR_MAY_29_2014_0110PM

#include <boost/iterator/iterator_adaptor.hpp>

namespace test
{
    template <typename BaseIterator, typename IteratorTag>
    struct test_iterator
      : boost::iterator_adaptor<
            test_iterator<BaseIterator, IteratorTag>,
            BaseIterator, boost::use_default, IteratorTag>
    {
    private:
        typedef boost::iterator_adaptor<
            test_iterator<BaseIterator, IteratorTag>,
            BaseIterator, boost::use_default, IteratorTag>
        base_type;

    public:
        test_iterator() : base_type() {}
        test_iterator(BaseIterator base) : base_type(base) {};
    };
}

#endif
