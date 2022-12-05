//  Copyright (c) 2019 Austin McCartney
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/iterator_support/iterator_adaptor.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <forward_list>
#include <iterator>
#include <list>
#include <type_traits>
#include <vector>

namespace test {

    template <typename BaseIterator, typename IteratorTag>
    struct test_iterator
      : hpx::util::iterator_adaptor<test_iterator<BaseIterator, IteratorTag>,
            BaseIterator, void, IteratorTag>
    {
    private:
        typedef hpx::util::iterator_adaptor<
            test_iterator<BaseIterator, IteratorTag>, BaseIterator, void,
            IteratorTag>
            base_type;

    public:
        test_iterator()
          : base_type()
        {
        }
        explicit test_iterator(BaseIterator base)
          : base_type(base)
        {
        }
    };
}    // namespace test

template <typename T, typename = void>
struct has_nested_type : std::integral_constant<bool, false>
{
};

template <typename T>
struct has_nested_type<T, std::void_t<typename T::type>>
  : std::integral_constant<bool, true>
{
};

struct bidirectional_traversal_iterator
{
    using difference_type = int;
    using value_type = int;
    using iterator_category = std::input_iterator_tag;
    using pointer = const int*;
    using reference = void;

    int state;

    int operator*() const
    {
        return this->state;
    }
    int operator->() const = delete;

    bidirectional_traversal_iterator& operator++()
    {
        ++(this->state);
        return *this;
    }

    bidirectional_traversal_iterator operator++(int)
    {
        bidirectional_traversal_iterator copy = *this;
        ++(*this);
        return copy;
    }

    bidirectional_traversal_iterator& operator--()
    {
        --(this->state);
        return *this;
    }

    bidirectional_traversal_iterator operator--(int)
    {
        bidirectional_traversal_iterator copy = *this;
        --(*this);
        return copy;
    }

    bool operator==(const bidirectional_traversal_iterator& that) const
    {
        return this->state == that.state;
    }

    bool operator!=(const bidirectional_traversal_iterator& that) const
    {
        return this->state != that.state;
    }

    bool operator<(const bidirectional_traversal_iterator& that) const
    {
        return this->state < that.state;
    }

    bool operator<=(const bidirectional_traversal_iterator& that) const
    {
        return this->state <= that.state;
    }

    bool operator>(const bidirectional_traversal_iterator& that) const
    {
        return this->state > that.state;
    }

    bool operator>=(const bidirectional_traversal_iterator& that) const
    {
        return this->state >= that.state;
    }
};

struct random_access_traversal_iterator
{
    using difference_type = int;
    using value_type = int;
    using iterator_category = std::input_iterator_tag;
    using pointer = const int*;
    using reference = void;

    int state;

    int operator*() const
    {
        return this->state;
    }
    int operator->() const = delete;

    random_access_traversal_iterator& operator++()
    {
        ++(this->state);
        return *this;
    }

    random_access_traversal_iterator operator++(int)
    {
        random_access_traversal_iterator copy = *this;
        ++(*this);
        return copy;
    }

    random_access_traversal_iterator& operator--()
    {
        --(this->state);
        return *this;
    }

    random_access_traversal_iterator operator--(int)
    {
        random_access_traversal_iterator copy = *this;
        --(*this);
        return copy;
    }

    int operator[](difference_type n) const
    {
        return this->state + n;
    }

    random_access_traversal_iterator& operator+=(difference_type n)
    {
        this->state += n;
        return *this;
    }

    random_access_traversal_iterator operator+(difference_type n)
    {
        random_access_traversal_iterator copy = *this;
        return copy += n;
    }

    random_access_traversal_iterator& operator-=(difference_type n)
    {
        this->state -= n;
        return *this;
    }

    random_access_traversal_iterator operator-(difference_type n)
    {
        random_access_traversal_iterator copy = *this;
        return copy -= n;
    }

    difference_type operator-(
        const random_access_traversal_iterator& that) const
    {
        return this->state - that.state;
    }

    bool operator==(const random_access_traversal_iterator& that) const
    {
        return this->state == that.state;
    }

    bool operator!=(const random_access_traversal_iterator& that) const
    {
        return this->state != that.state;
    }

    bool operator<(const random_access_traversal_iterator& that) const
    {
        return this->state < that.state;
    }

    bool operator<=(const random_access_traversal_iterator& that) const
    {
        return this->state <= that.state;
    }

    bool operator>(const random_access_traversal_iterator& that) const
    {
        return this->state > that.state;
    }

    bool operator>=(const random_access_traversal_iterator& that) const
    {
        return this->state >= that.state;
    }
};

void addition_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator+(const A&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG((std::is_same_v<B, addition_result_t<C, A>>), "deduced type");

    HPX_TEST_MSG(
        (!has_nested_type<addition_result<B, A>>::value), "invalid operation");
}

void dereference_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };

    struct B
    {
        A operator*() const
        {
            return A{};
        }
    };

    HPX_TEST_MSG((std::is_same_v<A, dereference_result_t<B>>), "deduced type");

    HPX_TEST_MSG(
        (!has_nested_type<dereference_result<A>>::value), "invalid operation");
}

void equality_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator==(const A&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG((std::is_same_v<B, equality_result_t<C, A>>), "deduced type");

    HPX_TEST_MSG(
        (!has_nested_type<equality_result<B, A>>::value), "invalid operation");
}

void inequality_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator!=(const A&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG(
        (std::is_same_v<B, inequality_result_t<C, A>>), "deduced type");

    HPX_TEST_MSG((!has_nested_type<inequality_result<B, A>>::value),
        "invalid operation");
}

void inplace_addition_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator+=(const A&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG(
        (std::is_same_v<B, inplace_addition_result_t<C, A>>), "deduced type");

    HPX_TEST_MSG((!has_nested_type<inplace_addition_result<B, A>>::value),
        "invalid operation");
}

void inplace_subtraction_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator-=(const A&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG((std::is_same_v<B, inplace_subtraction_result_t<C, A>>),
        "deduced type");

    HPX_TEST_MSG((!has_nested_type<inplace_subtraction_result<B, A>>::value),
        "invalid operation");
}

void predecrement_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };

    struct B
    {
        A& operator--() const
        {
            static A a;
            return a;
        }
    };

    HPX_TEST_MSG(
        (std::is_same_v<A&, predecrement_result_t<B>>), "deduced type");

    HPX_TEST_MSG(
        (!has_nested_type<predecrement_result<A>>::value), "invalid operation");
}

void preincrement_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };

    struct B
    {
        A& operator++() const
        {
            static A a;
            return a;
        }
    };

    HPX_TEST_MSG(
        (std::is_same_v<A&, preincrement_result_t<B>>), "deduced type");

    HPX_TEST_MSG(
        (!has_nested_type<preincrement_result<A>>::value), "invalid operation");
}

void postdecrement_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };

    struct B
    {
        A operator--(int) const
        {
            return A{};
        }
    };

    HPX_TEST_MSG(
        (std::is_same_v<A, postdecrement_result_t<B>>), "deduced type");

    HPX_TEST_MSG((!has_nested_type<postdecrement_result<A>>::value),
        "invalid operation");
}

void postincrement_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };

    struct B
    {
        A operator++(int) const
        {
            return A{};
        }
    };

    HPX_TEST_MSG(
        (std::is_same_v<A, postincrement_result_t<B>>), "deduced type");

    HPX_TEST_MSG((!has_nested_type<postincrement_result<A>>::value),
        "invalid operation");
}

void subscript_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator[](const A&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG((std::is_same_v<B, subscript_result_t<C, A>>), "deduced type");

    HPX_TEST_MSG(
        (!has_nested_type<subscript_result<B, A>>::value), "invalid operation");
}

void subtraction_result_test()
{
    using namespace hpx::traits::detail;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator-(const A&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG(
        (std::is_same_v<B, subtraction_result_t<C, A>>), "deduced type");

    HPX_TEST_MSG((!has_nested_type<subtraction_result<B, A>>::value),
        "invalid operation");
}

void bidirectional_concept_test()
{
    using hpx::traits::detail::bidirectional_concept;

    {
        using iterator = std::ostream_iterator<int>;
        HPX_TEST_MSG(
            (!bidirectional_concept<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        HPX_TEST_MSG(
            (!bidirectional_concept<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        HPX_TEST_MSG(
            (!bidirectional_concept<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        HPX_TEST_MSG(
            (bidirectional_concept<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        HPX_TEST_MSG(
            (bidirectional_concept<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        HPX_TEST_MSG((bidirectional_concept<iterator>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        HPX_TEST_MSG((bidirectional_concept<iterator>::value),
            "random access traversal input iterator");
    }
}

void random_access_concept_test()
{
    using namespace hpx::traits;
    using namespace hpx::traits::detail;

    {
        using iterator = std::ostream_iterator<int>;
        HPX_TEST_MSG(
            (!random_access_concept<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        HPX_TEST_MSG(
            (!random_access_concept<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        HPX_TEST_MSG(
            (!random_access_concept<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        HPX_TEST_MSG((!random_access_concept<iterator>::value),
            "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        HPX_TEST_MSG(
            (random_access_concept<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        HPX_TEST_MSG((!random_access_concept<iterator>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;

        using namespace hpx::traits::detail;
        static_assert(std::is_same_v<iterator,
            addition_result_t<iterator, iter_difference_t<iterator>>>);
        static_assert(std::is_same_v<
            typename std::add_lvalue_reference<iterator>::type,
            inplace_addition_result_t<iterator, iter_difference_t<iterator>>>);
        static_assert(std::is_same_v<iterator,
            subtraction_result_t<iterator, iter_difference_t<iterator>>>);
        static_assert(std::is_same_v<iter_difference_t<iterator>,
            subtraction_result_t<iterator, iterator>>);
        static_assert(std::is_same_v<std::add_lvalue_reference_t<iterator>,
            inplace_subtraction_result_t<iterator,
                iter_difference_t<iterator>>>);

        HPX_TEST_MSG((random_access_concept<iterator>::value),
            "random access traversal input iterator");
    }
}

void satisfy_traversal_concept_forward_test()
{
    using namespace hpx::traits::detail;

    {
        using iterator = std::ostream_iterator<int>;
        HPX_TEST_MSG((!satisfy_traversal_concept<iterator,
                         hpx::forward_traversal_tag>::value),
            "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        HPX_TEST_MSG((!satisfy_traversal_concept<iterator,
                         hpx::forward_traversal_tag>::value),
            "input iterator");
    }
    {
        // see comment on definition for explanation
        using iterator = typename std::forward_list<int>::iterator;
        HPX_TEST_MSG((!satisfy_traversal_concept<iterator,
                         hpx::forward_traversal_tag>::value),
            "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        HPX_TEST_MSG((satisfy_traversal_concept<iterator,
                         hpx::forward_traversal_tag>::value),
            "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        HPX_TEST_MSG((satisfy_traversal_concept<iterator,
                         hpx::forward_traversal_tag>::value),
            "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        HPX_TEST_MSG((satisfy_traversal_concept<iterator,
                         hpx::forward_traversal_tag>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        HPX_TEST_MSG((satisfy_traversal_concept<iterator,
                         hpx::forward_traversal_tag>::value),
            "random access traversal input iterator");
    }
}

void satisfy_traversal_concept_bidirectional_test()
{
    using namespace hpx::traits::detail;

    {
        using iterator = std::ostream_iterator<int>;
        HPX_TEST_MSG((!satisfy_traversal_concept<iterator,
                         hpx::bidirectional_traversal_tag>::value),
            "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        HPX_TEST_MSG((!satisfy_traversal_concept<iterator,
                         hpx::bidirectional_traversal_tag>::value),
            "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        HPX_TEST_MSG((!satisfy_traversal_concept<iterator,
                         hpx::bidirectional_traversal_tag>::value),
            "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        HPX_TEST_MSG((satisfy_traversal_concept<iterator,
                         hpx::bidirectional_traversal_tag>::value),
            "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        HPX_TEST_MSG((satisfy_traversal_concept<iterator,
                         hpx::bidirectional_traversal_tag>::value),
            "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        HPX_TEST_MSG((satisfy_traversal_concept<iterator,
                         hpx::bidirectional_traversal_tag>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        HPX_TEST_MSG((satisfy_traversal_concept<iterator,
                         hpx::bidirectional_traversal_tag>::value),
            "random access traversal input iterator");
    }
}

void satisfy_traversal_concept_random_access_test()
{
    using namespace hpx::traits::detail;

    {
        using iterator = std::ostream_iterator<int>;
        HPX_TEST_MSG((!satisfy_traversal_concept<iterator,
                         hpx::random_access_traversal_tag>::value),
            "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        HPX_TEST_MSG((!satisfy_traversal_concept<iterator,
                         hpx::random_access_traversal_tag>::value),
            "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        HPX_TEST_MSG((!satisfy_traversal_concept<iterator,
                         hpx::random_access_traversal_tag>::value),
            "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        HPX_TEST_MSG((!satisfy_traversal_concept<iterator,
                         hpx::random_access_traversal_tag>::value),
            "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        HPX_TEST_MSG((satisfy_traversal_concept<iterator,
                         hpx::random_access_traversal_tag>::value),
            "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        HPX_TEST_MSG((!satisfy_traversal_concept<iterator,
                         hpx::random_access_traversal_tag>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        HPX_TEST_MSG((satisfy_traversal_concept<iterator,
                         hpx::random_access_traversal_tag>::value),
            "random access traversal input iterator");
    }
}

void is_iterator_test()
{
    using hpx::traits::is_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        HPX_TEST_MSG((is_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        HPX_TEST_MSG((is_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        HPX_TEST_MSG((is_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        HPX_TEST_MSG((is_iterator<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        HPX_TEST_MSG((is_iterator<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        HPX_TEST_MSG((is_iterator<iterator>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        HPX_TEST_MSG((is_iterator<iterator>::value),
            "random access traversal input iterator");
    }
}

void is_output_iterator_test()
{
    using hpx::traits::is_output_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        HPX_TEST_MSG((is_output_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        HPX_TEST_MSG((!is_output_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        HPX_TEST_MSG((is_output_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        HPX_TEST_MSG(
            (is_output_iterator<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        HPX_TEST_MSG(
            (is_output_iterator<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        HPX_TEST_MSG((is_output_iterator<iterator>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        HPX_TEST_MSG((is_output_iterator<iterator>::value),
            "random access traversal input iterator");
    }
}

void is_input_iterator_test()
{
    using hpx::traits::is_input_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        HPX_TEST_MSG((!is_input_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        HPX_TEST_MSG((is_input_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        HPX_TEST_MSG((is_input_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        HPX_TEST_MSG(
            (is_input_iterator<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        HPX_TEST_MSG(
            (is_input_iterator<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        HPX_TEST_MSG((is_input_iterator<iterator>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        HPX_TEST_MSG((is_input_iterator<iterator>::value),
            "random access traversal input iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::random_access_iterator_tag>;

        HPX_TEST_MSG((is_input_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::bidirectional_iterator_tag>;

        HPX_TEST_MSG((is_input_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::forward_iterator_tag>;

        HPX_TEST_MSG((is_input_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::input_iterator_tag>;

        HPX_TEST_MSG((is_input_iterator<iterator>::value), "hpx test iterator");
    }
}

void is_forward_iterator_test()
{
    using hpx::traits::is_forward_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        HPX_TEST_MSG(
            (!is_forward_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        HPX_TEST_MSG((!is_forward_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        HPX_TEST_MSG(
            (is_forward_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        HPX_TEST_MSG(
            (is_forward_iterator<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        HPX_TEST_MSG(
            (is_forward_iterator<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        HPX_TEST_MSG((is_forward_iterator<iterator>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        HPX_TEST_MSG((is_forward_iterator<iterator>::value),
            "random access traversal input iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::random_access_iterator_tag>;

        HPX_TEST_MSG(
            (is_forward_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::bidirectional_iterator_tag>;

        HPX_TEST_MSG(
            (is_forward_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::forward_iterator_tag>;

        HPX_TEST_MSG(
            (is_forward_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::input_iterator_tag>;

        HPX_TEST_MSG(
            (!is_forward_iterator<iterator>::value), "hpx test iterator");
    }
}

void is_bidirectional_iterator_test()
{
    using hpx::traits::is_bidirectional_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        HPX_TEST_MSG(
            (!is_bidirectional_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        HPX_TEST_MSG(
            (!is_bidirectional_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        HPX_TEST_MSG(
            (!is_bidirectional_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        HPX_TEST_MSG((is_bidirectional_iterator<iterator>::value),
            "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        HPX_TEST_MSG((is_bidirectional_iterator<iterator>::value),
            "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        HPX_TEST_MSG((is_bidirectional_iterator<iterator>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        HPX_TEST_MSG((is_bidirectional_iterator<iterator>::value),
            "random access traversal input iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::random_access_iterator_tag>;

        HPX_TEST_MSG(
            (is_bidirectional_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::bidirectional_iterator_tag>;

        HPX_TEST_MSG(
            (is_bidirectional_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::forward_iterator_tag>;

        HPX_TEST_MSG(
            (!is_bidirectional_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::input_iterator_tag>;

        HPX_TEST_MSG(
            (!is_bidirectional_iterator<iterator>::value), "hpx test iterator");
    }
}

void is_random_access_iterator_test()
{
    using hpx::traits::is_random_access_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        HPX_TEST_MSG(
            (!is_random_access_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        HPX_TEST_MSG(
            (!is_random_access_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        HPX_TEST_MSG(
            (!is_random_access_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        HPX_TEST_MSG((!is_random_access_iterator<iterator>::value),
            "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        HPX_TEST_MSG((is_random_access_iterator<iterator>::value),
            "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        HPX_TEST_MSG((!is_random_access_iterator<iterator>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        HPX_TEST_MSG((is_random_access_iterator<iterator>::value),
            "random access traversal input iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::random_access_iterator_tag>;

        HPX_TEST_MSG(
            (is_random_access_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::bidirectional_iterator_tag>;

        HPX_TEST_MSG(
            (!is_random_access_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::forward_iterator_tag>;

        HPX_TEST_MSG(
            (!is_random_access_iterator<iterator>::value), "hpx test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator =
            test::test_iterator<base_iterator, std::input_iterator_tag>;

        HPX_TEST_MSG(
            (!is_random_access_iterator<iterator>::value), "hpx test iterator");
    }
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        addition_result_test();
        dereference_result_test();
        equality_result_test();
        inequality_result_test();
        inplace_addition_result_test();
        inplace_subtraction_result_test();
        predecrement_result_test();
        preincrement_result_test();
        postdecrement_result_test();
        postincrement_result_test();
        subscript_result_test();
        subtraction_result_test();
        bidirectional_concept_test();
        random_access_concept_test();
        satisfy_traversal_concept_forward_test();
        satisfy_traversal_concept_bidirectional_test();
        satisfy_traversal_concept_random_access_test();
        is_iterator_test();
        is_forward_iterator_test();
        is_bidirectional_iterator_test();
        is_random_access_iterator_test();
    }

    return hpx::util::report_errors();
}
