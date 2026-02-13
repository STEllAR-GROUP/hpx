//  Copyright (c) 2019 Austin McCartney
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/iterator_support.hpp>
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
        using base_type = hpx::util::iterator_adaptor<
            test_iterator<BaseIterator, IteratorTag>, BaseIterator, void,
            IteratorTag>;

    public:
        test_iterator() = default;

        explicit constexpr test_iterator(BaseIterator base)
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
    using pointer = int const*;
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

    bool operator==(bidirectional_traversal_iterator const& that) const
    {
        return this->state == that.state;
    }

    bool operator!=(bidirectional_traversal_iterator const& that) const
    {
        return this->state != that.state;
    }

    bool operator<(bidirectional_traversal_iterator const& that) const
    {
        return this->state < that.state;
    }

    bool operator<=(bidirectional_traversal_iterator const& that) const
    {
        return this->state <= that.state;
    }

    bool operator>(bidirectional_traversal_iterator const& that) const
    {
        return this->state > that.state;
    }

    bool operator>=(bidirectional_traversal_iterator const& that) const
    {
        return this->state >= that.state;
    }
};

struct random_access_traversal_iterator
{
    using difference_type = int;
    using value_type = int;
    using iterator_category = std::input_iterator_tag;
    using pointer = int const*;
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
        random_access_traversal_iterator const& that) const
    {
        return this->state - that.state;
    }

    bool operator==(random_access_traversal_iterator const& that) const
    {
        return this->state == that.state;
    }

    bool operator!=(random_access_traversal_iterator const& that) const
    {
        return this->state != that.state;
    }

    bool operator<(random_access_traversal_iterator const& that) const
    {
        return this->state < that.state;
    }

    bool operator<=(random_access_traversal_iterator const& that) const
    {
        return this->state <= that.state;
    }

    bool operator>(random_access_traversal_iterator const& that) const
    {
        return this->state > that.state;
    }

    bool operator>=(random_access_traversal_iterator const& that) const
    {
        return this->state >= that.state;
    }
};

template <typename T, typename U>
concept addition_result =
    requires { typename hpx::traits::detail::addition_result_t<T, U>; };

template <typename T>
concept dereference_result =
    requires { typename hpx::traits::detail::dereference_result_t<T>; };

template <typename T, typename U>
concept inplace_addition_result =
    requires { typename hpx::traits::detail::inplace_addition_result_t<T, U>; };

template <typename T, typename U>
concept subtraction_result =
    requires { typename hpx::traits::detail::subtraction_result_t<T, U>; };

template <typename T, typename U>
concept inplace_subtraction_result = requires {
    typename hpx::traits::detail::inplace_subtraction_result_t<T, U>;
};

template <typename T>
concept predecrement_result =
    requires { typename hpx::traits::detail::predecrement_result_t<T>; };

template <typename T>
concept preincrement_result =
    requires { typename hpx::traits::detail::preincrement_result_t<T>; };

template <typename T>
concept postdecrement_result =
    requires { typename hpx::traits::detail::postdecrement_result_t<T>; };

template <typename T>
concept postincrement_result =
    requires { typename hpx::traits::detail::postincrement_result_t<T>; };

template <typename T, typename U>
concept subscript_result =
    requires { typename hpx::traits::detail::subscript_result_t<T, U>; };

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
        B operator+(A const&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG((std::is_same_v<B, addition_result_t<C, A>>), "deduced type");

    HPX_TEST_MSG((!addition_result<B, A>), "invalid operation");
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

    HPX_TEST_MSG((!dereference_result<A>), "invalid operation");
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
        B operator==(A const&) const
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
        B operator!=(A const&) const
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
        B operator+=(A const&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG(
        (std::is_same_v<B, inplace_addition_result_t<C, A>>), "deduced type");

    HPX_TEST_MSG((!inplace_addition_result<B, A>), "invalid operation");
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
        B operator-=(A const&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG((std::is_same_v<B, inplace_subtraction_result_t<C, A>>),
        "deduced type");

    HPX_TEST_MSG((!inplace_subtraction_result<B, A>), "invalid operation");
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

    HPX_TEST_MSG((!predecrement_result<A>), "invalid operation");
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

    HPX_TEST_MSG((!preincrement_result<A>), "invalid operation");
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

    HPX_TEST_MSG((!postdecrement_result<A>), "invalid operation");
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

    HPX_TEST_MSG((!postincrement_result<A>), "invalid operation");
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
        B operator[](A const&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG((std::is_same_v<B, subscript_result_t<C, A>>), "deduced type");

    HPX_TEST_MSG((!subscript_result<B, A>), "invalid operation");
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
        B operator-(A const&) const
        {
            return B{};
        }
    };

    HPX_TEST_MSG(
        (std::is_same_v<B, subtraction_result_t<C, A>>), "deduced type");

    HPX_TEST_MSG((!subtraction_result<B, A>), "invalid operation");
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
    }

    return hpx::util::report_errors();
}
