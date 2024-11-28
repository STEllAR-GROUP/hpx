//  Copyright (c) 2021-2024 Hartmut Kaiser
//  Copyright (c) 2024 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright Ion Gaztanaga 2004-2014. Distributed under the Boost
// Copyright (C) 2013 Cromwell D. Enage
//
// See http://www.boost.org/libs/container for documentation.

#include <hpx/config.hpp>

#if !defined(HPX_HAVE_HIP)
#include <hpx/datastructures/detail/small_vector.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <deque>
#include <forward_list>
#include <iostream>
#include <list>
#include <utility>
#include <vector>

namespace test {

    class movable_and_copyable_int
    {
    public:
        static inline unsigned int count = 0;

        movable_and_copyable_int() noexcept
          : int_(0)
        {
            ++count;
        }

        explicit movable_and_copyable_int(int a) noexcept
          : int_(a)
        {
            // Disallow INT_MIN
            HPX_TEST(this->int_ != INT_MIN);
            ++count;
        }

        movable_and_copyable_int(movable_and_copyable_int const& mmi) noexcept
          : int_(mmi.int_)
        {
            ++count;
        }

        movable_and_copyable_int(movable_and_copyable_int&& mmi) noexcept
          : int_(mmi.int_)
        {
            mmi.int_ = 0;
            ++count;
        }

        ~movable_and_copyable_int()
        {
            HPX_TEST(this->int_ != INT_MIN);
            this->int_ = INT_MIN;
            --count;
        }

        movable_and_copyable_int& operator=(
            movable_and_copyable_int const& mi) noexcept
        {
            this->int_ = mi.int_;
            return *this;
        }

        movable_and_copyable_int& operator=(
            movable_and_copyable_int&& mmi) noexcept
        {
            this->int_ = mmi.int_;
            mmi.int_ = 0;
            return *this;
        }

        movable_and_copyable_int& operator=(int i) noexcept
        {
            this->int_ = i;
            HPX_TEST(this->int_ != INT_MIN);
            return *this;
        }

        friend bool operator==(const movable_and_copyable_int& l,
            const movable_and_copyable_int& r) noexcept
        {
            return l.int_ == r.int_;
        }

        friend bool operator!=(const movable_and_copyable_int& l,
            const movable_and_copyable_int& r) noexcept
        {
            return l.int_ != r.int_;
        }

        friend bool operator<(const movable_and_copyable_int& l,
            const movable_and_copyable_int& r) noexcept
        {
            return l.int_ < r.int_;
        }

        friend bool operator<=(const movable_and_copyable_int& l,
            const movable_and_copyable_int& r) noexcept
        {
            return l.int_ <= r.int_;
        }

        friend bool operator>=(const movable_and_copyable_int& l,
            const movable_and_copyable_int& r) noexcept
        {
            return l.int_ >= r.int_;
        }

        friend bool operator>(const movable_and_copyable_int& l,
            const movable_and_copyable_int& r) noexcept
        {
            return l.int_ > r.int_;
        }

        int get_int() const noexcept
        {
            return int_;
        }

        friend bool operator==(
            const movable_and_copyable_int& l, int r) noexcept
        {
            return l.get_int() == r;
        }

        friend bool operator==(
            int l, const movable_and_copyable_int& r) noexcept
        {
            return l == r.get_int();
        }

    private:
        int int_;
    };
}    // namespace test

namespace hpx {

    // Explicit instantiation to detect compilation errors

    // inplace_vector is an alias of small_vector so we have to write it out
    // explicitly:
    /*
    template <typename T, std::size_t MinInlineCapacity>
    using inplace_vector = small_vector<
        T,
        MinInlineCapacity,
        std::monostate, // Ignore the allocator
        true>; // emulate inplace_vector
     */

    // inplace_vector<int, 0>
    template class hpx::detail::small_vector<int, 0, std::monostate, true>;

    // inplace_vector<int, 1>
    template class hpx::detail::small_vector<int, 1, std::monostate, true>;

    // inplace_vector<int, 2>
    template class hpx::detail::small_vector<int, 2, std::monostate, true>;

    // inplace_vector<int, 10>
    template class hpx::detail::small_vector<int, 10, std::monostate, true>;

    // inplace_vector<int, 100>
    template class hpx::detail::small_vector<int, 100, std::monostate, true>;

    // inplace_vector<test::movable_and_copyable_int, 0>
    template class hpx::detail::small_vector<test::movable_and_copyable_int, 0,
        std::monostate, true>;

    // inplace_vector<test::movable_and_copyable_int, 10>
    template class hpx::detail::small_vector<test::movable_and_copyable_int, 10,
        std::monostate, true>;
}    // namespace hpx

namespace test {

    void inplace_vector_test()
    {
        // basic test with fewer elements than static size
        {
            using sm5_t = hpx::detail::small_vector<int, 5>;
            static_assert(
                sm5_t::static_capacity == 5, "sm5_t::static_capacity == 5");

            sm5_t sm5;
            sm5.push_back(1);
            HPX_TEST_EQ(sm5[0], 1);

            sm5_t const sm5_copy(sm5);
            HPX_TEST(sm5 == sm5_copy);
        }
        {
            using sm7_t = hpx::detail::inplace_vector<int, 7>;
            static_assert(
                sm7_t::static_capacity == 7, "sm7_t::static_capacity == 7");

            sm7_t sm7;
            sm7.push_back(1);
            HPX_TEST_EQ(sm7[0], 1);

            sm7_t const sm7_copy(sm7);
            HPX_TEST(sm7 == sm7_copy);
        }
        {
            using sm5_t = hpx::detail::inplace_vector<int, 5>;
            sm5_t sm5;
            sm5.push_back(1);
            HPX_TEST_EQ(sm5[0], 1);

            sm5_t sm5_copy(sm5);
            HPX_TEST(sm5 == sm5_copy);

            sm5.push_back(2);
            HPX_TEST_EQ(sm5[1], 2);
            HPX_TEST_EQ(sm5.size(), static_cast<std::size_t>(2));

            sm5_copy = sm5;
            HPX_TEST(sm5 == sm5_copy);

            sm5[0] = 3;
            HPX_TEST_EQ(sm5[0], 3);
            HPX_TEST_EQ(sm5_copy[0], 1);

            sm5_copy = sm5;
            sm5_t sm5_move(std::move(sm5));
            sm5 = sm5_t();
            HPX_TEST(sm5_move == sm5_copy);

            sm5 = sm5_copy;
            sm5_move = std::move(sm5);
            sm5 = sm5_t();
            HPX_TEST(sm5_move == sm5_copy);
        }

        // basic test with more elements than static size
        {
            using sm2_t = hpx::detail::inplace_vector<int, 2>;
            sm2_t sm2;
            sm2.push_back(1);
            HPX_TEST_EQ(sm2[0], 1);

            sm2_t sm2_copy(sm2);
            HPX_TEST(sm2 == sm2_copy);

            sm2.push_back(2);
            sm2.push_back(3);
            HPX_TEST_EQ(sm2[1], 2);
            HPX_TEST_EQ(sm2[2], 3);
            HPX_TEST_EQ(sm2.size(), static_cast<std::size_t>(3));

            sm2_copy = sm2;
            HPX_TEST(sm2 == sm2_copy);

            sm2[2] = 4;
            HPX_TEST_EQ(sm2[2], 4);
            HPX_TEST_EQ(sm2_copy[2], 3);

            sm2_copy = sm2;
            sm2_t sm2_move(std::move(sm2));
            sm2 = sm2_t();
            HPX_TEST(sm2_move == sm2_copy);

            sm2 = sm2_copy;
            sm2_move = std::move(sm2);
            sm2 = sm2_t();
            HPX_TEST(sm2_move == sm2_copy);
        }
    }

    // inplace vector has internal storage so some special swap cases must be
    // tested
    void test_swap()
    {
        using vec = hpx::detail::inplace_vector<int, 10>;

        {
            // v has elements, w empty
            vec v;
            for (std::size_t i = 0, max = v.capacity() - 1; i != max; ++i)
            {
                v.push_back(static_cast<int>(i));
            }

            vec w;

            vec const v_copy(v);
            vec const w_copy(w);

            v.swap(w);
            HPX_TEST(v == w_copy);
            HPX_TEST(w == v_copy);
        }
        {
            // v & w have elements
            vec v;
            for (std::size_t i = 0, max = v.capacity() - 1; i != max; ++i)
            {
                v.push_back(static_cast<int>(i));
            }

            vec w;
            for (std::size_t i = 0, max = v.capacity() / 2; i != max; ++i)
            {
                w.push_back(static_cast<int>(i));
            }

            vec const v_copy(v);
            vec const w_copy(w);

            v.swap(w);
            HPX_TEST(v == w_copy);
            HPX_TEST(w == v_copy);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContA, typename ContB>
    void check_equal_containers(ContA const& cont_a, ContB const& cont_b)
    {
        HPX_TEST_EQ(cont_a.size(), cont_b.size());

        auto itcont_a(cont_a.begin());
        auto itcont_a_end(cont_a.end());
        std::size_t dist = std::distance(itcont_a, itcont_a_end);
        HPX_TEST_EQ(dist, cont_a.size());

        auto itcont_b(cont_b.begin());
        auto itcont_b_end(cont_b.end());
        std::size_t dist2 = std::distance(itcont_b, itcont_b_end);
        HPX_TEST_EQ(dist2, cont_b.size());

        for (std::size_t i = 0; itcont_a != itcont_a_end;
            ++itcont_a, ++itcont_b, ++i)
        {
            HPX_TEST_EQ(*itcont_a, *itcont_b);
        }
    }

    template <typename SeqContainer>
    void test_insert_range(std::deque<int>& std_deque,
        SeqContainer& seq_container, std::deque<int> const& input_deque,
        std::size_t index)
    {
        HPX_ASSERT(std::size(input_deque) + std::size(seq_container) <=
            seq_container.max_size());
        check_equal_containers(std_deque, seq_container);

        std_deque.insert(
            std_deque.begin() + index, input_deque.begin(), input_deque.end());
        seq_container.insert(seq_container.begin() + index, input_deque.begin(),
            input_deque.end());

        check_equal_containers(std_deque, seq_container);
    }

    template <typename SeqContainer>
    void test_range_insertion()
    {
        using value_type = typename SeqContainer::value_type;
        constexpr auto max_size = SeqContainer::static_capacity;
        static_assert(
            max_size > 100, "max_size must be greater than 100 for this test.");

        std::deque<int> input_deque;
        for (int element = -10; element < 10; ++element)
        {
            input_deque.push_back(element + 20);
        }

        for (std::size_t i = 0; i <= input_deque.size(); ++i)
        {
            std::deque<int> std_deque;
            SeqContainer seq_container;

            for (int element = -10; element < 10; ++element)
            {
                std_deque.push_back(element);
                seq_container.push_back(value_type(element));
            }

            test_insert_range(std_deque, seq_container, input_deque, i);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    class emplace_int
    {
    public:
        explicit emplace_int(
            int a = 0, int b = 0, int c = 0, int d = 0, int e = 0) noexcept
          : a_(a)
          , b_(b)
          , c_(c)
          , d_(d)
          , e_(e)
        {
        }

        emplace_int(emplace_int const& o) = delete;
        emplace_int(emplace_int&& o) noexcept
          : a_(o.a_)
          , b_(o.b_)
          , c_(o.c_)
          , d_(o.d_)
          , e_(o.e_)
        {
        }

        emplace_int& operator=(emplace_int const& o) = delete;
        emplace_int& operator=(emplace_int&& o) noexcept
        {
            a_ = o.a_;
            b_ = o.b_;
            c_ = o.c_;
            d_ = o.d_;
            e_ = o.e_;
            return *this;
        }

        friend bool operator==(emplace_int const& l, emplace_int const& r)
        {
            return l.a_ == r.a_ && l.b_ == r.b_ && l.c_ == r.c_ &&
                l.d_ == r.d_ && l.e_ == r.e_;
        }

        friend bool operator<(emplace_int const& l, emplace_int const& r)
        {
            return l.sum() < r.sum();
        }

        friend bool operator>(emplace_int const& l, emplace_int const& r)
        {
            return l.sum() > r.sum();
        }

        friend bool operator!=(emplace_int const& l, emplace_int const& r)
        {
            return !(l == r);
        }

        ~emplace_int()
        {
            a_ = b_ = c_ = d_ = e_ = 0;
        }

        int sum() const
        {
            return a_ + b_ + c_ + d_ + e_;
        }

        int a_, b_, c_, d_, e_;
        int padding[6] = {};
    };

    static emplace_int expected[10];

    template <typename Container>
    void test_expected_container(Container const& ec,
        emplace_int const* expected, unsigned int only_first_n,
        unsigned int cont_offset = 0)
    {
        HPX_TEST(cont_offset <= ec.size());
        HPX_TEST(only_first_n <= (ec.size() - cont_offset));

        using const_iterator = typename Container::const_iterator;

        const_iterator itb(ec.begin()), ite(ec.end());
        unsigned int cur = 0;
        while (cont_offset--)
        {
            ++itb;
        }

        for (; itb != ite && only_first_n--; ++itb, ++cur)
        {
            emplace_int const& cr = *itb;
            HPX_TEST(cr == expected[cur]);
        }
    }

    template <typename Container>
    void test_emplace_back()
    {
        {
            new (&expected[0]) emplace_int();
            new (&expected[1]) emplace_int(1);
            new (&expected[2]) emplace_int(1, 2);
            new (&expected[3]) emplace_int(1, 2, 3);
            new (&expected[4]) emplace_int(1, 2, 3, 4);
            new (&expected[5]) emplace_int(1, 2, 3, 4, 5);

            Container c;
            using reference = typename Container::reference;

            {
                reference r = c.emplace_back();
                HPX_TEST(&r == &c.back());
                test_expected_container(c, &expected[0], 1);
            }
            {
                reference r = c.emplace_back(1);
                HPX_TEST(&r == &c.back());
                test_expected_container(c, &expected[0], 2);
            }

            c.emplace_back(1, 2);
            test_expected_container(c, &expected[0], 3);

            c.emplace_back(1, 2, 3);
            test_expected_container(c, &expected[0], 4);

            c.emplace_back(1, 2, 3, 4);
            test_expected_container(c, &expected[0], 5);

            c.emplace_back(1, 2, 3, 4, 5);
            test_expected_container(c, &expected[0], 6);
        }
    }

    template <typename Container>
    void test_emplace_before()
    {
        {
            new (&expected[0]) emplace_int();
            new (&expected[1]) emplace_int(1);
            new (&expected[2]) emplace_int();

            Container c;
            c.emplace(c.cend(), 1);
            c.emplace(c.cbegin());
            test_expected_container(c, &expected[0], 2);

            c.emplace(c.cend());
            test_expected_container(c, &expected[0], 3);
        }
        {
            new (&expected[0]) emplace_int();
            new (&expected[1]) emplace_int(1);
            new (&expected[2]) emplace_int(1, 2);
            new (&expected[3]) emplace_int(1, 2, 3);
            new (&expected[4]) emplace_int(1, 2, 3, 4);
            new (&expected[5]) emplace_int(1, 2, 3, 4, 5);

            // emplace_front-like
            Container c;
            c.emplace(c.cbegin(), 1, 2, 3, 4, 5);
            c.emplace(c.cbegin(), 1, 2, 3, 4);
            c.emplace(c.cbegin(), 1, 2, 3);
            c.emplace(c.cbegin(), 1, 2);
            c.emplace(c.cbegin(), 1);
            c.emplace(c.cbegin());
            test_expected_container(c, &expected[0], 6);
            c.clear();

            // emplace_back-like
            auto i = c.emplace(c.cend());
            test_expected_container(c, &expected[0], 1);

            i = c.emplace(++i, 1);
            test_expected_container(c, &expected[0], 2);

            i = c.emplace(++i, 1, 2);
            test_expected_container(c, &expected[0], 3);

            i = c.emplace(++i, 1, 2, 3);
            test_expected_container(c, &expected[0], 4);

            i = c.emplace(++i, 1, 2, 3, 4);
            test_expected_container(c, &expected[0], 5);

            i = c.emplace(++i, 1, 2, 3, 4, 5);
            test_expected_container(c, &expected[0], 6);
            c.clear();

            // emplace in the middle
            c.emplace(c.cbegin());
            test_expected_container(c, &expected[0], 1);

            i = c.emplace(c.cend(), 1, 2, 3, 4, 5);
            test_expected_container(c, &expected[0], 1);

            test_expected_container(c, &expected[5], 1, 1);

            i = c.emplace(i, 1, 2, 3, 4);
            test_expected_container(c, &expected[0], 1);

            test_expected_container(c, &expected[4], 2, 1);

            i = c.emplace(i, 1, 2, 3);
            test_expected_container(c, &expected[0], 1);
            test_expected_container(c, &expected[3], 3, 1);

            i = c.emplace(i, 1, 2);
            test_expected_container(c, &expected[0], 1);
            test_expected_container(c, &expected[2], 4, 1);

            i = c.emplace(i, 1);
            test_expected_container(c, &expected[0], 6);
        }
    }

    template <typename VectorContainerType>
    void test_vector_methods_with_initializer_list_as_argument_for()
    {
        using allocator_type = typename VectorContainerType::allocator_type;

        {
            VectorContainerType const tested_vector = {1, 2, 3};
            std::vector<int> const expected_vector = {1, 2, 3};
            check_equal_containers(tested_vector, expected_vector);
        }
        {
            VectorContainerType const tested_vector(
                {1, 2, 3}, allocator_type());
            std::vector<int> const expected_vector = {1, 2, 3};
            check_equal_containers(tested_vector, expected_vector);
        }
        {
            VectorContainerType tested_vector = {1, 2, 3};
            tested_vector = {11, 12, 13};

            std::vector<int> const expected_vector = {11, 12, 13};
            check_equal_containers(tested_vector, expected_vector);
        }

        {
            VectorContainerType tested_vector = {1, 2, 3};
            tested_vector.assign({5, 6, 7});

            std::vector<int> const expected_vector = {5, 6, 7};
            check_equal_containers(tested_vector, expected_vector);
        }

        {
            VectorContainerType tested_vector = {1, 2, 3};
            tested_vector.insert(tested_vector.cend(), {5, 6, 7});

            std::vector<int> const expected_vector = {1, 2, 3, 5, 6, 7};
            check_equal_containers(tested_vector, expected_vector);
        }
    }

    void test_general()
    {
        using hpx::detail::inplace_vector;

        {
            inplace_vector<int, 5> const v1 = {1, 2, 3, 4, 5};
            inplace_vector<int, 7> const v2 = {1, 2, 3, 4, 5};

            HPX_TEST(v1 == v2);
        }
        {
            inplace_vector<int, 5> const v_init_list = {1, 2, 3, 4, 5};
            inplace_vector<int, 5> v_push_back;
            inplace_vector<int, 5> v_emplace_back;
            inplace_vector<int, 5> v_insert;

            for (int i = 0; i < 5; ++i)
            {
                v_push_back.push_back(i);
                v_emplace_back.emplace_back(i);
                v_insert.insert(v_insert.end(), i);
            }

            HPX_TEST(v_init_list == v_push_back);
            HPX_TEST(v_init_list == v_emplace_back);
            HPX_TEST(v_init_list == v_insert);
        }
    }

    void test_exceptions()
    {
        using hpx::detail::inplace_vector;
        {
            inplace_vector<int, 5> v = {1, 2, 3, 4, 5};
            try
            {
                v.push_back(42);
            }
            catch (std::bad_alloc&)
            {
            }
            catch (...)
            {
                HPX_ASSERT(false);
            }
        }
        {
            inplace_vector<int, 5> v = {1, 2, 3, 4, 5};
            try
            {
                v.emplace_back(42);
            }
            catch (std::bad_alloc&)
            {
            }
            catch (...)
            {
                HPX_ASSERT(false);
            }
        }
        {
            inplace_vector<int, 5> v = {1, 2, 3, 4, 5};
            try
            {
                v.insert(v.end(), 3);
            }
            catch (std::bad_alloc&)
            {
            }
            catch (...)
            {
                HPX_ASSERT(false);
            }
        }
         {
            inplace_vector<int, 5> v = {1, 2, 3, 4, 5};
            try
            {
                (void)v.at(6);
            }
            catch (std::out_of_range&)
            {
            }
            catch (...)
            {
                HPX_ASSERT(false);
            }
        }


    }
}    // namespace test

int main()
{
    test::inplace_vector_test();
    test::test_swap();
    test::test_exceptions();

    // Emplace testing
    test::test_emplace_before<
        hpx::detail::inplace_vector<test::emplace_int, 100>>();

    // Initializer lists testing
    test::test_vector_methods_with_initializer_list_as_argument_for<
        hpx::detail::inplace_vector<int, 100>>();

    return hpx::util::report_errors();
}

#else

int main()
{
    return 0;
}

#endif
