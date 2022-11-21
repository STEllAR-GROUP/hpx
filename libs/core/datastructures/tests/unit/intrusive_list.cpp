//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/datastructures/detail/intrusive_list.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>

struct entry
{
    int value;

    entry* next = nullptr;
    entry* prev = nullptr;
};

void test_default_constructed()
{
    hpx::detail::intrusive_list<entry> list;

    HPX_TEST_EQ(list.size(), std::size_t(0));
    HPX_TEST(list.empty());

    HPX_TEST(list.front() == nullptr);
    HPX_TEST(list.back() == nullptr);
}

void test_default_constructed_splice()
{
    hpx::detail::intrusive_list<entry> list1;
    hpx::detail::intrusive_list<entry> list2;

    list1.splice(list2);

    HPX_TEST_EQ(list1.size(), std::size_t(0));
    HPX_TEST(list1.empty());

    HPX_TEST(list1.front() == nullptr);
    HPX_TEST(list1.back() == nullptr);
}

void test_one_element()
{
    hpx::detail::intrusive_list<entry> list;
    entry e1{1};

    list.push_back(e1);

    HPX_TEST_EQ(e1.value, 1);
    HPX_TEST(e1.prev == nullptr);
    HPX_TEST(e1.next == nullptr);

    HPX_TEST_EQ(list.size(), std::size_t(1));
    HPX_TEST(!list.empty());

    HPX_TEST(list.front() == &e1);
    HPX_TEST(list.back() == &e1);

    list.pop_front();

    HPX_TEST_EQ(list.size(), std::size_t(0));
    HPX_TEST(list.empty());

    HPX_TEST(list.front() == nullptr);
    HPX_TEST(list.back() == nullptr);
}

void test_one_element_erase()
{
    hpx::detail::intrusive_list<entry> list;
    entry e1{1};

    list.push_back(e1);

    HPX_TEST_EQ(e1.value, 1);
    HPX_TEST(e1.prev == nullptr);
    HPX_TEST(e1.next == nullptr);

    HPX_TEST_EQ(list.size(), std::size_t(1));
    HPX_TEST(!list.empty());

    HPX_TEST(list.front() == &e1);
    HPX_TEST(list.back() == &e1);

    list.erase(&e1);

    HPX_TEST_EQ(list.size(), std::size_t(0));
    HPX_TEST(list.empty());

    HPX_TEST(list.front() == nullptr);
    HPX_TEST(list.back() == nullptr);
}

void test_two_elements()
{
    hpx::detail::intrusive_list<entry> list;
    entry e1{1};
    entry e2{2};

    // add two entries
    list.push_back(e1);
    list.push_back(e2);

    HPX_TEST_EQ(e1.value, 1);
    HPX_TEST(e1.prev == nullptr);
    HPX_TEST(e1.next == &e2);

    HPX_TEST_EQ(e2.value, 2);
    HPX_TEST(e2.prev == &e1);
    HPX_TEST(e2.next == nullptr);

    HPX_TEST_EQ(list.size(), std::size_t(2));
    HPX_TEST(!list.empty());

    HPX_TEST(list.front() == &e1);
    HPX_TEST(list.back() == &e2);

    // remove e1
    list.pop_front();

    HPX_TEST_EQ(list.size(), std::size_t(1));
    HPX_TEST(!list.empty());

    HPX_TEST(list.front() == &e2);
    HPX_TEST(list.back() == &e2);

    HPX_TEST_EQ(e2.value, 2);
    HPX_TEST(e2.prev == nullptr);
    HPX_TEST(e2.next == nullptr);

    // remove e2
    list.pop_front();

    HPX_TEST_EQ(list.size(), std::size_t(0));
    HPX_TEST(list.empty());

    HPX_TEST(list.front() == nullptr);
    HPX_TEST(list.back() == nullptr);
}

void test_two_elements_erase()
{
    hpx::detail::intrusive_list<entry> list;
    entry e1{1};
    entry e2{2};

    // add two entries
    list.push_back(e1);
    list.push_back(e2);

    HPX_TEST_EQ(e1.value, 1);
    HPX_TEST(e1.prev == nullptr);
    HPX_TEST(e1.next == &e2);

    HPX_TEST_EQ(e2.value, 2);
    HPX_TEST(e2.prev == &e1);
    HPX_TEST(e2.next == nullptr);

    HPX_TEST_EQ(list.size(), std::size_t(2));
    HPX_TEST(!list.empty());

    HPX_TEST(list.front() == &e1);
    HPX_TEST(list.back() == &e2);

    // remove e1
    list.erase(&e1);

    HPX_TEST_EQ(list.size(), std::size_t(1));
    HPX_TEST(!list.empty());

    HPX_TEST(list.front() == &e2);
    HPX_TEST(list.back() == &e2);

    HPX_TEST_EQ(e2.value, 2);
    HPX_TEST(e2.prev == nullptr);
    HPX_TEST(e2.next == nullptr);

    // remove e2
    list.erase(&e2);

    HPX_TEST_EQ(list.size(), std::size_t(0));
    HPX_TEST(list.empty());

    HPX_TEST(list.front() == nullptr);
    HPX_TEST(list.back() == nullptr);
}

void test_two_elements_erase_reverse()
{
    hpx::detail::intrusive_list<entry> list;
    entry e1{1};
    entry e2{2};

    // add two entries
    list.push_back(e1);
    list.push_back(e2);

    HPX_TEST_EQ(e1.value, 1);
    HPX_TEST(e1.prev == nullptr);
    HPX_TEST(e1.next == &e2);

    HPX_TEST_EQ(e2.value, 2);
    HPX_TEST(e2.prev == &e1);
    HPX_TEST(e2.next == nullptr);

    HPX_TEST_EQ(list.size(), std::size_t(2));
    HPX_TEST(!list.empty());

    HPX_TEST(list.front() == &e1);
    HPX_TEST(list.back() == &e2);

    // remove e2
    list.erase(&e2);

    HPX_TEST_EQ(list.size(), std::size_t(1));
    HPX_TEST(!list.empty());

    HPX_TEST(list.front() == &e1);
    HPX_TEST(list.back() == &e1);

    HPX_TEST_EQ(e1.value, 1);
    HPX_TEST(e1.prev == nullptr);
    HPX_TEST(e1.next == nullptr);

    // remove e1
    list.erase(&e1);

    HPX_TEST_EQ(list.size(), std::size_t(0));
    HPX_TEST(list.empty());

    HPX_TEST(list.front() == nullptr);
    HPX_TEST(list.back() == nullptr);
}

void test_three_elements_iterate()
{
    hpx::detail::intrusive_list<entry> list;
    entry e1{1};
    entry e2{2};
    entry e3{3};

    // add three entries
    list.push_back(e1);
    list.push_back(e2);
    list.push_back(e3);

    HPX_TEST_EQ(e1.value, 1);
    HPX_TEST(e1.prev == nullptr);
    HPX_TEST(e1.next == &e2);

    HPX_TEST_EQ(e2.value, 2);
    HPX_TEST(e2.prev == &e1);
    HPX_TEST(e2.next == &e3);

    HPX_TEST_EQ(e3.value, 3);
    HPX_TEST(e3.prev == &e2);
    HPX_TEST(e3.next == nullptr);

    HPX_TEST_EQ(list.size(), std::size_t(3));
    HPX_TEST(!list.empty());

    HPX_TEST(list.front() == &e1);
    HPX_TEST(list.back() == &e3);

    // iterate
    int i = 1;
    for (entry* qe = list.front(); qe != nullptr; qe = qe->next, ++i)
    {
        HPX_TEST_EQ(qe->value, i);
    }
    HPX_TEST_EQ(i, 4);

    i = 3;
    for (entry* qe = list.back(); qe != nullptr; qe = qe->prev, --i)
    {
        HPX_TEST_EQ(qe->value, i);
    }
    HPX_TEST_EQ(i, 0);

    // remove e2
    list.erase(&e2);

    HPX_TEST_EQ(list.size(), std::size_t(2));
    HPX_TEST(!list.empty());

    HPX_TEST(list.front() == &e1);
    HPX_TEST(list.back() == &e3);
}

void test_three_elements_swap()
{
    hpx::detail::intrusive_list<entry> list1;
    entry e1{1};
    entry e2{2};
    entry e3{3};

    // add entries
    list1.push_back(e1);
    list1.push_back(e2);
    list1.push_back(e3);

    hpx::detail::intrusive_list<entry> list2;
    entry e4{4};
    entry e5{5};

    list2.push_back(e4);
    list2.push_back(e5);

    list1.swap(list2);

    HPX_TEST_EQ(list1.size(), std::size_t(2));
    HPX_TEST_EQ(list2.size(), std::size_t(3));

    // iterate
    int i = 1;
    for (entry* qe = list2.front(); qe != nullptr; qe = qe->next, ++i)
    {
        HPX_TEST_EQ(qe->value, i);
    }
    HPX_TEST_EQ(i, 4);

    for (entry* qe = list1.front(); qe != nullptr; qe = qe->next, ++i)
    {
        HPX_TEST_EQ(qe->value, i);
    }
    HPX_TEST_EQ(i, 6);
}

void test_three_elements_splice()
{
    hpx::detail::intrusive_list<entry> list1;
    entry e1{1};
    entry e2{2};
    entry e3{3};

    // add entries
    list1.push_back(e1);
    list1.push_back(e2);
    list1.push_back(e3);

    hpx::detail::intrusive_list<entry> list2;
    entry e4{4};
    entry e5{5};

    list2.push_back(e4);
    list2.push_back(e5);

    list1.splice(list2);

    HPX_TEST_EQ(list1.size(), std::size_t(5));
    HPX_TEST_EQ(list2.size(), std::size_t(0));

    // iterate
    int i = 1;
    for (entry* qe = list1.front(); qe != nullptr; qe = qe->next, ++i)
    {
        HPX_TEST_EQ(qe->value, i);
    }
    HPX_TEST_EQ(i, 6);

    i = 5;
    for (entry* qe = list1.back(); qe != nullptr; qe = qe->prev, --i)
    {
        HPX_TEST_EQ(qe->value, i);
    }
    HPX_TEST_EQ(i, 0);
}

void test_three_elements_empty_splice()
{
    hpx::detail::intrusive_list<entry> list1;
    entry e1{1};
    entry e2{2};
    entry e3{3};

    // add entries
    list1.push_back(e1);
    list1.push_back(e2);
    list1.push_back(e3);

    hpx::detail::intrusive_list<entry> list2;

    list1.splice(list2);

    HPX_TEST_EQ(list1.size(), std::size_t(3));
    HPX_TEST(!list1.empty());

    HPX_TEST(list1.front() == &e1);
    HPX_TEST(list1.back() == &e3);

    HPX_TEST_EQ(list2.size(), std::size_t(0));
    HPX_TEST(list2.empty());

    HPX_TEST(list2.front() == nullptr);
    HPX_TEST(list2.back() == nullptr);
}

void test_three_elements_splice_empty()
{
    hpx::detail::intrusive_list<entry> list1;
    entry e1{1};
    entry e2{2};
    entry e3{3};

    // add entries
    list1.push_back(e1);
    list1.push_back(e2);
    list1.push_back(e3);

    hpx::detail::intrusive_list<entry> list2;

    list2.splice(list1);

    HPX_TEST_EQ(list1.size(), std::size_t(0));
    HPX_TEST(list1.empty());

    HPX_TEST(list1.front() == nullptr);
    HPX_TEST(list1.back() == nullptr);

    HPX_TEST_EQ(list2.size(), std::size_t(3));
    HPX_TEST(!list2.empty());

    HPX_TEST(list2.front() == &e1);
    HPX_TEST(list2.back() == &e3);
}

int main()
{
    test_default_constructed();
    test_default_constructed_splice();
    test_one_element();
    test_one_element_erase();
    test_two_elements();
    test_two_elements_erase();
    test_two_elements_erase_reverse();
    test_three_elements_iterate();
    test_three_elements_swap();
    test_three_elements_splice();
    test_three_elements_empty_splice();
    test_three_elements_splice_empty();

    return hpx::util::report_errors();
}
