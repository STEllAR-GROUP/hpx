//  Copyright (C) 2019-2022 T. Zachary Laine
//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

#if defined(GNUC) && !defined(clang)
#include <bits/uses_allocator.h>
#endif

// clang-format off
#if (201703L < __cplusplus && defined(cpp_lib_ranges))
#define CPP20_CONCEPTS 1
#else
#define CPP20_CONCEPTS 0
#endif
#if (201703L <= __cplusplus && defined(__has_include) &&                       \
    __has_include(<stl2/ranges.hpp>))
#define CMCSTL2_CONCEPTS 1
#else
#define CMCSTL2_CONCEPTS 0
#endif
#define USE_CONCEPTS CPP20_CONCEPTS || CMCSTL2_CONCEPTS
// clang-format on

#if CPP20_CONCEPTS
#include <ranges>
#elif CMCSTL2_CONCEPTS
#include <stl2/algorithm.hpp>
#include <stl2/ranges.hpp>
#endif

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hpx::detail {

#if CMCSTL2_CONCEPTS && !CPP20_CONCEPTS
    using namespace std::experimental;
#endif

    // TODO: Remove once we have this in C++20 mode ubiquitously.
    template <typename T>
    using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

    struct sorted_unique_t
    {
        explicit sorted_unique_t() = default;
    };
    inline constexpr sorted_unique_t sorted_unique{};

    template <typename KeyRef, typename KeyIter>
    struct flat_set_iterator
    {
        static_assert(std::is_reference_v<KeyRef>);

        using iterator_category = std::random_access_iterator_tag;
        using value_type = remove_cvref_t<KeyRef>;
        using difference_type =
            typename std::iterator_traits<KeyIter>::difference_type;
        using reference = KeyRef;

        struct arrow_proxy
        {
            constexpr reference* operator->() noexcept
            {
                return &value_;
            }
            constexpr reference const* operator->() const noexcept
            {
                return &value_;
            }
            explicit arrow_proxy(reference value) noexcept
              : value_(HPX_MOVE(value))
            {
            }

        private:
            reference value_;
        };
        using pointer = arrow_proxy;

        flat_set_iterator() = default;
        explicit flat_set_iterator(KeyIter key_it)
          : key_it_(key_it)
        {
        }

        template <typename KeyRef2, typename KeyIter2,
            typename Enable =
                std::enable_if_t<std::is_convertible_v<KeyRef2, KeyRef> &&
                    std::is_convertible_v<KeyIter2, KeyIter>>>
        flat_set_iterator(flat_set_iterator<KeyRef2, KeyIter2> other)
          : key_it_(other.key_it_)
        {
        }

        constexpr decltype(auto) operator*() const noexcept
        {
            return ref();
        }
        constexpr pointer operator->() const noexcept
        {
            return arrow_proxy(ref());
        }

        constexpr auto& operator[](difference_type n) const noexcept
        {
            return *(key_it_ + n);
        }

        flat_set_iterator operator+(difference_type n) const noexcept
        {
            return flat_set_iterator(key_it_ + n);
        }
        flat_set_iterator operator-(difference_type n) const noexcept
        {
            return flat_set_iterator(key_it_ - n);
        }

        flat_set_iterator& operator++() noexcept
        {
            ++key_it_;
            return *this;
        }
        flat_set_iterator operator++(int) noexcept
        {
            flat_set_iterator tmp(*this);
            ++key_it_;
            return tmp;
        }

        flat_set_iterator& operator--() noexcept
        {
            --key_it_;
            return *this;
        }
        flat_set_iterator operator--(int) noexcept
        {
            flat_set_iterator tmp(*this);
            --key_it_;
            return tmp;
        }

        flat_set_iterator& operator+=(difference_type n) noexcept
        {
            key_it_ += n;
            return *this;
        }
        flat_set_iterator& operator-=(difference_type n) noexcept
        {
            key_it_ -= n;
            return *this;
        }

        KeyIter key_iter() const
        {
            return key_it_;
        }

        friend bool operator==(flat_set_iterator lhs, flat_set_iterator rhs)
        {
            return lhs.key_it_ == rhs.key_it_;
        }
        friend bool operator!=(flat_set_iterator lhs, flat_set_iterator rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator<(flat_set_iterator lhs, flat_set_iterator rhs)
        {
            return lhs.key_it_ < rhs.key_it_;
        }
        friend bool operator<=(flat_set_iterator lhs, flat_set_iterator rhs)
        {
            return lhs == rhs || lhs < rhs;
        }
        friend bool operator>(flat_set_iterator lhs, flat_set_iterator rhs)
        {
            return rhs < lhs;
        }
        friend bool operator>=(flat_set_iterator lhs, flat_set_iterator rhs)
        {
            return lhs == rhs || rhs < lhs;
        }

        friend difference_type operator-(
            flat_set_iterator lhs, flat_set_iterator rhs)
        {
            return lhs.key_it_ - rhs.key_it_;
        }

    private:
        template <typename KeyRef2, typename KeyIter2>
        friend struct flat_set_iterator;

        auto& ref() const
        {
            return *key_it_;
        }

        KeyIter key_it_;
    };

    template <typename Key, typename Compare = std::less<Key>,
        typename KeyContainer = std::vector<Key>>
    class flat_set
    {
        template <typename Alloc>
        using uses =
            std::enable_if_t<std::uses_allocator<KeyContainer, Alloc>::value>;

        template <typename Container, typename = void>
        struct has_begin_end : std::false_type
        {
        };
        template <typename Container>
        struct has_begin_end<Container,
            std::void_t<decltype(std::begin(std::declval<Container>())),
                decltype(std::end(std::declval<Container>()))>> : std::true_type
        {
        };
        template <typename Container>
        using container = std::enable_if_t<has_begin_end<Container>::value>;

    public:
        // types:
        using key_type = Key;
        using value_type = Key;
        using key_compare = Compare;
        using value_compare = Compare;
        using reference = value_type&;
        using const_reference = key_type const&;
        using size_type = std::size_t;
        using difference_type = ptrdiff_t;
        using iterator = flat_set_iterator<value_type&,
            typename KeyContainer::const_iterator>;    // see 21.2
        using const_iterator = flat_set_iterator<value_type const&,
            typename KeyContainer::const_iterator>;    // see 21.2
        using reverse_iterator = flat_set_iterator<value_type&,
            typename KeyContainer::const_reverse_iterator>;    // see 21.2
        using const_reverse_iterator = flat_set_iterator<value_type const&,
            typename KeyContainer::const_reverse_iterator>;    // see 21.2
        using key_container_type = KeyContainer;

        // ??, construct/copy/destroy
        flat_set()
          : flat_set(key_compare())
        {
        }
        explicit flat_set(key_container_type key_cont)
          : c{HPX_MOVE(key_cont)}
          , compare(key_compare())
        {
            mutable_iterator first(c.begin());
            mutable_iterator last(c.end());
#if USE_CONCEPTS
            std::ranges::sort(first, last, key_comp());
#else
            std::sort(first, last, key_comp());
#endif
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_set(key_container_type const& key_cont, Alloc const& a)
          : c{key_container_type(key_cont, a)}
          , compare()
        {
            mutable_iterator first(c.begin());
            mutable_iterator last(c.end());
#if USE_CONCEPTS
            std::ranges::sort(first, last, key_comp());
#else
            std::sort(first, last, key_comp());
#endif
        }
        template <typename Container, typename Enable = container<Container>>
        explicit flat_set(
            Container const& cont, key_compare const& comp = key_compare())
          : flat_set(std::begin(cont), std::end(cont), comp)
        {
        }
        template <typename Container, typename Alloc,
            typename Enable1 = container<Container>,
            typename Enable2 = uses<Alloc>>
        flat_set(Container const& cont, Alloc const& a)
          : flat_set(std::begin(cont), std::end(cont), a)
        {
        }
        flat_set(sorted_unique_t, key_container_type key_cont)
          : c{HPX_MOVE(key_cont)}
          , compare(key_compare())
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_set(
            sorted_unique_t, key_container_type const& key_cont, Alloc const& a)
          : c{key_container_type(key_cont, a)}
          , compare()
        {
        }
        template <typename Container, typename Enable = container<Container>>
        flat_set(sorted_unique_t s, Container const& cont,
            key_compare const& comp = key_compare())
          : flat_set(s, std::begin(cont), std::end(cont), comp)
        {
        }
        template <typename Container, typename Alloc,
            typename Enable1 = container<Container>,
            typename Enable2 = uses<Alloc>>
        flat_set(sorted_unique_t s, Container const& cont, Alloc const& a)
          : c{key_container_type(a)}
          , compare()
        {
            c.reserve(cont.size());
            insert(s, std::begin(cont), std::end(cont));
        }
        explicit flat_set(key_compare const& comp)
          : c()
          , compare(comp)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_set(key_compare const& comp, Alloc const& a)
          : c{key_container_type(a)}
          , compare(comp)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        explicit flat_set(Alloc const& a)
          : c{key_container_type(a)}
          , compare()
        {
        }
        template <typename InputIterator>
        flat_set(InputIterator first, InputIterator last,
            key_compare const& comp = key_compare())
          : c()
          , compare(comp)
        {
            insert(first, last);
        }
        template <typename InputIterator, typename Alloc,
            typename Enable = uses<Alloc>>
        flat_set(InputIterator first, InputIterator last,
            key_compare const& comp, Alloc const& a)
          : c{key_container_type(a)}
          , compare(comp)
        {
            insert(first, last);
        }
        template <typename InputIterator, typename Alloc,
            typename Enable = uses<Alloc>>
        flat_set(InputIterator first, InputIterator last, Alloc const& a)
          : flat_set(first, last, key_compare(), a)
        {
        }
        template <typename InputIterator>
        flat_set(sorted_unique_t s, InputIterator first, InputIterator last,
            key_compare const& comp = key_compare())
          : c()
          , compare(comp)
        {
            insert(s, first, last);
        }
        template <typename InputIterator, typename Alloc,
            typename Enable = uses<Alloc>>
        flat_set(sorted_unique_t s, InputIterator first, InputIterator last,
            key_compare const& comp, Alloc const& a)
          : c{key_container_type(a)}
          , compare(comp)
        {
            insert(s, first, last);
        }
        template <typename InputIterator, typename Alloc,
            typename Enable = uses<Alloc>>
        flat_set(sorted_unique_t s, InputIterator first, InputIterator last,
            Alloc const& a)
          : flat_set(s, first, last, key_compare(), a)
        {
        }
        flat_set(std::initializer_list<value_type>&& il,
            key_compare const& comp = key_compare())
          : flat_set(il, comp)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_set(std::initializer_list<value_type>&& il,
            key_compare const& comp, Alloc const& a)
          : flat_set(std::begin(il), std::end(il), comp, a)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_set(std::initializer_list<value_type>&& il, Alloc const& a)
          : flat_set(std::begin(il), std::end(il), key_compare(), a)
        {
        }
        flat_set(sorted_unique_t s, std::initializer_list<value_type>&& il,
            key_compare const& comp = key_compare())
          : flat_set(s, il, comp)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_set(sorted_unique_t s, std::initializer_list<value_type>&& il,
            key_compare const& comp, Alloc const& a)
          : flat_set(s, std::begin(il), std::end(il), comp, a)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_set(sorted_unique_t s, std::initializer_list<value_type>&& il,
            Alloc const& a)
          : flat_set(s, std::begin(il), std::end(il), key_compare(), a)
        {
        }
        flat_set& operator=(std::initializer_list<value_type> il)
        {
            flat_set tmp(il, compare);
            swap(tmp);
            return *this;
        }

        // iterators
        iterator begin() noexcept
        {
            return iterator(c.begin());
        }
        const_iterator begin() const noexcept
        {
            return const_iterator(c.begin());
        }
        iterator end() noexcept
        {
            return iterator(c.end());
        }
        const_iterator end() const noexcept
        {
            return const_iterator(c.end());
        }
        reverse_iterator rbegin() noexcept
        {
            return reverse_iterator(c.rbegin());
        }
        const_reverse_iterator rbegin() const noexcept
        {
            return const_reverse_iterator(c.rbegin());
        }
        reverse_iterator rend() noexcept
        {
            return reverse_iterator(c.rend());
        }
        const_reverse_iterator rend() const noexcept
        {
            return const_reverse_iterator(c.rend());
        }

        const_iterator cbegin() const noexcept
        {
            return begin();
        }
        const_iterator cend() const noexcept
        {
            return end();
        }
        const_reverse_iterator crbegin() const noexcept
        {
            return rbegin();
        }
        const_reverse_iterator crend() const noexcept
        {
            return rend();
        }

        // ??, capacity
        [[nodiscard]] bool empty() const noexcept
        {
            return c.empty();
        }
        size_type size() const noexcept
        {
            return c.size();
        }
        size_type max_size() const noexcept
        {
            return c.max_size();
        }

        // ??, modifiers
        template <typename... Args,
            typename Enable =
                std::enable_if_t<std::is_constructible_v<key_type, Args&&...>>>
        std::pair<iterator, bool> emplace(Args&&... args)
        {
            key_type k(HPX_FORWARD(Args, args)...);
            auto it = key_lower_bound(k);
            if (it == c.end() || compare(*it, k) || compare(k, *it))
            {
                it = c.insert(it, k);
                return std::pair<iterator, bool>(iterator(it), true);
            }
            return std::pair<iterator, bool>(iterator(it), false);
        }
        template <typename... Args,
            typename Enable =
                std::enable_if_t<std::is_constructible_v<key_type, Args&&...>>>
        iterator emplace_hint(const_iterator, Args&&... args)
        {
            return emplace(HPX_FORWARD(Args, args)...).first;
        }
        std::pair<iterator, bool> insert(value_type const& x)
        {
            return emplace(x);
        }
        std::pair<iterator, bool> insert(value_type&& x)
        {
            return emplace(HPX_MOVE(x));
        }
        iterator insert(const_iterator position, value_type const& x)
        {
            return emplace_hint(position, x);
        }
        iterator insert(const_iterator position, value_type&& x)
        {
            return emplace_hint(position, HPX_MOVE(x));
        }
        template <typename P,
            typename Enable =
                std::enable_if_t<std::is_constructible_v<key_type, P&&>>>
        std::pair<iterator, bool> insert(P&& x)
        {
            return emplace(HPX_FORWARD(P, x));
        }
        template <typename P,
            typename Enable =
                std::enable_if_t<std::is_constructible_v<key_type, P&&>>>
        iterator insert(const_iterator position, P&& x)
        {
            return emplace_hint(position, HPX_FORWARD(P, x));
        }
        template <typename InputIterator>
        void insert(InputIterator first, InputIterator last)
        {
            auto const prev_size = size();
            for (auto it = first; it != last; ++it)
            {
                c.push_back(*it);
            }

            mutable_iterator inserted_first(c.begin() + prev_size);
            mutable_iterator inserted_last(c.end());
#if USE_CONCEPTS
            std::ranges::sort(inserted_first, inserted_last, key_comp());
#else
            std::sort(inserted_first, inserted_last, key_comp());
#endif

            if (!prev_size)
                return;

            mutable_iterator mutable_first(c.begin());
#if USE_CONCEPTS
            std::ranges::inplace_merge(
                mutable_first, inserted_first, inserted_last, key_comp());
#else
            std::inplace_merge(
                mutable_first, inserted_first, inserted_last, key_comp());
#endif
        }
        template <typename InputIterator>
        void insert(sorted_unique_t, InputIterator first, InputIterator last)
        {
            auto const prev_size = size();
            for (auto it = first; it != last; ++it)
            {
                c.push_back(*it);
            }

            mutable_iterator inserted_first(c.begin() + prev_size);
            mutable_iterator inserted_last(c.end());

            if (!prev_size)
                return;

            mutable_iterator mutable_first(c.begin());
#if USE_CONCEPTS
            std::ranges::inplace_merge(
                mutable_first, inserted_first, inserted_last, key_comp());
#else
            std::inplace_merge(
                mutable_first, inserted_first, inserted_last, key_comp());
#endif
        }
        void insert(std::initializer_list<value_type> il)
        {
            insert(il.begin(), il.end());
        }
        void insert(sorted_unique_t s, std::initializer_list<value_type> il)
        {
            insert(s, il.begin(), il.end());
        }

        key_container_type extract() &&
        {
            scoped_clear _(this);
            return HPX_MOVE(c);
        }
        void replace(key_container_type&& key_cont)
        {
            scoped_clear _(this);
            c = HPX_MOVE(key_cont);
            _.release();
        }

        iterator erase(iterator position)
        {
            return iterator(c.erase(position.key_iter()));
        }
        iterator erase(const_iterator position)
        {
            return iterator(c.erase(position.key_iter()));
        }
        size_type erase(key_type const& x)
        {
            auto it = key_find(x);
            if (it == c.end())
                return static_cast<size_type>(0);
            c.erase(it);
            return static_cast<size_type>(1);
        }
        iterator erase(const_iterator first, const_iterator last)
        {
            return iterator(c.erase(first.key_iter(), last.key_iter()));
        }

        void swap(flat_set& fm) noexcept(
            std::is_nothrow_swappable_v<key_compare>)
        {
            using std::swap;
            swap(compare, fm.compare);
            swap(c, fm.c);
        }
        void clear() noexcept
        {
            c.clear();
        }

        // observers
        key_compare key_comp() const
        {
            return compare;
        }
        value_compare value_comp() const
        {
            return compare;
        }
        key_container_type const& keys() const noexcept
        {
            return c;
        }

        // map operations
        iterator find(key_type const& x)
        {
            auto it = key_find(x);
            return iterator(it);
        }
        const_iterator find(key_type const& x) const
        {
            auto it = key_find(x);
            return const_iterator(it);
        }
        template <typename K>
        iterator find(K const& x)
        {
            auto it = key_find(x);
            return iterator(it);
        }
        template <typename K>
        const_iterator find(K const& x) const
        {
            auto it = key_find(x);
            return iterator(it);
        }
        size_type count(key_type const& x) const
        {
            auto it = key_find(x);
            return static_cast<size_type>(it == c.end() ? 0 : 1);
        }
        template <typename K>
        size_type count(K const& x) const
        {
            auto it = key_find(x);
            return static_cast<size_type>(it == c.end() ? 0 : 1);
        }
        bool contains(key_type const& x) const
        {
            return count(x) == static_cast<size_type>(1);
        }
        template <typename K>
        bool contains(K const& x) const
        {
            return count(x) == static_cast<size_type>(1);
        }
        iterator lower_bound(key_type const& x)
        {
            auto it = key_lower_bound(x);
            return iterator(it);
        }
        const_iterator lower_bound(key_type const& x) const
        {
            auto it = key_lower_bound(x);
            return const_iterator(it);
        }
        template <typename K>
        iterator lower_bound(K const& x)
        {
            auto it = key_lower_bound(x);
            return iterator(it);
        }
        template <typename K>
        const_iterator lower_bound(K const& x) const
        {
            auto it = key_lower_bound(x);
            return const_iterator(it);
        }
        iterator upper_bound(key_type const& x)
        {
            auto it = key_upper_bound(x);
            return iterator(it);
        }
        const_iterator upper_bound(key_type const& x) const
        {
            auto it = key_upper_bound(x);
            return const_iterator(it);
        }
        template <typename K>
        iterator upper_bound(K const& x)
        {
            auto it = key_upper_bound(x);
            return iterator(it);
        }
        template <typename K>
        const_iterator upper_bound(K const& x) const
        {
            auto it = key_upper_bound(x);
            return const_iterator(it);
        }
        std::pair<iterator, iterator> equal_range(key_type const& k)
        {
            iterator const first = lower_bound(k);
            iterator const last = find_if(
                first, end(), [&](auto const& x) { return compare(k, x); });
            return std::pair<iterator, iterator>(first, last);
        }
        std::pair<const_iterator, const_iterator> equal_range(
            key_type const& k) const
        {
            const_iterator const first = lower_bound(k);
            const_iterator const last = find_if(
                first, end(), [&](auto const& x) { return compare(k, x); });
            return std::pair<const_iterator, const_iterator>(first, last);
        }
        template <typename K>
        std::pair<iterator, iterator> equal_range(K const& k)
        {
            iterator const first = lower_bound(k);
            iterator const last = find_if(
                first, end(), [&](auto const& x) { return compare(k, x); });
            return std::pair<iterator, iterator>(first, last);
        }
        template <typename K>
        std::pair<const_iterator, const_iterator> equal_range(K const& k) const
        {
            const_iterator const first = lower_bound(k);
            const_iterator const last = find_if(
                first, end(), [&](auto const& x) { return compare(k, x); });
            return std::pair<const_iterator, const_iterator>(first, last);
        }

        friend bool operator==(flat_set const& x, flat_set const& y)
        {
#if USE_CONCEPTS
            return std::ranges::equal(x, y);
#else
            return std::equal(x.begin(), x.end(), y.begin(), y.end());
#endif
        }
        friend bool operator!=(flat_set const& x, flat_set const& y)
        {
            return !(x == y);
        }
        friend bool operator<(flat_set const& x, flat_set const& y)
        {
#if USE_CONCEPTS
            return std::ranges::lexicographical_compare(
                x, y, [](auto lhs, auto rhs) { return lhs < rhs; });
#else
            return std::lexicographical_compare(
                x.begin(), x.end(), y.begin(), y.end());
#endif
        }
        friend bool operator>(flat_set const& x, flat_set const& y)
        {
            return y < x;
        }
        friend bool operator<=(flat_set const& x, flat_set const& y)
        {
            return !(y < x);
        }
        friend bool operator>=(flat_set const& x, flat_set const& y)
        {
            return !(x < y);
        }

        friend void swap(flat_set& x, flat_set& y) noexcept(noexcept(x.swap(y)))
        {
            return x.swap(y);
        }

    private:
        key_container_type c;    // exposition only
        key_compare compare;     // exposition only

        // exposition only
        struct scoped_clear
        {
            explicit scoped_clear(flat_set* fm)
              : fm_(fm)
            {
            }

            scoped_clear(scoped_clear const&) = delete;
            scoped_clear(scoped_clear&& rhs) noexcept
              : fm_(rhs.fm_)
            {
                rhs.fm_ = nullptr;
            }

            scoped_clear& operator=(scoped_clear const&) = delete;
            scoped_clear& operator=(scoped_clear&& rhs) noexcept
            {
                fm_ = rhs.fm_;
                rhs.fm_ = nullptr;
                return *this;
            }

            ~scoped_clear()
            {
                if (fm_)
                    fm_->clear();
            }
            void release()
            {
                fm_ = nullptr;
            }

        private:
            flat_set* fm_;
        };

        using key_iter_t = typename KeyContainer::iterator;
        using key_const_iter_t = typename KeyContainer::const_iterator;

        using mutable_iterator = flat_set_iterator<key_type&, key_iter_t>;

        template <typename K>
        key_iter_t key_lower_bound(K const& k)
        {
#if USE_CONCEPTS
            return std::ranges::lower_bound(c, k, compare);
#else
            return std::lower_bound(c.begin(), c.end(), k, compare);
#endif
        }
        template <typename K>
        key_const_iter_t key_lower_bound(K const& k) const
        {
#if USE_CONCEPTS
            return ranges::lower_bound(c, k, compare);
#else
            return std::lower_bound(c.begin(), c.end(), k, compare);
#endif
        }
        template <typename K>
        key_iter_t key_upper_bound(K const& k)
        {
#if USE_CONCEPTS
            return ranges::upper_bound(c, k, compare);
#else
            return std::upper_bound(c.begin(), c.end(), k, compare);
#endif
        }
        template <typename K>
        key_const_iter_t key_upper_bound(K const& k) const
        {
#if USE_CONCEPTS
            return ranges::upper_bound(c, k, compare);
#else
            return std::upper_bound(c.begin(), c.end(), k, compare);
#endif
        }
        template <typename K>
        key_iter_t key_find(K const& k)
        {
            auto it = key_lower_bound(k);
            if (it != c.end() && (compare(*it, k) || compare(k, *it)))
                it = c.end();
            return it;
        }
        template <typename K>
        key_const_iter_t key_find(K const& k) const
        {
            auto it = key_lower_bound(k);
            if (it != c.end() && (compare(*it, k) || compare(k, *it)))
                it = c.end();
            return it;
        }
    };
}    // namespace hpx::detail

#undef USE_CONCEPTS
#undef CMCSTL2_CONCEPTS
#undef CPP20_CONCEPTS
