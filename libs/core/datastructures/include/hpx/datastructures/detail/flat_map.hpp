//  Copyright (C) 2019-2022 T. Zachary Laine
//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/detail/flat_set.hpp>

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

    template <typename T1, typename T2>
    struct ref_pair
    {
        static_assert(std::is_reference<T1>{} && std::is_reference<T2>{});

        using pair_type = std::pair<remove_cvref_t<T1>, remove_cvref_t<T2>>;
        using pair_of_references_type =
            std::pair<remove_cvref_t<T1>&, remove_cvref_t<T2>&>;
        using const_pair_of_references_type =
            std::pair<remove_cvref_t<T1> const&, remove_cvref_t<T2> const&>;

        ref_pair(T1 t1, T2 t2)
          : first(t1)
          , second(t2)
        {
        }
        ref_pair(ref_pair const& other)
          : first(other.first)
          , second(other.second)
        {
        }
        ref_pair(ref_pair&& other) noexcept
          : first(other.first)
          , second(other.second)
        {
        }
        ref_pair& operator=(ref_pair const& other)
        {
            first = other.first;
            second = other.second;
            return *this;
        }
        ref_pair& operator=(ref_pair&& other) noexcept
        {
            first = other.first;
            second = other.second;
            return *this;
        }
        ref_pair& operator=(pair_type const& other)
        {
            first = other.first;
            second = other.second;
            return *this;
        }
        ref_pair& operator=(pair_type&& other) noexcept
        {
            first = HPX_MOVE(other.first);
            second = HPX_MOVE(other.second);
            return *this;
        }
        ~ref_pair() = default;

        operator pair_type() const
        {
            return pair_type(first, second);
        }
        operator pair_of_references_type() const
        {
            return pair_of_references_type(first, second);
        }
        operator const_pair_of_references_type() const
        {
            return const_pair_of_references_type(first, second);
        }
        bool operator==(ref_pair rhs) const noexcept
        {
            return first == rhs.first && second == rhs.second;
        }
        bool operator!=(ref_pair rhs) const noexcept
        {
            return !(*this == rhs);
        }
        bool operator<(ref_pair rhs) const noexcept
        {
            if (first < rhs.first)
                return true;
            if (rhs.first < first)
                return false;
            return second < rhs.second;
        }

        T1 first;
        T2 second;
    };

    template <typename T1, typename T2>
    void swap(ref_pair<T1, T2> const& lhs, ref_pair<T1, T2> const& rhs)
    {
        using std::swap;
        swap(lhs.first, rhs.first);
        swap(lhs.second, rhs.second);
    }

    template <typename KeyRef, typename TRef, typename KeyIter,
        typename MappedIter>
    struct flat_map_iterator
    {
        static_assert(std::is_reference<KeyRef>{} && std::is_reference<TRef>{});

        using iterator_category = std::random_access_iterator_tag;
        using value_type =
            std::pair<remove_cvref_t<KeyRef>, remove_cvref_t<TRef>>;
        using difference_type =
            typename std::iterator_traits<KeyIter>::difference_type;
        using reference = ref_pair<KeyRef, TRef>;

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

        flat_map_iterator() = default;
        flat_map_iterator(KeyIter key_it, MappedIter mapped_it)
          : key_it_(key_it)
          , mapped_it_(mapped_it)
        {
        }

        template <typename TRef2, typename MappedIter2,
            typename = std::enable_if_t<std::is_convertible_v<TRef2, TRef> &&
                std::is_convertible_v<MappedIter2, MappedIter>>>
        flat_map_iterator(
            flat_map_iterator<KeyRef, TRef2, KeyIter, MappedIter2> other)
          : key_it_(other.key_it_)
          , mapped_it_(other.mapped_it_)
        {
        }

        constexpr reference operator*() const noexcept
        {
            return ref();
        }
        constexpr pointer operator->() const noexcept
        {
            return arrow_proxy(ref());
        }

        constexpr reference operator[](difference_type n) const noexcept
        {
            return reference(*(key_it_ + n), *(mapped_it_ + n));
        }

        flat_map_iterator operator+(difference_type n) const noexcept
        {
            return flat_map_iterator(key_it_ + n, mapped_it_ + n);
        }
        flat_map_iterator operator-(difference_type n) const noexcept
        {
            return flat_map_iterator(key_it_ - n, mapped_it_ - n);
        }

        flat_map_iterator& operator++() noexcept
        {
            ++key_it_;
            ++mapped_it_;
            return *this;
        }
        flat_map_iterator operator++(int) noexcept
        {
            flat_map_iterator tmp(*this);
            ++key_it_;
            ++mapped_it_;
            return tmp;
        }

        flat_map_iterator& operator--() noexcept
        {
            --key_it_;
            --mapped_it_;
            return *this;
        }
        flat_map_iterator operator--(int) noexcept
        {
            flat_map_iterator tmp(*this);
            --key_it_;
            --mapped_it_;
            return tmp;
        }

        flat_map_iterator& operator+=(difference_type n) noexcept
        {
            key_it_ += n;
            mapped_it_ += n;
            return *this;
        }
        flat_map_iterator& operator-=(difference_type n) noexcept
        {
            key_it_ -= n;
            mapped_it_ -= n;
            return *this;
        }

        KeyIter key_iter() const
        {
            return key_it_;
        }
        MappedIter mapped_iter() const
        {
            return mapped_it_;
        }

        friend bool operator==(flat_map_iterator lhs, flat_map_iterator rhs)
        {
            return lhs.key_it_ == rhs.key_it_;
        }
        friend bool operator!=(flat_map_iterator lhs, flat_map_iterator rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator<(flat_map_iterator lhs, flat_map_iterator rhs)
        {
            return lhs.key_it_ < rhs.key_it_;
        }
        friend bool operator<=(flat_map_iterator lhs, flat_map_iterator rhs)
        {
            return lhs == rhs || lhs < rhs;
        }
        friend bool operator>(flat_map_iterator lhs, flat_map_iterator rhs)
        {
            return rhs < lhs;
        }
        friend bool operator>=(flat_map_iterator lhs, flat_map_iterator rhs)
        {
            return lhs == rhs || rhs < lhs;
        }

        friend difference_type operator-(
            flat_map_iterator lhs, flat_map_iterator rhs)
        {
            return lhs.key_it_ - rhs.key_it_;
        }

    private:
        template <typename KeyRef2, class TRef2, typename KeyIter2,
            typename MappedIter2>
        friend struct flat_map_iterator;

        reference ref() const
        {
            return reference(*key_it_, *mapped_it_);
        }

        KeyIter key_it_;
        MappedIter mapped_it_;
    };

    // NOTE: This overload was necessary, since iter_swap(it1, it2) calls
    // swap(*it1, *it2).  All std::swap() overloads expect lvalues, and
    // flat_map's iterators produce proxy rvalues when dereferenced.
    template <typename KeyRef, class TRef>
    inline void swap(
        std::pair<KeyRef, TRef>&& lhs, std::pair<KeyRef, TRef>&& rhs)
    {
        using std::swap;
        swap(lhs.first, rhs.first);
        swap(lhs.second, rhs.second);
    }

    template <typename Key, typename T, typename Compare = std::less<Key>,
        typename KeyContainer = std::vector<Key>,
        typename MappedContainer = std::vector<T>>
    class flat_map
    {
        template <typename Alloc>
        using uses =
            std::enable_if_t<std::uses_allocator<KeyContainer, Alloc>::value &&
                std::uses_allocator<MappedContainer, Alloc>::value>;

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
        using mapped_type = T;
        using value_type = std::pair<key_type const, mapped_type>;
        using key_compare = Compare;
        using reference = std::pair<key_type const&, mapped_type&>;
        using const_reference = std::pair<key_type const&, mapped_type const&>;
        using size_type = std::size_t;
        using difference_type = ptrdiff_t;
        using iterator = flat_map_iterator<key_type const&, mapped_type&,
            typename KeyContainer::const_iterator,
            typename MappedContainer::iterator>;    // see 21.2
        using const_iterator = flat_map_iterator<key_type const&,
            mapped_type const&, typename KeyContainer::const_iterator,
            typename MappedContainer::const_iterator>;    // see 21.2
        using reverse_iterator = flat_map_iterator<key_type const&,
            mapped_type&, typename KeyContainer::const_reverse_iterator,
            typename MappedContainer::reverse_iterator>;    // see 21.2
        using const_reverse_iterator = flat_map_iterator<key_type const&,
            mapped_type const&, typename KeyContainer::const_reverse_iterator,
            typename MappedContainer::const_reverse_iterator>;    // see 21.2
        using key_container_type = KeyContainer;
        using mapped_container_type = MappedContainer;

        class value_compare
        {
            friend flat_map;

        private:
            key_compare comp;
            explicit value_compare(key_compare c)
              : comp(c)
            {
            }

        public:
            bool operator()(const_reference x, const_reference y) const
            {
                return comp(x.first, y.first);
            }
        };

        struct containers
        {
            key_container_type keys;
            mapped_container_type values;
        };

        // ??, construct/copy/destroy
        flat_map()
          : flat_map(key_compare())
        {
        }
        flat_map(key_container_type key_cont, mapped_container_type mapped_cont)
          : c{HPX_MOVE(key_cont), HPX_MOVE(mapped_cont)}
          , compare(key_compare())
        {
            mutable_iterator first(c.keys.begin(), c.values.begin());
            mutable_iterator last(c.keys.end(), c.values.end());
#if USE_CONCEPTS
            std::ranges::sort(first, last, value_comp());
#else
            std::sort(first, last, value_comp());
#endif
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_map(key_container_type const& key_cont,
            mapped_container_type const& mapped_cont, Alloc const& a)
          : c{key_container_type(key_cont, a),
                mapped_container_type(mapped_cont, a)}
          , compare()
        {
            mutable_iterator first(c.keys.begin(), c.values.begin());
            mutable_iterator last(c.keys.end(), c.values.end());
#if USE_CONCEPTS
            std::ranges::sort(first, last, value_comp());
#else
            std::sort(first, last, value_comp());
#endif
        }
        template <typename Container, typename Enable = container<Container>>
        explicit flat_map(
            Container const& cont, key_compare const& comp = key_compare())
          : flat_map(std::begin(cont), std::end(cont), comp)
        {
        }
        template <typename Container, typename Alloc,
            typename Enable1 = container<Container>,
            typename Enable2 = uses<Alloc>>
        flat_map(Container const& cont, Alloc const& a)
          : flat_map(std::begin(cont), std::end(cont), a)
        {
        }
        flat_map(sorted_unique_t, key_container_type key_cont,
            mapped_container_type mapped_cont)
          : c{HPX_MOVE(key_cont), HPX_MOVE(mapped_cont)}
          , compare(key_compare())
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_map(sorted_unique_t, key_container_type const& key_cont,
            mapped_container_type const& mapped_cont, Alloc const& a)
          : c{key_container_type(key_cont, a),
                mapped_container_type(mapped_cont, a)}
          , compare()
        {
        }
        template <typename Container, typename Enable = container<Container>>
        flat_map(sorted_unique_t s, Container const& cont,
            key_compare const& comp = key_compare())
          : flat_map(s, std::begin(cont), std::end(cont), comp)
        {
        }
        template <typename Container, typename Alloc,
            typename Enable1 = container<Container>,
            typename Enable2 = uses<Alloc>>
        flat_map(sorted_unique_t s, Container const& cont, Alloc const& a)
          : c{key_container_type(a), mapped_container_type(a)}
          , compare()
        {
            c.keys.reserve(cont.size());
            c.values.reserve(cont.size());
            insert(s, std::begin(cont), std::end(cont));
        }
        explicit flat_map(key_compare const& comp)
          : c()
          , compare(comp)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_map(key_compare const& comp, Alloc const& a)
          : c{key_container_type(a), mapped_container_type(a)}
          , compare(comp)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        explicit flat_map(Alloc const& a)
          : c{key_container_type(a), mapped_container_type(a)}
          , compare()
        {
        }
        template <typename InputIterator>
        flat_map(InputIterator first, InputIterator last,
            key_compare const& comp = key_compare())
          : c()
          , compare(comp)
        {
            insert(first, last);
        }
        template <typename InputIterator, typename Alloc,
            typename Enable = uses<Alloc>>
        flat_map(InputIterator first, InputIterator last,
            key_compare const& comp, Alloc const& a)
          : c{key_container_type(a), mapped_container_type(a)}
          , compare(comp)
        {
            insert(first, last);
        }
        template <typename InputIterator, typename Alloc,
            typename Enable = uses<Alloc>>
        flat_map(InputIterator first, InputIterator last, Alloc const& a)
          : flat_map(first, last, key_compare(), a)
        {
        }
        template <typename InputIterator>
        flat_map(sorted_unique_t s, InputIterator first, InputIterator last,
            key_compare const& comp = key_compare())
          : c()
          , compare(comp)
        {
            insert(s, first, last);
        }
        template <typename InputIterator, typename Alloc,
            typename Enable = uses<Alloc>>
        flat_map(sorted_unique_t s, InputIterator first, InputIterator last,
            key_compare const& comp, Alloc const& a)
          : c{key_container_type(a), mapped_container_type(a)}
          , compare(comp)
        {
            insert(s, first, last);
        }
        template <typename InputIterator, typename Alloc,
            typename Enable = uses<Alloc>>
        flat_map(sorted_unique_t s, InputIterator first, InputIterator last,
            Alloc const& a)
          : flat_map(s, first, last, key_compare(), a)
        {
        }
        flat_map(std::initializer_list<value_type>&& il,
            key_compare const& comp = key_compare())
          : flat_map(il, comp)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_map(std::initializer_list<value_type>&& il,
            key_compare const& comp, Alloc const& a)
          : flat_map(std::begin(il), std::end(il), comp, a)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_map(std::initializer_list<value_type>&& il, Alloc const& a)
          : flat_map(std::begin(il), std::end(il), key_compare(), a)
        {
        }
        flat_map(sorted_unique_t s, std::initializer_list<value_type>&& il,
            key_compare const& comp = key_compare())
          : flat_map(s, il, comp)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_map(sorted_unique_t s, std::initializer_list<value_type>&& il,
            key_compare const& comp, Alloc const& a)
          : flat_map(s, std::begin(il), std::end(il), comp, a)
        {
        }
        template <typename Alloc, typename Enable = uses<Alloc>>
        flat_map(sorted_unique_t s, std::initializer_list<value_type>&& il,
            Alloc const& a)
          : flat_map(s, std::begin(il), std::end(il), key_compare(), a)
        {
        }
        flat_map& operator=(std::initializer_list<value_type> il)
        {
            flat_map tmp(il, compare);
            swap(tmp);
            return *this;
        }

        // iterators
        iterator begin() noexcept
        {
            return iterator(c.keys.begin(), c.values.begin());
        }
        const_iterator begin() const noexcept
        {
            return const_iterator(c.keys.begin(), c.values.begin());
        }
        iterator end() noexcept
        {
            return iterator(c.keys.end(), c.values.end());
        }
        const_iterator end() const noexcept
        {
            return const_iterator(c.keys.end(), c.values.end());
        }
        reverse_iterator rbegin() noexcept
        {
            return reverse_iterator(c.keys.rbegin(), c.values.rbegin());
        }
        const_reverse_iterator rbegin() const noexcept
        {
            return const_reverse_iterator(c.keys.rbegin(), c.values.rbegin());
        }
        reverse_iterator rend() noexcept
        {
            return reverse_iterator(c.keys.rend(), c.values.rend());
        }
        const_reverse_iterator rend() const noexcept
        {
            return const_reverse_iterator(c.keys.rend(), c.values.rend());
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
            return c.keys.empty();
        }
        size_type size() const noexcept
        {
            return c.keys.size();
        }
        size_type max_size() const noexcept
        {
            return std::min<size_type>(c.keys.max_size(), c.values.max_size());
        }

        // ??, element access
        mapped_type& operator[](key_type const& x)
        {
            return try_emplace(x).first->second;
        }
        mapped_type& operator[](key_type&& x)
        {
            return try_emplace(HPX_MOVE(x)).first->second;
        }
        mapped_type& at(key_type const& x)
        {
            auto it = key_find(x);
            if (it == c.keys.end())
                throw std::out_of_range("Value not found by flat_map.at()");
            return *project(it);
        }
        mapped_type const& at(key_type const& x) const
        {
            auto it = key_find(x);
            if (it == c.keys.end())
                throw std::out_of_range("Value not found by flat_map.at()");
            return *project(it);
        }

        // ??, modifiers
        template <typename... Args,
            typename Enable = std::enable_if_t<std::is_constructible_v<
                std::pair<key_type, mapped_type>, Args&&...>>>
        std::pair<iterator, bool> emplace(Args&&... args)
        {
            std::pair<key_type, mapped_type> p(HPX_FORWARD(Args, args)...);
            return try_emplace(HPX_MOVE(p.first), HPX_MOVE(p.second));
        }
        template <typename... Args,
            typename Enable = std::enable_if_t<std::is_constructible_v<
                std::pair<key_type, mapped_type>, Args&&...>>>
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
            typename Enable = std::enable_if_t<
                std::is_constructible_v<std::pair<key_type, mapped_type>, P&&>>>
        std::pair<iterator, bool> insert(P&& x)
        {
            return emplace(HPX_FORWARD(P, x));
        }
        template <typename P,
            typename Enable = std::enable_if_t<
                std::is_constructible_v<std::pair<key_type, mapped_type>, P&&>>>
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
                c.keys.push_back(it->first);
                c.values.push_back(it->second);
            }

            mutable_iterator inserted_first(
                c.keys.begin() + prev_size, c.values.begin() + prev_size);
            mutable_iterator inserted_last(c.keys.end(), c.values.end());
#if USE_CONCEPTS
            std::ranges::sort(inserted_first, inserted_last, value_comp());
#else
            std::sort(inserted_first, inserted_last, value_comp());
#endif

            if (!prev_size)
                return;

            mutable_iterator mutable_first(c.keys.begin(), c.values.begin());
#if USE_CONCEPTS
            std::ranges::inplace_merge(
                mutable_first, inserted_first, inserted_last, value_comp());
#else
            std::inplace_merge(
                mutable_first, inserted_first, inserted_last, value_comp());
#endif
        }
        template <typename InputIterator>
        void insert(sorted_unique_t, InputIterator first, InputIterator last)
        {
            auto const prev_size = size();
            for (auto it = first; it != last; ++it)
            {
                c.keys.push_back(it->first);
                c.values.push_back(it->second);
            }

            mutable_iterator inserted_first(
                c.keys.begin() + prev_size, c.values.begin() + prev_size);
            mutable_iterator inserted_last(c.keys.end(), c.values.end());

            if (!prev_size)
                return;

            mutable_iterator mutable_first(c.keys.begin(), c.values.begin());
#if USE_CONCEPTS
            std::ranges::inplace_merge(
                mutable_first, inserted_first, inserted_last, value_comp());
#else
            std::inplace_merge(
                mutable_first, inserted_first, inserted_last, value_comp());
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

        containers extract() &&
        {
            scoped_clear _(this);
            return HPX_MOVE(c);
        }
        void replace(
            key_container_type&& key_cont, mapped_container_type&& mapped_cont)
        {
            scoped_clear _(this);
            c.keys = HPX_MOVE(key_cont);
            c.values = HPX_MOVE(mapped_cont);
            _.release();
        }

        template <typename... Args,
            typename Enable = std::enable_if_t<
                std::is_constructible_v<mapped_type, Args&&...>>>
        std::pair<iterator, bool> try_emplace(key_type const& k, Args&&... args)
        {
            auto it = key_lower_bound(k);
            if (it == c.keys.end() || compare(*it, k) || compare(k, *it))
            {
                auto values_it =
                    c.values.emplace(project(it), HPX_FORWARD(Args, args)...);
                it = c.keys.insert(it, k);
                return std::pair<iterator, bool>(iterator(it, values_it), true);
            }
            return std::pair<iterator, bool>(iterator(it, project(it)), false);
        }
        template <typename... Args,
            typename Enable = std::enable_if_t<
                std::is_constructible_v<mapped_type, Args&&...>>>
        std::pair<iterator, bool> try_emplace(key_type&& k, Args&&... args)
        {
            auto it = key_lower_bound(k);
            if (it == c.keys.end() || compare(*it, k) || compare(k, *it))
            {
                auto values_it =
                    c.values.emplace(project(it), HPX_FORWARD(Args, args)...);
                it = c.keys.insert(it, HPX_FORWARD(key_type, k));
                return std::pair<iterator, bool>(iterator(it, values_it), true);
            }
            return std::pair<iterator, bool>(iterator(it, project(it)), false);
        }
        template <typename... Args,
            typename Enable = std::enable_if_t<
                std::is_constructible_v<mapped_type, Args&&...>>>
        iterator try_emplace(const_iterator, key_type const& k, Args&&... args)
        {
            return try_emplace(k, HPX_FORWARD(Args, args)...).first;
        }
        template <typename... Args,
            typename Enable = std::enable_if_t<
                std::is_constructible_v<mapped_type, Args&&...>>>
        iterator try_emplace(const_iterator, key_type&& k, Args&&... args)
        {
            return try_emplace(
                HPX_FORWARD(key_type, k), HPX_FORWARD(Args, args)...)
                .first;
        }

        template <typename M,
            typename Enable =
                std::enable_if_t<std::is_assignable_v<mapped_type&, M> &&
                    std::is_constructible_v<mapped_type, M&&>>>
        std::pair<iterator, bool> insert_or_assign(key_type const& k, M&& obj)
        {
            auto it = key_lower_bound(k);
            if (it == c.keys.end() || compare(*it, k) || compare(k, *it))
            {
                auto values_it =
                    c.values.insert(project(it), HPX_FORWARD(M, obj));
                it = c.keys.insert(it, k);
                return std::pair<iterator, bool>(iterator(it, values_it), true);
            }
            auto values_it = project(it);
            *values_it = HPX_FORWARD(M, obj);
            return std::pair<iterator, bool>(iterator(it, values_it), false);
        }
        template <typename M,
            typename Enable =
                std::enable_if_t<std::is_assignable_v<mapped_type&, M> &&
                    std::is_constructible_v<mapped_type, M&&>>>
        std::pair<iterator, bool> insert_or_assign(key_type&& k, M&& obj)
        {
            auto it = key_lower_bound(k);
            if (it == c.keys.end() || compare(*it, k) || compare(k, *it))
            {
                auto values_it =
                    c.values.insert(project(it), HPX_FORWARD(M, obj));
                it = c.keys.insert(it, HPX_FORWARD(key_type, k));
                return std::pair<iterator, bool>(iterator(it, values_it), true);
            }
            auto values_it = project(it);
            *values_it = HPX_FORWARD(M, obj);
            return std::pair<iterator, bool>(iterator(it, values_it), false);
        }
        template <typename M,
            typename Enable =
                std::enable_if_t<std::is_assignable_v<mapped_type&, M> &&
                    std::is_constructible_v<mapped_type, M&&>>>
        iterator insert_or_assign(const_iterator, key_type const& k, M&& obj)
        {
            return insert_or_assign(k, HPX_FORWARD(M, obj)).first;
        }
        template <typename M,
            typename Enable =
                std::enable_if_t<std::is_assignable_v<mapped_type&, M> &&
                    std::is_constructible_v<mapped_type, M&&>>>
        iterator insert_or_assign(const_iterator, key_type&& k, M&& obj)
        {
            return insert_or_assign(
                HPX_FORWARD(key_type, k), HPX_FORWARD(M, obj))
                .first;
        }

        iterator erase(iterator position)
        {
            return iterator(c.keys.erase(position.key_iter()),
                c.values.erase(position.mapped_iter()));
        }
        iterator erase(const_iterator position)
        {
            return iterator(c.keys.erase(position.key_iter()),
                c.values.erase(position.mapped_iter()));
        }
        size_type erase(key_type const& x)
        {
            auto it = key_find(x);
            if (it == c.keys.end())
                return static_cast<size_type>(0);
            c.values.erase(project(it));
            c.keys.erase(it);
            return static_cast<size_type>(1);
        }
        iterator erase(const_iterator first, const_iterator last)
        {
            return iterator(c.keys.erase(first.key_iter(), last.key_iter()),
                c.values.erase(first.mapped_iter(), last.mapped_iter()));
        }

        void swap(flat_map& fm) noexcept(
            std::is_nothrow_swappable_v<key_compare>)
        {
            using std::swap;
            swap(compare, fm.compare);
            swap(c.keys, fm.c.keys);
            swap(c.values, fm.c.values);
        }
        void clear() noexcept
        {
            c.keys.clear();
            c.values.clear();
        }

        // observers
        key_compare key_comp() const
        {
            return compare;
        }
        value_compare value_comp() const
        {
            return value_compare(compare);
        }
        key_container_type const& keys() const noexcept
        {
            return c.keys;
        }
        mapped_container_type const& values() const noexcept
        {
            return c.values;
        }

        // map operations
        iterator find(key_type const& x)
        {
            auto it = key_find(x);
            return iterator(it, project(it));
        }
        const_iterator find(key_type const& x) const
        {
            auto it = key_find(x);
            return const_iterator(it, project(it));
        }
        template <typename K>
        iterator find(K const& x)
        {
            auto it = key_find(x);
            return iterator(it, project(it));
        }
        template <typename K>
        const_iterator find(K const& x) const
        {
            auto it = key_find(x);
            return iterator(it, project(it));
        }
        size_type count(key_type const& x) const
        {
            auto it = key_find(x);
            return static_cast<size_type>(it == c.keys.end() ? 0 : 1);
        }
        template <typename K>
        size_type count(K const& x) const
        {
            auto it = key_find(x);
            return static_cast<size_type>(it == c.keys.end() ? 0 : 1);
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
            return iterator(it, project(it));
        }
        const_iterator lower_bound(key_type const& x) const
        {
            auto it = key_lower_bound(x);
            return const_iterator(it, project(it));
        }
        template <typename K>
        iterator lower_bound(K const& x)
        {
            auto it = key_lower_bound(x);
            return iterator(it, project(it));
        }
        template <typename K>
        const_iterator lower_bound(K const& x) const
        {
            auto it = key_lower_bound(x);
            return const_iterator(it, project(it));
        }
        iterator upper_bound(key_type const& x)
        {
            auto it = key_upper_bound(x);
            return iterator(it, project(it));
        }
        const_iterator upper_bound(key_type const& x) const
        {
            auto it = key_upper_bound(x);
            return const_iterator(it, project(it));
        }
        template <typename K>
        iterator upper_bound(K const& x)
        {
            auto it = key_upper_bound(x);
            return iterator(it, project(it));
        }
        template <typename K>
        const_iterator upper_bound(K const& x) const
        {
            auto it = key_upper_bound(x);
            return const_iterator(it, project(it));
        }
        std::pair<iterator, iterator> equal_range(key_type const& k)
        {
            iterator const first = lower_bound(k);
            iterator const last = find_if(first, end(),
                [&](auto const& x) { return compare(k, x.first); });
            return std::pair<iterator, iterator>(first, last);
        }
        std::pair<const_iterator, const_iterator> equal_range(
            key_type const& k) const
        {
            const_iterator const first = lower_bound(k);
            const_iterator const last = find_if(first, end(),
                [&](auto const& x) { return compare(k, x.first); });
            return std::pair<const_iterator, const_iterator>(first, last);
        }
        template <typename K>
        std::pair<iterator, iterator> equal_range(K const& k)
        {
            iterator const first = lower_bound(k);
            iterator const last = find_if(first, end(),
                [&](auto const& x) { return compare(k, x.first); });
            return std::pair<iterator, iterator>(first, last);
        }
        template <typename K>
        std::pair<const_iterator, const_iterator> equal_range(K const& k) const
        {
            const_iterator const first = lower_bound(k);
            const_iterator const last = find_if(first, end(),
                [&](auto const& x) { return compare(k, x.first); });
            return std::pair<const_iterator, const_iterator>(first, last);
        }

        friend bool operator==(flat_map const& x, flat_map const& y)
        {
#if USE_CONCEPTS
            return std::ranges::equal(x, y);
#else
            return std::equal(x.begin(), x.end(), y.begin(), y.end());
#endif
        }
        friend bool operator!=(flat_map const& x, flat_map const& y)
        {
            return !(x == y);
        }
        friend bool operator<(flat_map const& x, flat_map const& y)
        {
#if USE_CONCEPTS
            return std::ranges::lexicographical_compare(
                x, y, [](auto lhs, auto rhs) { return lhs < rhs; });
#else
            return std::lexicographical_compare(
                x.begin(), x.end(), y.begin(), y.end());
#endif
        }
        friend bool operator>(flat_map const& x, flat_map const& y)
        {
            return y < x;
        }
        friend bool operator<=(flat_map const& x, flat_map const& y)
        {
            return !(y < x);
        }
        friend bool operator>=(flat_map const& x, flat_map const& y)
        {
            return !(x < y);
        }

        friend void swap(flat_map& x, flat_map& y) noexcept(noexcept(x.swap(y)))
        {
            return x.swap(y);
        }

    private:
        containers c;           // exposition only
        key_compare compare;    // exposition only
        // exposition only
        struct scoped_clear
        {
            explicit scoped_clear(flat_map* fm)
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
            flat_map* fm_;
        };

        using key_iter_t = typename KeyContainer::iterator;
        using key_const_iter_t = typename KeyContainer::const_iterator;
        using mapped_iter_t = typename MappedContainer::iterator;
        using mapped_const_iter_t = typename MappedContainer::const_iterator;

        using mutable_iterator = flat_map_iterator<key_type&, mapped_type&,
            key_iter_t, mapped_iter_t>;

        mapped_iter_t project(key_iter_t key_it)
        {
            return c.values.begin() + (key_it - c.keys.begin());
        }
        mapped_const_iter_t project(key_const_iter_t key_it) const
        {
            return c.values.begin() + (key_it - c.keys.begin());
        }

        template <typename K>
        key_iter_t key_lower_bound(K const& k)
        {
#if USE_CONCEPTS
            return std::ranges::lower_bound(c.keys, k, compare);
#else
            return std::lower_bound(c.keys.begin(), c.keys.end(), k, compare);
#endif
        }
        template <typename K>
        key_const_iter_t key_lower_bound(K const& k) const
        {
#if USE_CONCEPTS
            return ranges::lower_bound(c.keys, k, compare);
#else
            return std::lower_bound(c.keys.begin(), c.keys.end(), k, compare);
#endif
        }
        template <typename K>
        key_iter_t key_upper_bound(K const& k)
        {
#if USE_CONCEPTS
            return ranges::upper_bound(c.keys, k, compare);
#else
            return std::upper_bound(c.keys.begin(), c.keys.end(), k, compare);
#endif
        }
        template <typename K>
        key_const_iter_t key_upper_bound(K const& k) const
        {
#if USE_CONCEPTS
            return ranges::upper_bound(c.keys, k, compare);
#else
            return std::upper_bound(c.keys.begin(), c.keys.end(), k, compare);
#endif
        }
        template <typename K>
        key_iter_t key_find(K const& k)
        {
            auto it = key_lower_bound(k);
            if (it != c.keys.end() && (compare(*it, k) || compare(k, *it)))
                it = c.keys.end();
            return it;
        }
        template <typename K>
        key_const_iter_t key_find(K const& k) const
        {
            auto it = key_lower_bound(k);
            if (it != c.keys.end() && (compare(*it, k) || compare(k, *it)))
                it = c.keys.end();
            return it;
        }
    };
}    // namespace hpx::detail

#undef USE_CONCEPTS
#undef CMCSTL2_CONCEPTS
#undef CPP20_CONCEPTS
