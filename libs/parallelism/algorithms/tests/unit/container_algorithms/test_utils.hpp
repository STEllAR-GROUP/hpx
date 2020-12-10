//  Copyright (c) 2014-2015 Hartmut Kaiser
//  Copyright (c)      2018 Taeguk Kwon
//  Copyright (c)      2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/concepts/concepts.hpp>
#include <hpx/include/parallel_execution_policy.hpp>
#include <hpx/include/util.hpp>

#include <atomic>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace test {
    ///////////////////////////////////////////////////////////////////////////
    // Sentinel constructed from an Iterator just for the purpose of the
    // overloads tests
    template <typename IterType,
        HPX_CONCEPT_REQUIRES_(hpx::traits::is_iterator<IterType>::value)>
    struct sentinel_from_iterator
    {
        explicit sentinel_from_iterator(IterType end_iter)
          : end(end_iter)
        {
        }

        IterType get()
        {
            return end;
        }

        friend bool operator==(IterType i, sentinel_from_iterator<IterType> s)
        {
            return i == s.get();
        }

        friend bool operator==(sentinel_from_iterator<IterType> s, IterType i)
        {
            return i == s.get();
        }

        friend bool operator!=(IterType i, sentinel_from_iterator<IterType> s)
        {
            return i != s.get();
        }

        friend bool operator!=(sentinel_from_iterator<IterType> s, IterType i)
        {
            return i != s.get();
        }

    private:
        IterType end;
    };

    ///////////////////////////////////////////////////////////////////////////
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
        test_iterator(BaseIterator base)
          : base_type(base)
        {
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseContainer, typename IteratorTag>
    struct test_container : BaseContainer
    {
        template <typename... Ts>
        test_container(Ts&&... ts)
          : BaseContainer(std::forward<Ts>(ts)...)
        {
        }

        BaseContainer& base()
        {
            return *this;
        }
        BaseContainer const& base() const
        {
            return *this;
        }

        typedef test_iterator<typename BaseContainer::iterator, IteratorTag>
            iterator;
        typedef test_iterator<typename BaseContainer::const_iterator,
            IteratorTag>
            const_iterator;

        iterator begin()
        {
            return iterator(this->BaseContainer::begin());
        }
        const_iterator begin() const
        {
            return const_iterator(this->BaseContainer::begin());
        }
        const_iterator cbegin() const
        {
            return const_iterator(this->BaseContainer::cbegin());
        }

        iterator end()
        {
            return iterator(this->BaseContainer::end());
        }
        const_iterator end() const
        {
            return const_iterator(this->BaseContainer::end());
        }
        const_iterator cend() const
        {
            return const_iterator(this->BaseContainer::cend());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseIterator, typename IteratorTag>
    struct decorated_iterator
      : hpx::util::iterator_adaptor<
            decorated_iterator<BaseIterator, IteratorTag>, BaseIterator, void,
            IteratorTag>
    {
    private:
        typedef hpx::util::iterator_adaptor<
            decorated_iterator<BaseIterator, IteratorTag>, BaseIterator, void,
            IteratorTag>
            base_type;

    public:
        decorated_iterator() {}

        decorated_iterator(BaseIterator base)
          : base_type(base)
        {
        }

        decorated_iterator(BaseIterator base, std::function<void()> f)
          : base_type(base)
          , m_callback(f)
        {
        }

    private:
        friend class hpx::util::iterator_core_access;

        typename base_type::reference dereference() const
        {
            if (m_callback)
                m_callback();
            return *(this->base());
        }

    private:
        std::function<void()> m_callback;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct count_instances_v
    {
        count_instances_v()
        {
            ++instance_count;
            ++max_instance_count;
        }
        count_instances_v(T value)
          : value_(value)
        {
            ++instance_count;
            ++max_instance_count;
        }

        count_instances_v(count_instances_v const& rhs)
          : value_(rhs.value_)
        {
            ++instance_count;
        }
        count_instances_v(count_instances_v&& rhs)
          : value_(rhs.value_)
        {
            ++instance_count;
        }

        count_instances_v& operator=(count_instances_v const& rhs)
        {
            value_ = rhs.value_;
            return *this;
        }
        count_instances_v& operator=(count_instances_v&& rhs)
        {
            value_ = rhs.value_;
            return *this;
        }

        ~count_instances_v()
        {
            --instance_count;
        }

        T value_;
        static std::atomic<std::size_t> instance_count;
        static std::atomic<std::size_t> max_instance_count;
    };

    template <typename T>
    std::atomic<std::size_t> count_instances_v<T>::instance_count(0);

    template <typename T>
    std::atomic<std::size_t> count_instances_v<T>::max_instance_count(0);

    using count_instances = count_instances_v<std::size_t>;

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct test_num_exceptions_base
    {
        static void call(ExPolicy, hpx::exception_list const& e)
        {
            // The static partitioner uses four times the number of
            // threads/cores for the number chunks to create.
            HPX_TEST_LTE(e.size(), 4 * hpx::get_num_worker_threads());
        }
    };

    template <>
    struct test_num_exceptions_base<hpx::execution::sequenced_policy>
    {
        static void call(hpx::execution::sequenced_policy const&,
            hpx::exception_list const& e)
        {
            HPX_TEST_EQ(e.size(), 1u);
        }
    };

    template <typename ExPolicy, typename IteratorTag>
    struct test_num_exceptions : test_num_exceptions_base<ExPolicy>
    {
    };

    template <typename ExPolicy>
    struct test_num_exceptions<ExPolicy, std::input_iterator_tag>
    {
        static void call(ExPolicy, hpx::exception_list const& e)
        {
            HPX_TEST_EQ(e.size(), 1u);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    inline std::vector<std::size_t> iota(std::size_t size, std::size_t start)
    {
        std::vector<std::size_t> c(size);
        std::iota(std::begin(c), std::end(c), start);
        return c;
    }

    inline std::vector<std::size_t> random_iota(std::size_t size)
    {
        std::vector<std::size_t> c(size);
        std::iota(std::begin(c), std::end(c), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(std::begin(c), std::end(c), g);
        return c;
    }

    template <typename T>
    inline std::vector<T> random_iota(std::size_t size)
    {
        std::vector<T> c(size);
        std::iota(std::begin(c), std::end(c), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(std::begin(c), std::end(c), g);
        return c;
    }

    inline std::vector<std::size_t> random_fill(std::size_t size)
    {
        std::vector<std::size_t> c(size);
        std::generate(std::begin(c), std::end(c), std::rand);
        return c;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline void make_ready(
        std::vector<hpx::lcos::local::promise<std::size_t>>& p,
        std::vector<std::size_t>& idx)
    {
        std::for_each(std::begin(idx), std::end(idx),
            [&p](std::size_t i) { p[i].set_value(i); });
    }

    inline std::vector<hpx::future<std::size_t>> fill_with_futures(
        std::vector<hpx::lcos::local::promise<std::size_t>>& p)
    {
        std::vector<hpx::future<std::size_t>> f;
        std::transform(std::begin(p), std::end(p), std::back_inserter(f),
            [](hpx::lcos::local::promise<std::size_t>& pr) {
                return pr.get_future();
            });

        return f;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline std::vector<std::size_t> fill_all_any_none(
        std::size_t size, std::size_t num_filled)
    {
        if (num_filled == 0)
            return std::vector<std::size_t>(size, 0);

        if (num_filled == size)
            return std::vector<std::size_t>(size, 1);

        std::vector<std::size_t> c(size, 0);
        for (std::size_t i = 0; i < num_filled; /**/)
        {
            std::size_t pos = std::rand() % c.size();    //-V104
            if (c[pos])
                continue;

            c[pos] = 1;
            ++i;
        }
        return c;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename InputIter1, typename InputIter2>
    bool equal(InputIter1 first1, InputIter1 last1, InputIter2 first2,
        InputIter2 last2)
    {
        if (std::distance(first1, last1) != std::distance(first2, last2))
            return false;

        return std::equal(first1, last1, first2);
    }
}    // namespace test
