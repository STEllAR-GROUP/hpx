//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_ITERATOR_MAY_29_2014_0110PM)
#define HPX_PARALLEL_TEST_ITERATOR_MAY_29_2014_0110PM

#include <hpx/include/parallel_execution_policy.hpp>
#include <hpx/include/util.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

namespace test
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseIterator, typename IteratorTag>
    struct test_iterator
      : hpx::util::iterator_adaptor<
            test_iterator<BaseIterator, IteratorTag>,
            BaseIterator, void, IteratorTag>
    {
    private:
        typedef hpx::util::iterator_adaptor<
            test_iterator<BaseIterator, IteratorTag>,
            BaseIterator, void, IteratorTag>
        base_type;

    public:
        test_iterator() : base_type() {}
        test_iterator(BaseIterator base) : base_type(base) {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseIterator, typename IteratorTag>
    struct decorated_iterator
      : hpx::util::iterator_adaptor<
            decorated_iterator<BaseIterator, IteratorTag>,
            BaseIterator, void, IteratorTag>
    {
    private:
        typedef hpx::util::iterator_adaptor<
            decorated_iterator<BaseIterator, IteratorTag>,
            BaseIterator, void, IteratorTag>
        base_type;

    public:
        decorated_iterator()
        {}

        decorated_iterator(BaseIterator base)
          : base_type(base)
        {}

        decorated_iterator(BaseIterator base, std::function<void()> f)
          : base_type(base), m_callback(f)
        {}

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
        count_instances_v(count_instances_v && rhs)
          : value_(rhs.value_)
        {
            ++instance_count;
        }

        count_instances_v& operator=(count_instances_v const& rhs)
        {
            value_ = rhs.value_;
            return *this;
        }
        count_instances_v& operator=(count_instances_v && rhs)
        {
            value_ = rhs.value_;
            return *this;
        }

        ~count_instances_v()
        {
            --instance_count;
        }

        T value_;
        static boost::atomic<std::size_t> instance_count;
        static boost::atomic<std::size_t> max_instance_count;
    };

    template <typename T>
    boost::atomic<std::size_t> count_instances_v<T>::instance_count(0);

    template <typename T>
    boost::atomic<std::size_t> count_instances_v<T>::max_instance_count(0);

    using count_instances = count_instances_v<std::size_t>;

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename IteratorTag>
    struct test_num_exceptions
    {
        static void call(ExPolicy, hpx::exception_list const& e)
        {
            // The static partitioner uses four times the number of
            // threads/cores for the number chunks to create.
            HPX_TEST_LTE(e.size(), 4 * hpx::get_num_worker_threads());
        }
    };

    template <typename IteratorTag>
    struct test_num_exceptions<
        hpx::parallel::execution::sequenced_policy, IteratorTag>
    {
        static void call(hpx::parallel::execution::sequenced_policy const&,
            hpx::exception_list const& e)
        {
            HPX_TEST_EQ(e.size(), 1u);
        }
    };

    template <typename ExPolicy>
    struct test_num_exceptions<ExPolicy, std::input_iterator_tag>
    {
        static void call(ExPolicy, hpx::exception_list const& e)
        {
            HPX_TEST_EQ(e.size(), 1u);
        }
    };

    template <>
    struct test_num_exceptions<
        hpx::parallel::execution::sequenced_policy, std::input_iterator_tag>
    {
        static void call(hpx::parallel::execution::sequenced_policy const&,
            hpx::exception_list const& e)
        {
            HPX_TEST_EQ(e.size(), 1u);
        }
    };

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    template <typename IteratorTag>
    struct test_num_exceptions<hpx::parallel::execution_policy, IteratorTag>
    {
        static void call(hpx::parallel::execution_policy const& policy,
            hpx::exception_list const& e)
        {
            using namespace hpx::parallel::v1::detail;

            if (policy.type() == typeid(hpx::parallel::execution::sequenced_policy))
            {
                HPX_TEST_EQ(e.size(), 1u);
            }
            else
            {
                // The static partitioner uses the number of threads/cores for
                // the number chunks to create.
                HPX_TEST_LTE(e.size(), hpx::get_num_worker_threads());
            }
        }
    };

    template <>
    struct test_num_exceptions<
        hpx::parallel::execution_policy, std::input_iterator_tag>
    {
        static void call(hpx::parallel::execution_policy const&,
            hpx::exception_list const& e)
        {
            HPX_TEST_EQ(e.size(), 1u);
        }
    };
#endif

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
        std::random_shuffle(std::begin(c), std::end(c));
        return c;
    }

    template <typename T>
    inline std::vector<T> random_iota(std::size_t size)
    {
        std::vector<T> c(size);
        std::iota(std::begin(c), std::end(c), 0);
        std::random_shuffle(std::begin(c), std::end(c));
        return c;
    }

    inline std::vector<std::size_t> random_fill(std::size_t size)
    {
        std::vector<std::size_t> c(size);
        std::generate(std::begin(c), std::end(c), std::rand);
        return c;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline void make_ready(std::vector<hpx::promise<std::size_t> >& p,
        std::vector<std::size_t>& idx)
    {
        std::for_each(std::begin(idx), std::end(idx),
            [&p](std::size_t i)
            {
                p[i].set_value(i);
            });
    }

    inline std::vector<hpx::future<std::size_t> > fill_with_futures(
        std::vector<hpx::promise<std::size_t> >& p)
    {
        std::vector<hpx::future<std::size_t> > f;
        std::transform(std::begin(p), std::end(p), std::back_inserter(f),
            [](hpx::promise<std::size_t>& pr)
            {
                return pr.get_future();
            });

        return f;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline std::vector<std::size_t>
    fill_all_any_none(std::size_t size, std::size_t num_filled)
    {
        if (num_filled == 0)
            return std::vector<std::size_t>(size, 0);

        if (num_filled == size)
            return std::vector<std::size_t>(size, 1);

        std::vector<std::size_t> c(size, 0);
        for (std::size_t i = 0; i < num_filled; /**/)
        {
            std::size_t pos = std::rand() % c.size(); //-V104
            if (c[pos])
                continue;

            c[pos] = 1;
            ++i;
        }
        return c;
    }
}

#endif
