//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/transform_iterator.hpp>
#include <hpx/include/iostreams.hpp>

#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/range/irange.hpp>
#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
int test_count = 100;
int partition_size = 10000;

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace experimental { namespace detail
{
    template <typename Iterator>
    HPX_FORCEINLINE
    Iterator previous(Iterator it, std::false_type)
    {
        return --it;
    }

    template <typename Iterator>
    HPX_FORCEINLINE
    Iterator previous(Iterator const& it, std::true_type)
    {
        return it - 1;
    }

    template <typename Iterator>
    HPX_FORCEINLINE
    Iterator previous(Iterator const& it)
    {
        return previous(it, hpx::traits::is_random_access_iterator<Iterator>());
    }

    template <typename Iterator>
    HPX_FORCEINLINE
    Iterator next(Iterator it, std::false_type)
    {
        return ++it;
    }

    template <typename Iterator>
    HPX_FORCEINLINE
    Iterator next(Iterator const& it, std::true_type)
    {
        return it + 1;
    }

    template <typename Iterator>
    HPX_FORCEINLINE
    Iterator next(Iterator const& it)
    {
        return next(it, hpx::traits::is_random_access_iterator<Iterator>());
    }
}}}

///////////////////////////////////////////////////////////////////////////////
// Version of stencil3_iterator which handles boundary elements internally
namespace hpx { namespace experimental
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator,
        typename IterBegin = Iterator, typename IterValueBegin = Iterator,
        typename IterEnd = IterBegin, typename IterValueEnd = IterValueBegin>
    class stencil3_iterator_full;

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorBase, typename IteratorValue>
        struct previous_transformer
        {
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
                return *detail::previous(it);
            }

        private:
            IteratorBase begin_;
            IteratorValue value_;
        };

        template <typename IteratorBase, typename IteratorValue>
        inline previous_transformer<IteratorBase, IteratorValue>
        make_previous_transformer(
            IteratorBase const& base, IteratorValue const& value)
        {
            return previous_transformer<IteratorBase, IteratorValue>(base, value);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorBase, typename IteratorValue>
        struct next_transformer
        {
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
                return *detail::next(it);
            }

        private:
            IteratorBase end_;
            IteratorValue value_;
        };

        template <typename IteratorBase, typename IteratorValue>
        inline next_transformer<IteratorBase, IteratorValue>
        make_next_transformer(
            IteratorBase const& base, IteratorValue const& value)
        {
            return next_transformer<IteratorBase, IteratorValue>(base, value);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator, typename IterBegin, typename IterValueBegin,
            typename IterEnd, typename IterValueEnd>
        struct stencil3_iterator_base
        {
            typedef previous_transformer<IterBegin, IterValueBegin> left_transformer;
            typedef next_transformer<IterEnd, IterValueEnd> right_transformer;

            typedef util::transform_iterator<Iterator, left_transformer> left_iterator;
            typedef util::transform_iterator<Iterator, right_transformer> right_iterator;

            typedef util::detail::zip_iterator_base<
                    util::tuple<left_iterator, Iterator, right_iterator>,
                    stencil3_iterator_full<
                        Iterator, IterBegin, IterValueBegin, IterEnd, IterValueEnd
                    >
                > type;

            static type create(Iterator const& it,
                IterBegin const& begin, IterValueBegin const& begin_val,
                IterEnd const& end, IterValueEnd const& end_val)
            {
                auto prev = make_previous_transformer(begin, begin_val);
                auto next = make_next_transformer(end, end_val);

                return type(util::make_tuple(
                    util::make_transform_iterator(it, prev), it,
                    util::make_transform_iterator(it, next)));
            }

            static type create(Iterator const& it)
            {
                return type(util::make_tuple(
                    util::make_transform_iterator(it, left_transformer()), it,
                    util::make_transform_iterator(it, right_transformer())));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename IterBegin, typename IterValueBegin,
        typename IterEnd, typename IterValueEnd>
    class stencil3_iterator_full
      : public detail::stencil3_iterator_base<
            Iterator, IterBegin, IterValueBegin, IterEnd, IterValueEnd
        >::type
    {
    private:
        typedef detail::stencil3_iterator_base<
                Iterator, IterBegin, IterValueBegin, IterEnd, IterValueEnd
            > base_maker;
        typedef typename base_maker::type base_type;

    public:
        stencil3_iterator_full() {}

        stencil3_iterator_full(Iterator const& it,
                IterBegin const& begin, IterValueBegin const& begin_val,
                IterEnd const& end, IterValueEnd const& end_val)
          : base_type(base_maker::create(it, begin, begin_val, end, end_val))
        {}

        explicit stencil3_iterator_full(Iterator const& it)
          : base_type(base_maker::create(it))
        {}

    private:
        friend class hpx::util::iterator_core_access;

        bool equal(stencil3_iterator_full const& other) const
        {
            return util::get<1>(this->get_iterator_tuple()) ==
                   util::get<1>(other.get_iterator_tuple());
        }
    };

    template <typename Iterator, typename IterBegin, typename IterValueBegin,
        typename IterEnd, typename IterValueEnd>
    inline stencil3_iterator_full<
        Iterator, IterBegin, IterValueBegin, IterEnd, IterValueEnd
    >
    make_stencil3_full_iterator(Iterator const& it,
        IterBegin const& begin, IterValueBegin const& begin_val,
        IterEnd const& end, IterValueEnd const& end_val)
    {
        typedef stencil3_iterator_full<
                Iterator, IterBegin, IterValueBegin, IterEnd, IterValueEnd
            > result_type;
        return result_type(it, begin, begin_val, end, end_val);
    }

    template <typename StencilIterator, typename Iterator>
    inline StencilIterator
    make_stencil3_full_iterator(Iterator const& it)
    {
        return StencilIterator(it);
    }

    template <typename Iterator, typename IterValue>
    inline std::pair<
        stencil3_iterator_full<Iterator, Iterator, IterValue, Iterator, IterValue>,
        stencil3_iterator_full<Iterator, Iterator, IterValue, Iterator, IterValue>
    >
    make_stencil3_full_range(Iterator const& begin, Iterator const& end,
        IterValue const& begin_val, IterValue const& end_val)
    {
        auto b = make_stencil3_full_iterator(begin, begin, begin_val,
            detail::previous(end), end_val);
        return std::make_pair(b, make_stencil3_full_iterator<decltype(b)>(end));
    }
}}

std::uint64_t bench_stencil3_iterator_full()
{
    std::vector<int> values(partition_size);
    std::iota(boost::begin(values), boost::end(values), 0);

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    auto r = hpx::experimental::make_stencil3_full_range(
        values.begin(), values.end(), &values.back(), &values.front());

    typedef std::iterator_traits<decltype(r.first)>::reference reference;

    int result = 0;

    std::for_each(r.first, r.second,
        [&result](reference val)
        {
            using hpx::util::get;
            result += get<0>(val) + get<1>(val) + get<2>(val);
        });

    return hpx::util::high_resolution_clock::now() - start;
}

///////////////////////////////////////////////////////////////////////////////
// compare with unchecked stencil3_iterator (version 1)
namespace hpx { namespace experimental
{
    template <typename Iterator>
    class stencil3_iterator_v1
      : public util::detail::zip_iterator_base<
                util::tuple<Iterator, Iterator, Iterator>,
                stencil3_iterator_v1<Iterator>
            >
    {
    private:
        typedef util::detail::zip_iterator_base<
                util::tuple<Iterator, Iterator, Iterator>,
                stencil3_iterator_v1<Iterator>
            > base_type;

    public:
        stencil3_iterator_v1() {}

        explicit stencil3_iterator_v1(Iterator const& it)
          : base_type(util::make_tuple(
                detail::previous(it), it, detail::next(it)))
        {}

    private:
        friend class hpx::util::iterator_core_access;

        bool equal(stencil3_iterator_v1 const& other) const
        {
            return util::get<1>(this->get_iterator_tuple()) ==
                   util::get<1>(other.get_iterator_tuple());
        }
    };

    template <typename Iterator>
    inline stencil3_iterator_v1<Iterator>
    make_stencil3_iterator_v1(Iterator const& it)
    {
        return stencil3_iterator_v1<Iterator>(it);
    }

    template <typename Iterator>
    inline std::pair<
        stencil3_iterator_v1<Iterator>,
        stencil3_iterator_v1<Iterator>
    >
    make_stencil3_range_v1(Iterator const& begin, Iterator const& end)
    {
        return std::make_pair(
            make_stencil3_iterator_v1(begin),
            make_stencil3_iterator_v1(end));
    }
}}

std::uint64_t bench_stencil3_iterator_v1()
{
    std::vector<int> values(partition_size);
    std::iota(boost::begin(values), boost::end(values), 0);

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    auto r = hpx::experimental::make_stencil3_range_v1(
        values.begin()+1, values.end()-1);

    typedef std::iterator_traits<decltype(r.first)>::reference reference;

    // handle boundary elements explicitly
    int result = values.back() + values.front() + values[1];

    std::for_each(r.first, r.second,
        [&result](reference val)
        {
            using hpx::util::get;
            result += get<0>(val) + get<1>(val) + get<2>(val);
        });

    result += values[partition_size-2] + values.back() + values.front();
    HPX_UNUSED(result);

    return hpx::util::high_resolution_clock::now() - start;
}

///////////////////////////////////////////////////////////////////////////////
// compare with unchecked stencil3_iterator (version 2)
namespace hpx { namespace experimental
{
    namespace detail
    {
        struct stencil_transformer_v2
        {
            template <typename Iterator>
            struct result
            {
                typedef typename std::iterator_traits<Iterator>::reference
                    element_type;
                typedef hpx::util::tuple<
                        element_type, element_type, element_type
                    > type;
            };

            // it will dereference tuple(it-1, it, it+1)
            template <typename Iterator>
            typename result<Iterator>::type
            operator()(Iterator const& it) const
            {
                typedef typename result<Iterator>::type type;
                return type(*detail::previous(it), *it, *detail::next(it));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator,
        typename Transformer = detail::stencil_transformer_v2>
    class stencil3_iterator_v2
      : public hpx::util::transform_iterator<Iterator, Transformer>
    {
    private:
        typedef hpx::util::transform_iterator<Iterator, Transformer> base_type;

    public:
        stencil3_iterator_v2() {}

        explicit stencil3_iterator_v2(Iterator const& it)
          : base_type(it, Transformer())
        {}

        stencil3_iterator_v2(Iterator const& it, Transformer const& t)
          : base_type(it, t)
        {}
    };

    template <typename Iterator, typename Transformer>
    inline stencil3_iterator_v2<Iterator, Transformer>
    make_stencil3_iterator_v2(Iterator const& it, Transformer const& t)
    {
        return stencil3_iterator_v2<Iterator, Transformer>(it, t);
    }

    template <typename Iterator, typename Transformer>
    inline std::pair<
        stencil3_iterator_v2<Iterator, Transformer>,
        stencil3_iterator_v2<Iterator, Transformer>
    >
    make_stencil3_range_v2(Iterator const& begin, Iterator const& end,
        Transformer const& t)
    {
        return std::make_pair(
            make_stencil3_iterator_v2(begin, t),
            make_stencil3_iterator_v2(end, t));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    inline stencil3_iterator_v2<Iterator>
    make_stencil3_iterator_v2(Iterator const& it)
    {
        return stencil3_iterator_v2<Iterator>(it);
    }

    template <typename Iterator>
    inline std::pair<
        stencil3_iterator_v2<Iterator>,
        stencil3_iterator_v2<Iterator>
    >
    make_stencil3_range_v2(Iterator const& begin, Iterator const& end)
    {
        return std::make_pair(
            make_stencil3_iterator_v2(begin),
            make_stencil3_iterator_v2(end));
    }
}}

std::uint64_t bench_stencil3_iterator_v2()
{
    std::vector<int> values(partition_size);
    std::iota(boost::begin(values), boost::end(values), 0);

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    auto r = hpx::experimental::make_stencil3_range_v2(
        values.begin()+1, values.end()-1);

    typedef std::iterator_traits<decltype(r.first)>::reference reference;

    // handle boundary elements explicitly
    int result = values.back() + values.front() + values[1];

    std::for_each(r.first, r.second,
        [&result](reference val)
        {
            using hpx::util::get;
            result += get<0>(val) + get<1>(val) + get<2>(val);
        });

    result += values[partition_size-2] + values.back() + values.front();
    HPX_UNUSED(result);

    return hpx::util::high_resolution_clock::now() - start;
}

///////////////////////////////////////////////////////////////////////////////
std::uint64_t bench_stencil3_iterator_explicit()
{
    std::vector<int> values(partition_size);
    std::iota(boost::begin(values), boost::end(values), 0);

    std::uint64_t start = hpx::util::high_resolution_clock::now();

    // handle all elements explicitly
    int result = values.back() + values.front() + values[1];

    auto range = boost::irange(0, partition_size);

    std::for_each(boost::begin(range), boost::end(range),
        [&result, &values](std::size_t i)
        {
            result += values[i-1] + values[i] + values[i+1];
        });

    result += values[partition_size-2] + values.back() + values.front();
    HPX_UNUSED(result);

    return hpx::util::high_resolution_clock::now() - start;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
//     bool csvoutput = vm.count("csv_output") != 0;

    test_count = vm["test_count"].as<int>();
    partition_size = vm["partition_size"].as<int>();

    // verify that input is within domain of program
    if (test_count <= 0) {
        hpx::cout << "test_count cannot be zero or negative..." << std::endl;
    }
    else if (partition_size < 3) {
        hpx::cout << "partition_size cannot be smaller than 3..." << std::endl;
    }
    else {
        // first run full stencil3 tests
        {
            std::uint64_t t = 0;
            for (int i = 0; i != test_count; ++i)
                t += bench_stencil3_iterator_full();
            hpx::cout << "full: " << (t * 1e-9) / test_count << std::endl;
        }

        // now run explicit (no-check) stencil3 tests
        {
            std::uint64_t t = 0;
            for (int i = 0; i != test_count; ++i)
                t += bench_stencil3_iterator_v1();
            hpx::cout << "nocheck(v1): " << (t * 1e-9) / test_count << std::endl;
        }

        {
            std::uint64_t t = 0;
            for (int i = 0; i != test_count; ++i)
                t += bench_stencil3_iterator_v2();
            hpx::cout << "nocheck(v2): " << (t * 1e-9) / test_count << std::endl;
        }

        // now run explicit tests
        {
            std::uint64_t t = 0;
            for (int i = 0; i != test_count; ++i)
                t += bench_stencil3_iterator_explicit();
            hpx::cout << "explicit: " << (t * 1e-9) / test_count << std::endl;
        }
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // initialize program
    boost::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ("test_count",
          boost::program_options::value<int>()->default_value(100),
          "number of tests to be averaged (default: 100)")

        ("partition_size",
          boost::program_options::value<int>()->default_value(10000),
          "number of elements to iterate over (default: 10000)")
        ;

    return hpx::init(cmdline, argc, argv);
}

