//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/util/transform_iterator.hpp>

#include <boost/range/functions.hpp>

#include <sstream>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace test
{
    template <typename Iterator>
    HPX_FORCEINLINE
    Iterator previous(Iterator it)
    {
        return --it;
    }

    template <typename Iterator>
    HPX_FORCEINLINE
    Iterator next(Iterator it)
    {
        return ++it;
    }

    namespace detail
    {
        struct stencil_transformer
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
                return type(*test::previous(it), *it, *test::next(it));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator,
        typename Transformer = detail::stencil_transformer>
    class stencil3_iterator
      : public hpx::util::transform_iterator<Iterator, Transformer>
    {
    private:
        typedef hpx::util::transform_iterator<Iterator, Transformer> base_type;

    public:
        stencil3_iterator() {}

        explicit stencil3_iterator(Iterator const& it)
          : base_type(it, Transformer())
        {}

        stencil3_iterator(Iterator const& it, Transformer const& t)
          : base_type(it, t)
        {}
    };

    template <typename Iterator, typename Transformer>
    inline stencil3_iterator<Iterator, Transformer>
    make_stencil3_iterator(Iterator const& it, Transformer const& t)
    {
        return stencil3_iterator<Iterator, Transformer>(it, t);
    }

    template <typename Iterator, typename Transformer>
    inline std::pair<
        stencil3_iterator<Iterator, Transformer>,
        stencil3_iterator<Iterator, Transformer>
    >
    make_stencil3_range(Iterator const& begin, Iterator const& end,
        Transformer const& t)
    {
        return std::make_pair(
            make_stencil3_iterator(begin, t),
            make_stencil3_iterator(end, t));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    inline stencil3_iterator<Iterator>
    make_stencil3_iterator(Iterator const& it)
    {
        return stencil3_iterator<Iterator>(it);
    }

    template <typename Iterator>
    inline std::pair<
        stencil3_iterator<Iterator>,
        stencil3_iterator<Iterator>
    >
    make_stencil3_range(Iterator const& begin, Iterator const& end)
    {
        return std::make_pair(
            make_stencil3_iterator(begin),
            make_stencil3_iterator(end));
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_stencil3_iterator()
{
    std::vector<int> values(10);
    std::iota(boost::begin(values), boost::end(values), 0);

    auto r = test::make_stencil3_range(values.begin()+1, values.end()-1);

    typedef std::iterator_traits<decltype(r.first)>::reference reference;

    std::ostringstream str;

    std::for_each(r.first, r.second,
        [&str](reference val)
        {
            using hpx::util::get;
            str << get<0>(val) << get<1>(val) << get<2>(val) << " ";
        });

    HPX_TEST_EQ(str.str(), std::string("012 123 234 345 456 567 678 789 "));
}

///////////////////////////////////////////////////////////////////////////////
namespace test
{
    template <typename F>
    struct custom_stencil_transformer
    {
        template <typename Iterator>
        struct result
        {
            typedef typename std::iterator_traits<Iterator>::reference
                element_type;
            typedef typename hpx::util::result_of<F(element_type)>::type
                value_type;

            typedef hpx::util::tuple<value_type, element_type, value_type> type;
        };

        custom_stencil_transformer(F f)
          : f_(std::move(f))
        {}

        // it will dereference tuple(it-1, it, it+1)
        template <typename Iterator>
        typename result<Iterator>::type
        operator()(Iterator const& it) const
        {
            typedef typename result<Iterator>::type type;
            return type(f_(*test::previous(it)), *it, f_(*test::next(it)));
        }

        F f_;
    };

    template <typename F>
    inline custom_stencil_transformer<typename hpx::util::decay<F>::type>
    make_custom_stencil_transformer(F && f)
    {
        typedef custom_stencil_transformer<
                typename hpx::util::decay<F>::type>
            transformer_type;
        return transformer_type(std::forward<F>(f));
    }
}

void test_stencil3_iterator_custom()
{
    std::vector<int> values(10);
    std::iota(boost::begin(values), boost::end(values), 0);

    auto t = test::make_custom_stencil_transformer([](int i) -> int { return 2*i; });
    auto r = test::make_stencil3_range(values.begin()+1, values.end()-1, t);

    typedef std::iterator_traits<decltype(r.first)>::reference reference;

    std::ostringstream str;

    std::for_each(r.first, r.second,
        [&str](reference val)
        {
            using hpx::util::get;
            str << get<0>(val) << get<1>(val) << get<2>(val) << " ";
        });

    HPX_TEST_EQ(str.str(), std::string("014 226 438 6410 8512 10614 12716 14818 "));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_stencil3_iterator();
    test_stencil3_iterator_custom();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
