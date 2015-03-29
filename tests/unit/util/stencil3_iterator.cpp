//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/util/transform_iterator.hpp>
#include <hpx/util/stencil3_iterator.hpp>

#include <strstream>

///////////////////////////////////////////////////////////////////////////////
void test_stencil3_iterator()
{
    std::vector<int> values(10);
    std::iota(std::begin(values), std::end(values), 0);

    auto r = hpx::util::make_stencil3_range(values.begin()+1, values.end()-1);

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
        template <typename T>
        struct result;

        template <typename This, typename Iterator>
        struct result<This(Iterator)>
        {
            typedef typename std::iterator_traits<Iterator>::reference
                element_type;
            typedef typename hpx::util::result_of<F(element_type)>::type
                value_type;

            typedef hpx::util::tuple<value_type, element_type, value_type> type;
        };

        template <typename F_>
        custom_stencil_transformer(F_ && f)
          : f_(std::forward<F_>(f))
        {}

        // it will dereference tuple(it-1, it, it+1)
        template <typename Iterator>
        typename result<custom_stencil_transformer(Iterator)>::type
        operator()(Iterator const& it) const
        {
            typedef typename result<custom_stencil_transformer(Iterator)>::type type;
            return type(f_(*hpx::util::detail::previous(it)), *it,
                f_(*hpx::util::detail::next(it)));
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
    std::iota(std::begin(values), std::end(values), 0);

    auto t = test::make_custom_stencil_transformer([](int i) { return 2*i; });
    auto r = hpx::util::make_stencil3_range(values.begin()+1, values.end()-1, t);

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
