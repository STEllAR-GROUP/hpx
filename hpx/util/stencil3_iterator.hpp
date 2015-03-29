//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_STENCIL_ITERATOR_MAR_23_2015_1123AM)
#define HPX_UTIL_STENCIL_ITERATOR_MAR_23_2015_1123AM

#include <hpx/config/forceinline.hpp>
#include <hpx/util/transform_iterator.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>

#include <iterator>

#include <boost/type_traits/is_same.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Iterator>
        struct is_random_access_iterator
            : boost::is_same<
                std::random_access_iterator_tag,
                typename std::iterator_traits<Iterator>::iterator_category
              >
        {};

        template <typename Iterator>
        BOOST_FORCEINLINE
        Iterator previous(Iterator it, boost::mpl::false_)
        {
            return --it;
        }

        template <typename Iterator>
        BOOST_FORCEINLINE
        Iterator previous(Iterator const& it, boost::mpl::true_)
        {
            return it - 1;
        }

        template <typename Iterator>
        BOOST_FORCEINLINE
        Iterator previous(Iterator const& it)
        {
            return previous(it, is_random_access_iterator<Iterator>());
        }

        template <typename Iterator>
        BOOST_FORCEINLINE
        Iterator next(Iterator it, boost::mpl::false_)
        {
            return ++it;
        }

        template <typename Iterator>
        BOOST_FORCEINLINE
        Iterator next(Iterator const& it, boost::mpl::true_)
        {
            return it + 1;
        }

        template <typename Iterator>
        BOOST_FORCEINLINE
        Iterator next(Iterator const& it)
        {
            return next(it, is_random_access_iterator<Iterator>());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct stencil_transformer
        {
            template <typename T>
            struct result;

            template <typename This, typename Iterator>
            struct result<This(Iterator)>
            {
                typedef typename std::iterator_traits<Iterator>::reference
                    element_type;
                typedef tuple<element_type, element_type, element_type> type;
            };

            // it will dereference tuple(it-1, it, it+1)
            template <typename Iterator>
            typename result<stencil_transformer(Iterator)>::type
            operator()(Iterator const& it) const
            {
                typedef typename result<stencil_transformer(Iterator)>::type type;
                return type(*detail::previous(it), *it, *detail::next(it));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator,
        typename Transformer = detail::stencil_transformer>
    class stencil3_iterator
      : public transform_iterator<Iterator, Transformer>
    {
    private:
        typedef transform_iterator<Iterator, Transformer> base_type;

    public:
        stencil3_iterator() {}

        explicit stencil3_iterator(Iterator const& it)
          : base_type(it, Transformer())
        {}

        stencil3_iterator(Iterator const& it, Transformer const& t)
          : base_type(it, t)
        {}
    };

    ///////////////////////////////////////////////////////////////////////////
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
}}

#endif

