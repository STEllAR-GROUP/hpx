//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_STENCIL_ITERATOR_MAR_23_2015_1123AM)
#define HPX_UTIL_STENCIL_ITERATOR_MAR_23_2015_1123AM

#include <hpx/util/zip_iterator.hpp>

#include <type_traits>
#include <iterator>

namespace hpx { namespace util
{
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
        Iterator previous(Iterator const& it, boost::mpl::false_)
        {
            Iterator prev = it;
            return --prev;
        }

        template <typename Iterator>
        Iterator previous(Iterator const& it, boost::mpl::true_)
        {
            return it - 1;
        }

        template <typename Iterator>
        Iterator previous(Iterator const& it)
        {
            return previous(it, is_random_access_iterator<Iterator>());
        }

        template <typename Iterator>
        Iterator next(Iterator const& it, boost::mpl::false_)
        {
            Iterator prev = it;
            return ++prev;
        }

        template <typename Iterator>
        Iterator next(Iterator const& it, boost::mpl::true_)
        {
            return it + 1;
        }

        template <typename Iterator>
        Iterator next(Iterator const& it)
        {
            return next(it, is_random_access_iterator<Iterator>());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    class stencil3_iterator
      : public detail::zip_iterator_base<
                tuple<Iterator, Iterator, Iterator>,
                stencil3_iterator<Iterator>
            >
    {
    private:
        typedef detail::zip_iterator_base<
                tuple<Iterator, Iterator, Iterator>,
                stencil3_iterator<Iterator>
            > base_type;

    public:
        stencil3_iterator() {}

        explicit stencil3_iterator(Iterator const& it)
          : base_type(util::make_tuple(
                detail::previous(it), it, detail::next(it)))
        {}

    private:
        friend class boost::iterator_core_access;

        bool equal(stencil3_iterator const& other) const
        {
            return util::get<1>(this->get_iterator_tuple()) ==
                   util::get<1>(other.get_iterator_tuple());
        }
    };

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

