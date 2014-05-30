//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_DETAIL_ZIP_ITERATOR_MAY_29_2014_0852PM)
#define HPX_STL_DETAIL_ZIP_ITERATOR_MAY_29_2014_0852PM

#include <hpx/hpx_fwd.hpp>

#include <iterator>

#include <boost/iterator/zip_iterator.hpp>

namespace hpx { namespace parallel { namespace detail
{
    // Helper that picks the lowest common iterator type from given iterator
    // tags
    template <typename IterTag0, typename IterTag1>
    struct zip_iterator_category_helper
    {
        typedef std::input_iterator_tag iterator_category;
    };

    template <>
    struct zip_iterator_category_helper<
        std::random_access_iterator_tag,
        std::random_access_iterator_tag>
    {
        typedef std::random_access_iterator_tag iterator_category;
    };

    template <>
    struct zip_iterator_category_helper<
        std::random_access_iterator_tag,
        std::bidirectional_iterator_tag>
    {
        typedef std::bidirectional_iterator_tag iterator_category;
    };

    template <>
    struct zip_iterator_category_helper<
        std::bidirectional_iterator_tag,
        std::random_access_iterator_tag>
    {
        typedef std::bidirectional_iterator_tag iterator_category;
    };

    template <>
    struct zip_iterator_category_helper<
        std::random_access_iterator_tag,
        std::forward_iterator_tag>
    {
        typedef std::forward_iterator_tag iterator_category;
    };

    template <>
    struct zip_iterator_category_helper<
        std::forward_iterator_tag,
        std::random_access_iterator_tag>
    {
        typedef std::forward_iterator_tag iterator_category;
    };

    template <>
    struct zip_iterator_category_helper<
        std::bidirectional_iterator_tag,
        std::bidirectional_iterator_tag>
    {
        typedef std::bidirectional_iterator_tag iterator_category;
    };

    template <>
    struct zip_iterator_category_helper<
        std::bidirectional_iterator_tag,
        std::forward_iterator_tag>
    {
        typedef std::forward_iterator_tag iterator_category;
    };

    template <>
    struct zip_iterator_category_helper<
        std::forward_iterator_tag,
        std::bidirectional_iterator_tag>
    {
        typedef std::forward_iterator_tag iterator_category;
    };

    template <>
    struct zip_iterator_category_helper<
        std::forward_iterator_tag,
        std::forward_iterator_tag>
    {
        typedef std::forward_iterator_tag iterator_category;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename IteratorTuple>
    struct zip_iterator;

    template<typename IteratorTuple>
    struct zip_iterator_base
    {
    private:
        // Difference type is the first iterator's difference type
        typedef typename std::iterator_traits<
            typename boost::tuples::element<0, IteratorTuple>::type
        >::difference_type difference_type;

        // Traversal category is the minimum traversal category in the
        // iterator tuple.
        typedef typename
            boost::detail::minimum_traversal_category_in_iterator_tuple<
                IteratorTuple
            >::type
        traversal_category;

     public:
        // The iterator facade type from which the zip iterator will
        // be derived.
        typedef boost::iterator_facade<
            zip_iterator<IteratorTuple>,
            IteratorTuple,
            traversal_category,
            IteratorTuple,
            difference_type
        > type;
    };

    // This zip_iterator is almost identical to the boost::zip_iterator, except
    // that dereferencing returns the iterator tuple
    template <typename IteratorTuple>
    struct zip_iterator : zip_iterator_base<IteratorTuple>::type
    {
        typedef typename zip_iterator_base<IteratorTuple>::type base_type;

        zip_iterator() {}

        zip_iterator(IteratorTuple iterator_tuple)
          : iterator_tuple_(iterator_tuple)
        {}

        // Get method for the iterator tuple.
        IteratorTuple const& get_iterator_tuple() const
        {
            return iterator_tuple_;
        }

  private:
        friend class boost::iterator_core_access;

        IteratorTuple dereference() const
        {
            return iterator_tuple_;
        }

        template <typename OtherIteratorTuple>
        bool equal(const zip_iterator<OtherIteratorTuple>& other) const
        {
            return boost::detail::tuple_impl_specific::tuple_equal(
                get_iterator_tuple(), other.get_iterator_tuple());
        }

        void advance(typename base_type::difference_type n)
        {
            boost::detail::tuple_impl_specific::tuple_for_each(
                iterator_tuple_,
                boost::detail::advance_iterator<
                    typename base_type::difference_type>(n));
        }

        void increment()
        {
            boost::detail::tuple_impl_specific::tuple_for_each(
                iterator_tuple_, boost::detail::increment_iterator());
        }

        void decrement()
        {
            boost::detail::tuple_impl_specific::tuple_for_each(
                iterator_tuple_, boost::detail::decrement_iterator());
        }

        template <typename OtherIteratorTuple>
        typename base_type::difference_type distance_to(
            zip_iterator<OtherIteratorTuple> const& other) const
        {
            return boost::tuples::get<0>(other.get_iterator_tuple()) -
                   boost::tuples::get<0>(this->get_iterator_tuple());
        }

        IteratorTuple iterator_tuple_;
    };

    template <typename IteratorTuple>
    zip_iterator<IteratorTuple>
    make_zip_iterator(IteratorTuple t)
    {
        return zip_iterator<IteratorTuple>(t);
    }
}}}

#endif
