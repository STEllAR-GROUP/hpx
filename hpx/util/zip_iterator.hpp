//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ZIP_ITERATOR_MAY_29_2014_0852PM)
#define HPX_UTIL_ZIP_ITERATOR_MAY_29_2014_0852PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pack.hpp>

#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/utility/enable_if.hpp>

#include <iterator>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct zip_iterator_value;

        template <typename ...Ts>
        struct zip_iterator_value<tuple<Ts...> >
        {
            typedef tuple<typename std::iterator_traits<Ts>::value_type...> type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct zip_iterator_reference;

        template <typename ...Ts>
        struct zip_iterator_reference<tuple<Ts...> >
        {
            typedef tuple<typename std::iterator_traits<Ts>::reference...> type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename U>
        struct zip_iterator_category_impl
        {};

        // random_access_iterator_tag
        template <>
        struct zip_iterator_category_impl<
            std::random_access_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::random_access_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::random_access_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::bidirectional_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::random_access_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::forward_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::random_access_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::input_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // bidirectional_iterator_tag
        template <>
        struct zip_iterator_category_impl<
            std::bidirectional_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::bidirectional_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::forward_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::bidirectional_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::input_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // forward_iterator_tag
        template <>
        struct zip_iterator_category_impl<
            std::forward_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::input_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::forward_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // input_iterator_tag
        template <>
        struct zip_iterator_category_impl<
            std::input_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple, typename Enable = void>
        struct zip_iterator_category;

        template <typename T>
        struct zip_iterator_category<
            tuple<T>
          , typename boost::enable_if_c<
                tuple_size<tuple<T> >::value == 1
            >::type
        >
        {
            typedef typename std::iterator_traits<T>::iterator_category type;
        };

        template <typename T, typename U>
        struct zip_iterator_category<
            tuple<T, U>
          , typename boost::enable_if_c<
                tuple_size<tuple<T, U> >::value == 2
            >::type
        > : zip_iterator_category_impl<
                typename std::iterator_traits<T>::iterator_category
              , typename std::iterator_traits<U>::iterator_category
            >
        {};

        template <typename T, typename U, typename ...Tail>
        struct zip_iterator_category<
            tuple<T, U, Tail...>
          , typename boost::enable_if_c<
                (tuple_size<tuple<T, U, Tail...> >::value > 2)
            >::type
        > : zip_iterator_category_impl<
                typename zip_iterator_category_impl<
                    typename std::iterator_traits<T>::iterator_category
                  , typename std::iterator_traits<U>::iterator_category
                >::type
              , typename zip_iterator_category<tuple<Tail...> >::type
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct dereference_iterator;

        template <typename ...Ts>
        struct dereference_iterator<tuple<Ts...> >
        {
            typedef typename zip_iterator_reference<
                tuple<Ts...>
            >::type result_type;

            template <std::size_t ...Is>
            static result_type call(detail::pack_c<std::size_t, Is...>,
                tuple<Ts...> const& iterators)
            {
                return util::forward_as_tuple(*util::get<Is>(iterators)...);
            }
        };

        struct increment_iterator
        {
            typedef void result_type;

            template <typename T>
            void operator()(T& iter) const
            {
                ++iter;
            }
        };

        struct decrement_iterator
        {
            typedef void result_type;

            template <typename T>
            void operator()(T& iter) const
            {
                --iter;
            }
        };

        struct advance_iterator
        {
            explicit advance_iterator(std::ptrdiff_t n) : n_(n) {}

            typedef void result_type;

            template <typename T>
            void operator()(T& iter) const
            {
                iter += n_;
            }

            std::ptrdiff_t n_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        class zip_iterator_base
          : public boost::iterator_facade<
                zip_iterator_base<IteratorTuple>
              , typename zip_iterator_value<IteratorTuple>::type
              , typename zip_iterator_category<IteratorTuple>::type
              , typename zip_iterator_reference<IteratorTuple>::type
            >
        {
            typedef
                boost::iterator_facade<
                    zip_iterator_base<IteratorTuple>
                  , typename zip_iterator_value<IteratorTuple>::type
                  , typename zip_iterator_category<IteratorTuple>::type
                  , typename zip_iterator_reference<IteratorTuple>::type
                >
                base_type;

        public:
            zip_iterator_base() {}

            explicit zip_iterator_base(IteratorTuple iterators)
              : iterators_(iterators) {}

            typedef IteratorTuple iterator_tuple_type;

            iterator_tuple_type const& get_iterator_tuple() const
            {
                return iterators_;
            }

        private:
            friend class boost::iterator_core_access;

            bool equal(zip_iterator_base const& other) const
            {
                return iterators_ == other.iterators_;
            }

            typename base_type::reference dereference() const
            {
                return dereference_iterator<IteratorTuple>::call(
                    typename detail::make_index_pack<
                        util::tuple_size<IteratorTuple>::value
                    >::type(), iterators_);
            }

            void increment()
            {
                return boost::fusion::for_each(iterators_,
                    increment_iterator());
            }

            void decrement()
            {
                return boost::fusion::for_each(iterators_,
                    decrement_iterator());
            }

            void advance(std::ptrdiff_t n)
            {
                return boost::fusion::for_each(iterators_,
                    advance_iterator(n));
            }

            std::ptrdiff_t distance_to(zip_iterator_base const& other) const
            {
                return util::get<0>(other.iterators_) - util::get<0>(iterators_);
            }

        private:
            IteratorTuple iterators_;
        };
    }

    template<typename ...Ts>
    class zip_iterator
      : public detail::zip_iterator_base<tuple<Ts...> >
    {
        typedef detail::zip_iterator_base<tuple<Ts...> > base_type;

    public:
        zip_iterator() : base_type() {}

        explicit zip_iterator(Ts const&... vs)
          : base_type(util::tie(vs...))
        {}
    };

    template <typename ...Ts>
    zip_iterator<typename decay<Ts>::type...>
    make_zip_iterator(Ts&&... vs)
    {
        typedef zip_iterator<typename decay<Ts>::type...> result_type;

        return result_type(std::forward<Ts>(vs)...);
    }
}}

#endif
