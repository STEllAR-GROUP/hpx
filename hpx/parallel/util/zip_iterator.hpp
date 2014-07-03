//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_PARALLEL_UTIL_ZIP_ITERATOR_MAY_29_2014_0852PM)
#define HPX_PARALLEL_UTIL_ZIP_ITERATOR_MAY_29_2014_0852PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#include <iterator>

#define N HPX_TUPLE_LIMIT

namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct zip_iterator_value_impl
        {
            typedef typename std::iterator_traits<T>::value_type type;
        };

        template <>
        struct zip_iterator_value_impl<void>
        {
            typedef void type;
        };

        template <typename IteratorTuple>
        struct zip_iterator_value;

#       define HPX_PARALLEL_UTIL_ZIP_ITERATOR_VALUE(Z, N, D)                  \
            typename zip_iterator_value_impl<BOOST_PP_CAT(T, N)>::type        \
        /**/
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        struct zip_iterator_value<hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, T)> >
        {
            typedef hpx::util::tuple<
                BOOST_PP_ENUM(N, HPX_PARALLEL_UTIL_ZIP_ITERATOR_VALUE, _)
            > type;
        };
#       undef HPX_PARALLEL_UTIL_ZIP_ITERATOR_VALUE

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct zip_iterator_reference_impl
        {
            typedef typename std::iterator_traits<T>::reference type;
        };

        template <>
        struct zip_iterator_reference_impl<void>
        {
            typedef void type;
        };

        template <typename IteratorTuple>
        struct zip_iterator_reference;

#       define HPX_PARALLEL_UTIL_ZIP_ITERATOR_REFERENCE(Z, N, D)              \
            typename zip_iterator_reference_impl<BOOST_PP_CAT(T, N)>::type    \
        /**/
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        struct zip_iterator_reference<hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, T)> >
        {
            typedef hpx::util::tuple<
                BOOST_PP_ENUM(N, HPX_PARALLEL_UTIL_ZIP_ITERATOR_REFERENCE, _)
            > type;
        };
#       undef HPX_PARALLEL_UTIL_ZIP_ITERATOR_REFERENCE

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename U>
        struct zip_iterator_category_impl
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<
            std::random_access_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::random_access_iterator_tag iterator_category;
        };

        template <>
        struct zip_iterator_category_impl<
            std::random_access_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag iterator_category;
        };

        template <>
        struct zip_iterator_category_impl<
            std::bidirectional_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag iterator_category;
        };

        template <>
        struct zip_iterator_category_impl<
            std::random_access_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag iterator_category;
        };

        template <>
        struct zip_iterator_category_impl<
            std::forward_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::forward_iterator_tag iterator_category;
        };

        template <>
        struct zip_iterator_category_impl<
            std::bidirectional_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag iterator_category;
        };

        template <>
        struct zip_iterator_category_impl<
            std::bidirectional_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag iterator_category;
        };

        template <>
        struct zip_iterator_category_impl<
            std::forward_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::forward_iterator_tag iterator_category;
        };

        template <>
        struct zip_iterator_category_impl<
            std::forward_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag iterator_category;
        };

        template <typename IteratorTuple>
        struct zip_iterator_category;

        template <typename T>
        struct zip_iterator_category<hpx::util::tuple<T> >
        {
            typedef typename std::iterator_traits<T>::iterator_category type;
        };

#       define HPX_PARALLEL_UTIL_ZIP_ITERATOR_CATEGORY(Z, N, D)               \
            typename zip_iterator_reference_impl<BOOST_PP_CAT(T, N)>::type    \
        /**/
        template <typename T, typename U,
            BOOST_PP_ENUM_PARAMS(BOOST_PP_SUB(N, 2), typename T)>
        struct zip_iterator_category<
            hpx::util::tuple<T, U, BOOST_PP_ENUM_PARAMS(BOOST_PP_SUB(N, 2), T)>
        > : zip_iterator_category_impl<
                zip_iterator_category_impl<
                    typename std::iterator_traits<T>::iterator_category
                  , typename std::iterator_traits<U>::iterator_category
                >
              , zip_iterator_category<hpx::util::tuple<
                    BOOST_PP_ENUM_PARAMS(BOOST_PP_SUB(N, 2), T)
                > >
            >
        {};
#       undef HPX_PARALLEL_UTIL_ZIP_ITERATOR_CATEGORY

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct dereference_iterator;

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
                iter + n_;
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
                return dereference_iterator<IteratorTuple>::call(iterators_);
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

            std::size_t distance_to(zip_iterator_base const& other) const
            {
                return hpx::util::get<0>(other.iterators_)
                  - hpx::util::get<0>(iterators_);
            }

        private:
            IteratorTuple iterators_;
        };
    }

    template<
        BOOST_PP_ENUM_BINARY_PARAMS(N, typename T, = void BOOST_PP_INTERCEPT)
    >
    class zip_iterator;
}}}

#undef N

namespace hpx { namespace parallel { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename R, typename ZipIter>
    R get_iter(ZipIter&& zipiter)
    {
        return hpx::util::get<N>(zipiter.get_iterator_tuple());
    }

    template <int N, typename R, typename ZipIter>
    R get_iter(hpx::future<ZipIter>&& zipiter)
    {
        typedef typename hpx::util::tuple_element<
            N, typename ZipIter::iterator_tuple_type
        >::type result_type;

        return zipiter.then(
            [](hpx::future<ZipIter>&& f) -> result_type {
             return hpx::util::get<N>(f.get().get_iterator_tuple());
            });
    }
}}}

#   if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#       include <hpx/parallel/util/preprocessed/zip_iterator.hpp>
#   else
#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(preserve: 1, line: 0, output: "preprocessed/zip_iterator_" HPX_LIMIT_STR ".hpp")
#       endif

        ///////////////////////////////////////////////////////////////////////
#       define BOOST_PP_ITERATION_PARAMS_1                                    \
        (                                                                     \
            3                                                                 \
          , (                                                                 \
                1                                                             \
              , HPX_TUPLE_LIMIT                                               \
              , <hpx/parallel/util/zip_iterator.hpp>                          \
            )                                                                 \
        )                                                                     \
        /**/
#       include BOOST_PP_ITERATE()

#       if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#           pragma wave option(output: null)
#       endif
#   endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>
        struct dereference_iterator<hpx::util::tuple<
            BOOST_PP_ENUM_PARAMS(N, T)
        > >
        {
            typedef typename zip_iterator_reference<hpx::util::tuple<
                BOOST_PP_ENUM_PARAMS(N, T)
            > >::type result_type;

#           define HPX_PARALLEL_UTIL_ZIP_ITERATOR_GET(Z, N, D)                \
            *hpx::util::get<N>(iterators)                                     \
            /**/
            static result_type call(
                hpx::util::tuple<BOOST_PP_ENUM_PARAMS(N, T)> const& iterators)
            {
                return hpx::util::forward_as_tuple(
                    BOOST_PP_ENUM(N, HPX_PARALLEL_UTIL_ZIP_ITERATOR_GET, _));
            }
#           undef HPX_PARALLEL_UTIL_ZIP_ITERATOR_GET
        };
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
#   if N != HPX_TUPLE_LIMIT
    class zip_iterator<BOOST_PP_ENUM_PARAMS(N, T)>
#   else
    class zip_iterator
#   endif
      : public detail::zip_iterator_base<hpx::util::tuple<
            BOOST_PP_ENUM_PARAMS(N, T)
        > >
    {
        typedef detail::zip_iterator_base<hpx::util::tuple<
            BOOST_PP_ENUM_PARAMS(N, T)
        > > base_type;

    public:
        zip_iterator() : base_type() {}

#       define HPX_PARALLEL_UTIL_ZIP_ITERATOR_CONST_LVREF_PARAM(Z, N, D)      \
        BOOST_PP_CAT(T, N) const& BOOST_PP_CAT(v, N)                          \
        /**/
        explicit zip_iterator(
            BOOST_PP_ENUM(N, HPX_PARALLEL_UTIL_ZIP_ITERATOR_CONST_LVREF_PARAM, _)
        ) : base_type(hpx::util::tie(BOOST_PP_ENUM_PARAMS(N, v)))
        {}
#       undef HPX_PARALLEL_UTIL_ZIP_ITERATOR_CONST_LVREF_PARAM
    };

#   define HPX_PARALLEL_UTIL_ZIP_ITERATOR_DECAY_ELEM(Z, N, D)                 \
    typename hpx::util::decay<BOOST_PP_CAT(T, N)>::type                       \
    /**/
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    zip_iterator<BOOST_PP_ENUM(N, HPX_PARALLEL_UTIL_ZIP_ITERATOR_DECAY_ELEM, _)>
    make_zip_iterator(HPX_ENUM_FWD_ARGS(N, T, v))
    {
        typedef zip_iterator<
            BOOST_PP_ENUM(N, HPX_PARALLEL_UTIL_ZIP_ITERATOR_DECAY_ELEM, _)
        > result_type;

        return result_type(HPX_ENUM_FORWARD_ARGS(N, T, v));
    }
#   undef HPX_PARALLEL_UTIL_ZIP_ITERATOR_DECAY_ELEM
}}}

#undef N

#endif
