//  Copyright (c) 2006, Giovanni P. Deretta
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_COROUTINE_TUPLE_TRAITS_HPP
#define HPX_RUNTIME_COROUTINE_TUPLE_TRAITS_HPP

#include <hpx/config.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/if.hpp>
#include <boost/tuple/tuple.hpp>

#include <type_traits>

namespace hpx { namespace coroutines
{
    namespace detail
    {
        /*
         * NOTE & FIXME: coroutine_traits relies on the fact that we can construct a
         * boost::tuple specifing more arguments of type null_type than
         * required without generating an error. This is an undocumented
         * 'feature' and doesn't actually  work for nullary tuples,
         * so we need to detect this case and handle it with tuple_workaround.
         * If ever boost::tuple is changed (for example by switching to fusion)
         * tuple_workaround should be used in all cases to handle extra
         * parameters.
         * The real solution would be to have an internal tuple
         * type that derives from boost::tuple and handles all cases we care about.
         * Or better, just use Boost.Fusion.
         * In general the code in this file needs to be put in a better shape,
         * eliminating all corner cases.
         */

         /*
          * A boost::tuple<> is not constructible from an arbitrary
          * number of null_types (while non nullary tuples are).
          * This class takes care of this asymmetry.
          */
        struct tuple_workaround : boost::tuple<>
        {
            tuple_workaround(const boost::tuples::null_type&) {}
            tuple_workaround(const tuple_workaround&) {}
            tuple_workaround() {}
        };
    } /* detail */

    // All tuple traits must be derived from this
    // class to be correctly recognized.
    struct tuple_traits_tag {};

    template <typename T>
    struct get_length
    {
        enum { length = T::length };
    };

    template <typename T>
    struct is_nullary
      : boost::mpl::bool_<boost::tuples::length<T>::value == 0>
    {};

    template <typename T>
    struct is_singular
      : boost::mpl::bool_<boost::tuples::length<T>::value == 1>
    {};

    // Given a tuple_traits, makes a tuple of it
    // Simply returns the internal tuple type, unless
    // the tuple is nullary, then apply the nullary tuple workaround
    template <typename T>
    struct make_as_tuple
      : boost::mpl::if_<is_nullary<T>, detail::tuple_workaround, T>
    {};

    // Used to implement the next meta-function,
    // Split in two parts to satisfy the compiler.
    template <typename T>
    struct step_2
      : boost::mpl::eval_if<
            is_singular<T>,
            boost::tuples::element<0, typename make_as_tuple<T>::type>,
            boost::mpl::identity<typename make_as_tuple<T>::type>
        >
    {};

    // Given a trait class return the internal tuple type modified
    // as a return value.
    // The algorithm is as follow:
    // - If the tuple is nullary returns 'void'.
    // - If it singular returns the first type
    // - Else return the tuple itself.
    template <typename T>
    struct make_result_type
      : boost::mpl::eval_if<
            is_nullary<T>,
            boost::mpl::identity<void>,
            step_2<T>
        >
    {};

    template <typename T0 = boost::tuples::null_type>
    struct tuple_traits
      : tuple_traits_tag
    {
    public:

        // This is the straightforward boost::tuple trait
        // derived from the argument list. It is not
        // directly used in all cases.
        typedef boost::tuple<T0> internal_tuple_type;

        // FIXME: Currently coroutine code does not use this typedef in all cases
        // and expect it to be equal to boost::tuples::null_type
        typedef boost::tuples::null_type null_type;
        enum { length = boost::tuples::length<internal_tuple_type>::value };

        // Return the element at the Indext'th position in the typelist.
        // If the index is not less than the tuple length, it returns
        // null_type.
        template <int Index>
        struct at
          : boost::mpl::eval_if_c<
                Index < boost::tuples::length<
                    typename tuple_traits::internal_tuple_type
                >::value,
                boost::tuples::element<
                    Index, typename tuple_traits::internal_tuple_type>,
                boost::mpl::identity<typename tuple_traits::null_type>
            >
        {};

        typedef typename make_as_tuple<internal_tuple_type>::type as_tuple;
        typedef typename make_result_type<internal_tuple_type>::type as_result;
    };

    template <typename T>
    struct is_tuple_traits
      : std::is_base_of<tuple_traits_tag, T>
    {};
}}

#endif /*HPX_RUNTIME_COROUTINE_TUPLE_TRAITS_HPP*/
