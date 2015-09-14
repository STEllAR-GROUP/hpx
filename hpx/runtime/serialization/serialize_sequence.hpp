//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZATION_SERIALIZE_SEQUENCE_MAY_17_2008_0545PM)
#define HPX_SERIALIZATION_SERIALIZE_SEQUENCE_MAY_17_2008_0545PM

#include <hpx/config.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/fusion/sequence/intrinsic/begin.hpp>
#include <boost/fusion/sequence/intrinsic/end.hpp>
#include <boost/fusion/iterator/deref.hpp>
#include <boost/fusion/iterator/next.hpp>
#include <boost/fusion/iterator/equal_to.hpp>
#include <boost/fusion/include/is_sequence.hpp>

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/runtime/serialization/map.hpp>
#include <hpx/runtime/serialization/array.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

namespace hpx { namespace serialization
{
    template <typename Archive, typename Sequence>
    void serialize_sequence(Archive& ar, Sequence& seq);

    namespace detail
    {
        /// serialization support for a boost::fusion::sequence
        struct serialize_sequence_loop
        {
            template <typename Archive, typename Element>
            static void serialize_element(Archive & ar, Element & e, boost::mpl::false_)
            {
                ar & e;
            }

            template <typename Archive, typename Element>
            static void serialize_element(Archive & ar, Element & e, boost::mpl::true_)
            {
                ar & hpx::serialization::make_array(&e, 1);
            }

            template <typename Archive, typename Element>
            static void serialize(Archive& ar, Element& e, boost::mpl::false_)
            {
                typedef typename boost::remove_const<Element>::type element_type;
                typedef typename
                    hpx::traits::is_bitwise_serializable<element_type>::type
                predicate;

                if(ar.disable_array_optimization())
                {
                    serialize_element(ar, e, boost::mpl::false_());
                }
                else
                {
                    serialize_element(ar, e, predicate());
                }
            }

            template <typename Archive>
            static void serialize(Archive&, util::unused_type&, boost::mpl::false_)
            {
            }

            template <typename Archive>
            static void serialize(Archive&, util::unused_type const&, boost::mpl::false_)
            {
            }

            template <typename Archive, typename Element>
            static void serialize(Archive& ar, Element& e, boost::mpl::true_)
            {
                serialize_sequence(ar, e);
            }

            template <typename Archive, typename Element>
            static void serialize(Archive& ar, Element& e)
            {
                typedef
                    typename boost::fusion::traits::is_sequence<Element>::type
                is_sequence;
                serialize(ar, e, is_sequence());
            }

            template <typename Archive, typename First, typename Last>
            static void
            call (Archive&, First const&, Last const&, boost::mpl::true_)
            {
            }

            template <typename Archive, typename First, typename Last>
            static void
            call(Archive& ar, First const& first, Last const& last, boost::mpl::false_)
            {
                boost::fusion::result_of::equal_to<
                    typename boost::fusion::result_of::next<First>::type, Last
                > is_last;

                serialize(ar, *first);
                call(ar, boost::fusion::next(first), last, is_last);
            }

            template <typename Archive, typename First, typename Last>
            static void
            call(Archive& ar, First const& first, Last const& last)
            {
                boost::fusion::result_of::equal_to<First, Last> is_last;
                call(ar, first, last, is_last);
            }
        };

        template <typename Archive, typename Sequence>
        inline void
        serialize_sequence(Archive& ar, Sequence& seq, boost::mpl::false_)
        {
            serialize_sequence_loop::call(ar, boost::fusion::begin(seq),
                boost::fusion::end(seq));
        }

        ///////////////////////////////////////////////////////////////////////
        // optimized serialization, all of the tuple is stored as a binary blob
        template <typename Archive, typename Sequence>
        inline void
        serialize_sequence(Archive& ar, Sequence& seq, boost::mpl::true_)
        {
            ar & hpx::serialization::make_array(&seq, 1);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename Sequence>
    inline void
    serialize_sequence(Archive& ar, Sequence& seq)
    {
        typedef typename boost::remove_const<Sequence>::type sequence_type;
        typedef typename
            hpx::traits::is_bitwise_serializable<sequence_type>::type
        predicate;

        if(boost::fusion::size(seq) != 0)
        {
#if defined(HPX_DEBUG_SERIALIZATION)
            char type = 'S';
            std::size_t size = boost::fusion::size(seq);
            ar & type & size;
            HPX_ASSERT(type == 'S');
            HPX_ASSERT(size == boost::fusion::size(seq));
#endif

            if(ar.disable_array_optimization())
            {
                detail::serialize_sequence(ar, seq, boost::mpl::false_());
            }
            else
            {
                detail::serialize_sequence(ar, seq, predicate());
            }

#if defined(HPX_DEBUG_SERIALIZATION)
            type = 'E';
            ar & type;
            HPX_ASSERT(type == 'E');
#endif
        }
    }
}}

#endif
