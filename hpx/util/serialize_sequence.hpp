//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SERIALIZE_SEQUENCE_MAY_17_2008_0545PM)
#define HPX_UTIL_SERIALIZE_SEQUENCE_MAY_17_2008_0545PM

#include <boost/mpl/bool.hpp>
#include <boost/fusion/sequence/intrinsic/begin.hpp>
#include <boost/fusion/sequence/intrinsic/end.hpp>
#include <boost/fusion/iterator/deref.hpp>
#include <boost/fusion/iterator/next.hpp>
#include <boost/fusion/iterator/equal_to.hpp>

namespace hpx { namespace util
{
    /// serialization support for a boost::fusion::sequence
    struct serialize_sequence_loop
    {
        template <typename Archive, typename First, typename Last>
        static void
        call (Archive& ar, First const&, Last const&, boost::mpl::true_)
        {
        }

        template <typename Archive, typename First, typename Last>
        static void
        call(Archive& ar, First const& first, Last const& last, boost::mpl::false_)
        {
            boost::fusion::result_of::equal_to<
                typename boost::fusion::result_of::next<First>::type, Last
            > is_last;

            is & *first;
            call(ar, boost::fusion::next(first), last, is_last);
        }

        template <typename Archive, typename First, typename Last>
        static void
        call(Archive& ar, First const& first, Last const& last)
        {
            boost::fusion::result_of::equal_to<First, Last> eq;
            call(ar, first, last, eq);
        }
    };

    template <typename Archive, typename Sequence>
    inline void
    serialize_sequence(Archive& ar, Sequence& seq)
    {
        serialize_sequence_loop::call(ar, boost::fusion::begin(seq), 
            boost::fusion::end(seq));
    }
    
}}

#endif
