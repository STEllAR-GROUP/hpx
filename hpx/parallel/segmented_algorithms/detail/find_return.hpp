//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_FIND_END_RETURN)
#define HPX_PARALLEL_SEGMENTED_FIND_END_RETURN

#include <cstddef>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_find_return
    namespace detail
    {
        template <typename FwdIter>
        struct find_return
        {
            //Position of complete sequence in the vector
            FwdIter complete_sequence_position;
            //Position in the sequence till which complete match is found
            std::size_t complete_sequence_cursor;
            //Position of partial sequence in the give vector
            FwdIter partial_sequence_position;
            // Position in the sequence till which partial match is found
            std::size_t partial_sequence_cursor;

            template <typename Archive>
            void serialize(Archive& ar, unsigned)
            {
                ar & complete_sequence_position & complete_sequence_cursor &
                    partial_sequence_position & partial_sequence_cursor;
            }
        };
    }
}}}
#endif
