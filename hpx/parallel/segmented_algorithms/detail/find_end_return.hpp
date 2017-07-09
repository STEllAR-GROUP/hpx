//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_FIND_END_RETURN)
#define HPX_PARALLEL_SEGMENTED_FIND_END_RETURN

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_find_end_return
    namespace detail
    {
        template <typename FwdIter>
        struct find_end_return
        {
            FwdIter seq_first;
            unsigned int partial_position;

            template <typename Archive>
            void serialize(Archive& ar, unsigned)
            {
                ar & seq_first & partial_position;
            }
        };
    }
}}}
#endif
