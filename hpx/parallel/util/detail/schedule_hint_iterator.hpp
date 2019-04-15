//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_DETAIL_SCHEDULE_HINT_ITERATOR_HPP)
#define HPX_PARALLEL_UTIL_DETAIL_SCHEDULE_HINT_ITERATOR_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/iterator_facade.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    struct schedule_hint_iterator
      : public hpx::util::iterator_facade<schedule_hint_iterator,
            threads::thread_schedule_hint const,
            std::input_iterator_tag,
            threads::thread_schedule_hint>
    {
    private:
        typedef hpx::util::iterator_facade<schedule_hint_iterator,
            threads::thread_schedule_hint const,
            std::input_iterator_tag,
            threads::thread_schedule_hint>
            base_type;

    public:
        using schedule_hint_fn_type =
            hpx::util::function_nonser<threads::thread_schedule_hint(
                std::size_t)>;
        using reference = threads::thread_schedule_hint;

        HPX_HOST_DEVICE
        schedule_hint_iterator(std::size_t task_idx,
            schedule_hint_fn_type schedule_hint_fn = schedule_hint_fn_type())
          : task_idx_(task_idx)
          , schedule_hint_fn_(schedule_hint_fn)
        {
        }

    protected:
        friend class hpx::util::iterator_core_access;

        HPX_HOST_DEVICE bool equal(schedule_hint_iterator const& other) const
        {
            return task_idx_ == other.task_idx_;
        }

        HPX_HOST_DEVICE reference dereference() const
        {
            HPX_ASSERT_MSG(schedule_hint_fn_,
                "The begin iterator for schedule_hint_iterator requires a "
                "non-empty function");
            return schedule_hint_fn_(task_idx_);
        }

        HPX_HOST_DEVICE void increment()
        {
            ++task_idx_;
        }

    private:
        std::size_t task_idx_;
        schedule_hint_fn_type schedule_hint_fn_;
    };
}}}}

#endif
