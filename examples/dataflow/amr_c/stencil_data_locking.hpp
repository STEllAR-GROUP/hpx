//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_LOCKING_AUG_13_1139AM)
#define HPX_COMPONENTS_AMR_LOCKING_AUG_13_1139AM

#include <hpx/hpx.hpp>
#include <hpx/util/locking_helpers.hpp>

#include <boost/thread/locks.hpp>
#include <boost/foreach.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <vector>

#include "stencil_data.hpp"

namespace hpx { namespace components { namespace amr
{
#define HPX_LOCK_MUTEX(z, n, _) BOOST_PP_COMMA_IF(n) mutexes[n].get()
#define HPX_LOCK_MUTEXES(z, n, _)                                             \
    case n:                                                                   \
        boost::lock(BOOST_PP_REPEAT_ ## z(n, HPX_LOCK_MUTEX, _));             \
        break;                                                                \
    /**/

    namespace detail
    {
        template <typename Mutex>
        inline void
        lock(std::vector<boost::reference_wrapper<Mutex> >& mutexes)
        {
            switch(mutexes.size()) {
            case 1:
                mutexes[0].get().lock();
                break;

            // case 2: ...
            BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(HPX_LOCK_LIMIT), HPX_LOCK_MUTEXES, _)

            default:
                HPX_THROW_EXCEPTION(bad_parameter,
                    "hpx::components::amr::detail::lock",
                    "invalid number of arguments" + boost::lexical_cast<std::string>(mutexes.size()));
                break;
            }
        }

#undef HPX_LOCK_MUTEXES
#undef HPX_LOCK_MUTEX

        template <typename Mutex>
        inline void
        unlock(std::vector<boost::reference_wrapper<Mutex> > mutexes)
        {
            BOOST_FOREACH(Mutex& mutex, mutexes)
                mutex.unlock();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // helper allowing to do a scoping lock of several mutex's at once
    // unlock and re-lock on exit
    struct compare_mutexes
    {
        template <typename Mutex>
        bool operator()(
            boost::reference_wrapper<Mutex> const& lhs,
            boost::reference_wrapper<Mutex> const& rhs) const
        {
            return lhs.get_pointer() < rhs.get_pointer();
        }
    };

    template <typename Mutex>
    struct scoped_values_lock
    {
        scoped_values_lock(
            std::vector<access_memory_block<stencil_data> >& values)
        {
            mutexes_.reserve(values.size());
            BOOST_FOREACH(access_memory_block<stencil_data>& val, values)
                mutexes_.push_back(boost::ref(val->mtx_));

            std::sort(mutexes_.begin(), mutexes_.end(), compare_mutexes());

            detail::lock(mutexes_);
        }
        scoped_values_lock(
            access_memory_block<stencil_data>& value,
            std::vector<access_memory_block<stencil_data> >& values)
        {
            mutexes_.reserve(values.size()+1);
            BOOST_FOREACH(access_memory_block<stencil_data>& val, values)
                mutexes_.push_back(boost::ref(val->mtx_));
            mutexes_.push_back(boost::ref(value->mtx_));

            std::sort(mutexes_.begin(), mutexes_.end(), compare_mutexes());

            detail::lock(mutexes_);
        }
        scoped_values_lock(
            access_memory_block<stencil_data>& value1,
            access_memory_block<stencil_data>& value2)
        {
            mutexes_.reserve(2);
            mutexes_.push_back(boost::ref(value1->mtx_));
            mutexes_.push_back(boost::ref(value2->mtx_));

            std::sort(mutexes_.begin(), mutexes_.end(), compare_mutexes());

            detail::lock(mutexes_);
        }

        ~scoped_values_lock()
        {
            detail::unlock(mutexes_);
        }

        std::vector<boost::reference_wrapper<Mutex> > mutexes_;
    };

    template <typename Mutex>
    struct unlock_scoped_values_lock
    {
        unlock_scoped_values_lock(scoped_values_lock<Mutex>& l)
          : l_(l)
        {
            detail::unlock(l_.mutexes_);
        }

        ~unlock_scoped_values_lock()
        {
            detail::lock(l_.mutexes_);
        }

        scoped_values_lock<Mutex>& l_;
    };
}}}

#endif


