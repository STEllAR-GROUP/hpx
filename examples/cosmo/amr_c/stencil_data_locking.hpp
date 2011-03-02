//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_LOCKING_AUG_13_1139AM)
#define HPX_COMPONENTS_AMR_LOCKING_AUG_13_1139AM

#include <hpx/hpx.hpp>

#include <boost/thread/locks.hpp>
#include <boost/foreach.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <vector>

#include "stencil_data.hpp"

namespace boost 
{
    // boost doesn't have an overload for more than 5 mutexes
    namespace detail
    {
        template <typename MutexType1, typename MutexType2, typename MutexType3,
            typename MutexType4, typename MutexType5, typename MutexType6>
        unsigned lock_helper(MutexType1& m1, MutexType2& m2, MutexType3& m3,
                             MutexType4& m4, MutexType5& m5, MutexType6& m6)
        {
            boost::unique_lock<MutexType1> l1(m1);
            if (unsigned const failed_lock = try_lock_internal(m2, m3, m4, m5, m6))
            {
                return failed_lock;
            }
            l1.release();
            return 0;
        }

        template <typename MutexType1, typename MutexType2, typename MutexType3,
            typename MutexType4, typename MutexType5, typename MutexType6>
        unsigned try_lock_internal(MutexType1& m1, MutexType2& m2, MutexType3& m3,
                                   MutexType4& m4, MutexType5& m5, MutexType6& m6)
        {
            boost::unique_lock<MutexType1> l1(m1, boost::try_to_lock);
            if(!l1)
            {
                return 1;
            }
            if(unsigned const failed_lock=try_lock_internal(m2, m3, m4, m5, m6))
            {
                return failed_lock+1;
            }
            l1.release();
            return 0;
        }

        template <typename MutexType1, typename MutexType2, typename MutexType3,
            typename MutexType4, typename MutexType5, typename MutexType6,
            typename MutexType7>
        unsigned lock_helper(MutexType1& m1, MutexType2& m2, MutexType3& m3,
                             MutexType4& m4, MutexType5& m5, MutexType6& m6,
                             MutexType7& m7)
        {
            boost::unique_lock<MutexType1> l1(m1);
            if (unsigned const failed_lock = try_lock_internal(m2, m3, m4, m5, m6, m7))
            {
                return failed_lock;
            }
            l1.release();
            return 0;
        }
    }

    template <typename MutexType1, typename MutexType2, typename MutexType3,
        typename MutexType4, typename MutexType5, typename MutexType6>
    void lock(MutexType1& m1, MutexType2& m2, MutexType3& m3,
              MutexType4& m4, MutexType5& m5, MutexType6& m6)
    {
        unsigned const lock_count = 6;
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
            case 0:
                lock_first=detail::lock_helper(m1, m2, m3, m4, m5, m6);
                if (!lock_first)
                    return;
                break;
            case 1:
                lock_first = detail::lock_helper(m2, m3, m4, m5, m6, m1);
                if (!lock_first)
                    return;
                lock_first = (lock_first+1)%lock_count;
                break;
            case 2:
                lock_first = detail::lock_helper(m3, m4, m5, m6, m1, m2);
                if (!lock_first)
                    return;
                lock_first = (lock_first+2) % lock_count;
                break;
            case 3:
                lock_first = detail::lock_helper(m4, m5, m6, m1, m2, m3);
                if (!lock_first)
                    return;
                lock_first = (lock_first+3) % lock_count;
                break;
            case 4:
                lock_first = detail::lock_helper(m5, m6, m1, m2, m3, m4);
                if (!lock_first)
                    return;
                lock_first = (lock_first+4) % lock_count;
                break;
            case 5:
                lock_first = detail::lock_helper(m6, m1, m2, m3, m4, m5);
                if (!lock_first)
                    return;
                lock_first = (lock_first+5) % lock_count;
                break;
            }
        }
    }

    template <typename MutexType1, typename MutexType2, typename MutexType3,
        typename MutexType4, typename MutexType5, typename MutexType6,
        typename MutexType7>
    void lock(MutexType1& m1, MutexType2& m2, MutexType3& m3,
              MutexType4& m4, MutexType5& m5, MutexType6& m6,
              MutexType7& m7)
    {
        unsigned const lock_count = 7;
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
            case 0:
                lock_first=detail::lock_helper(m1, m2, m3, m4, m5, m6, m7);
                if (!lock_first)
                    return;
                break;
            case 1:
                lock_first = detail::lock_helper(m2, m3, m4, m5, m6, m7, m1);
                if (!lock_first)
                    return;
                lock_first = (lock_first+1) % lock_count;
                break;
            case 2:
                lock_first = detail::lock_helper(m3, m4, m5, m6, m7, m1, m2);
                if (!lock_first)
                    return;
                lock_first = (lock_first+2) % lock_count;
                break;
            case 3:
                lock_first = detail::lock_helper(m4, m5, m6, m7, m1, m2, m3);
                if (!lock_first)
                    return;
                lock_first = (lock_first+3) % lock_count;
                break;
            case 4:
                lock_first = detail::lock_helper(m5, m6, m7, m1, m2, m3, m4);
                if (!lock_first)
                    return;
                lock_first = (lock_first+4) % lock_count;
                break;
            case 5:
                lock_first = detail::lock_helper(m6, m7, m1, m2, m3, m4, m5);
                if (!lock_first)
                    return;
                lock_first = (lock_first+5) % lock_count;
                break;
            case 6:
                lock_first = detail::lock_helper(m7, m1, m2, m3, m4, m5, m6);
                if (!lock_first)
                    return;
                lock_first = (lock_first+6) % lock_count;
                break;
            }
        }
    }
}

namespace hpx { namespace components { namespace amr 
{
    namespace detail
    {
        template <typename Mutex>
        inline void 
        lock(std::vector<boost::reference_wrapper<Mutex> > mutexes)
        {
            switch(mutexes.size()) {
            case 1:
                mutexes[0].get().lock();
                break;

            case 2:
                boost::lock(mutexes[0].get(), mutexes[1].get());
                break;

            case 3:
                boost::lock(
                    mutexes[0].get(), mutexes[1].get(), mutexes[2].get());
                break;

            case 4:
                boost::lock(
                    mutexes[0].get(), mutexes[1].get(), mutexes[2].get(), 
                    mutexes[3].get());
                break;

            case 5:
                boost::lock(
                    mutexes[0].get(), mutexes[1].get(), mutexes[2].get(), 
                    mutexes[3].get(), mutexes[4].get());
                break;

            case 6:
                boost::lock(
                    mutexes[0].get(), mutexes[1].get(), mutexes[2].get(), 
                    mutexes[3].get(), mutexes[4].get(), mutexes[5].get());
                break;

            case 7:
                boost::lock(
                    mutexes[0].get(), mutexes[1].get(), mutexes[2].get(), 
                    mutexes[3].get(), mutexes[4].get(), mutexes[5].get(), 
                    mutexes[6].get());
                break;

            default:
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "hpx::components::amr::detail::lock", 
                    "invalid number of arguments");
                break;
            }
        }

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


