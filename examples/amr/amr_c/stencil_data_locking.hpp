//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_LOCKING_AUG_13_1139AM)
#define HPX_COMPONENTS_AMR_LOCKING_AUG_13_1139AM

#include <hpx/hpx.hpp>

#include <boost/thread/locks.hpp>
#include <boost/foreach.hpp>

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
    }

    template <typename MutexType1, typename MutexType2, typename MutexType3,
        typename MutexType4, typename MutexType5, typename MutexType6>
    void lock(MutexType1& m1, MutexType2& m2, MutexType3& m3,
              MutexType4& m4, MutexType5& m5, MutexType5& m6)
    {
        unsigned const lock_count = 6;
        unsigned lock_first = 0;
        for(;;)
        {
            switch(lock_first)
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
}

namespace hpx { namespace components { namespace amr 
{
    namespace detail
    {
        inline void lock(access_memory_block<stencil_data>& result,
            std::vector<access_memory_block<stencil_data> >& values)
        {
            boost::lock(result->mtx_);
            switch(values.size()) {
            case 1:
                boost::lock(values[0]->mtx_);
                break;

            case 2:
                boost::lock(values[0]->mtx_, values[1]->mtx_);
                break;

            case 3:
                boost::lock(values[0]->mtx_, values[1]->mtx_, values[2]->mtx_);
                break;

            case 4:
                boost::lock(values[0]->mtx_, values[1]->mtx_, values[2]->mtx_,
                    values[3]->mtx_);
                break;

            case 5:
                boost::lock(values[0]->mtx_, values[1]->mtx_, values[2]->mtx_,
                    values[3]->mtx_, values[4]->mtx_);
                break;

            case 6:
                boost::lock(values[0]->mtx_, values[1]->mtx_, values[2]->mtx_,
                    values[3]->mtx_, values[4]->mtx_, values[5]->mtx_);
                break;

            default:
                HPX_THROW_EXCEPTION(bad_parameter, 
                    "hpx::components::amr::detail::lock", 
                    "invalid number of arguments");
                break;
            }
        }

        inline void unlock(access_memory_block<stencil_data>& result,
            std::vector<access_memory_block<stencil_data> >& values)
        {
            BOOST_FOREACH(access_memory_block<stencil_data>& val, values_)
                boost::unlock(val->mtx_);
            boost::unlock(result_->mtx_);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // helper allowing to do a scoping lock of several mutex's at once 
    struct scoped_values_lock
    {
        scoped_values_lock(access_memory_block<stencil_data>& result,
                std::vector<access_memory_block<stencil_data> >& values)
          : result_(result), values_(values)
        {
            detail::lock(result, values);
        }
        ~scoped_values_lock()
        {
            detail::unlock(result_, values_);
        }

        access_memory_block<stencil_data>& result_;
        std::vector<access_memory_block<stencil_data> >& values_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // unlock and re-lock on exit
    struct unlock_scoped_values_lock
    {
        unlock_scoped_values_lock(scoped_values_lock& l)
          : l_(l)
        {
            detail::unlock(l.result_, l.values_);
        }
        ~unlock_scoped_values_lock()
        {
            detail::lock(l_.result_, l_.values_);
        }

        scoped_values_lock& l_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct unlock_value_lock
    {
        unlock_value_lock(lcos::mutex& m) : m_(m) 
        {
            m.unlock();
        }
        ~unlock_value_lock()
        {
            m_.lock();
        }
        
        lcos::mutex& l_;
    };
}}}

#endif


