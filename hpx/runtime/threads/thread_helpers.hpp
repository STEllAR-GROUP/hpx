//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREAD_HELPERS_NOV_15_2008_0504PM)
#define HPX_THREAD_HELPERS_NOV_15_2008_0504PM

#include <hpx/hpx_fwd.hpp>
#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads 
{
    ///////////////////////////////////////////////////////////////////////
    /// \brief  Set the thread state of the \a thread referenced by the 
    ///         thread_id \a id.
    ///
    /// \param self       [in] A reference to the \a thread executing 
    ///                   this function. 
    /// \param id         [in] The thread id of the thread the state should 
    ///                   be modified for.
    /// \param newstate   [in] The new state to be set for the thread 
    ///                   referenced by the \a id parameter.
    ///
    /// \returns          This function returns the previous state of the 
    ///                   thread referenced by the \a id parameter. It will 
    ///                   return one of the values as defined by the 
    ///                   \a thread_state enumeration. If the 
    ///                   thread is not known to the threadmanager the 
    ///                   return value will be \a thread_state#unknown.
    ///
    /// \note             This function yields the \a thread specified 
    ///                   by the parameter \a self if the thread referenced 
    ///                   by  the parameter \a id is in \a 
    ///                   thread_state#active state.
    HPX_API_EXPORT thread_state 
        set_thread_state(thread_self& self, thread_id_type id, 
            thread_state newstate);

    ///////////////////////////////////////////////////////////////////////
    /// \brief  Set the thread state of the \a thread referenced by the 
    ///         thread_id \a id.
    ///
    /// \param id         [in] The thread id of the thread the state should 
    ///                   be modified for.
    /// \param newstate   [in] The new state to be set for the thread 
    ///                   referenced by the \a id parameter.
    ///
    /// \returns          This function returns the previous state of the 
    ///                   thread referenced by the \a id parameter. It will 
    ///                   return one of the values as defined by the 
    ///                   \a thread_state enumeration. If the 
    ///                   thread is not known to the threadmanager the 
    ///                   return value will be \a thread_state#unknown.
    ///
    /// \note             If the thread referenced by the parameter \a id 
    ///                   is in \a thread_state#active state this function 
    ///                   does nothing except returning 
    ///                   thread_state#unknown. 
    HPX_API_EXPORT thread_state 
        set_thread_state(thread_id_type id, thread_state newstate);

    ///////////////////////////////////////////////////////////////////////
    /// \brief  Set the thread state of the \a thread referenced by the 
    ///         thread_id \a id.
    ///
    /// \param id         [in] The thread id of the thread the state should 
    ///                   be modified for.
    /// \param newstate   [in] The new state to be set for the thread 
    ///                   referenced by the \a id parameter.
    /// \param at_time
    ///
    /// \returns
    HPX_API_EXPORT thread_id_type 
        set_thread_state(thread_id_type id, thread_state newstate, 
            boost::posix_time::ptime const& at_time);

    ///////////////////////////////////////////////////////////////////////
    /// \brief  Set the thread state of the \a thread referenced by the 
    ///         thread_id \a id.
    ///
    /// \param id         [in] The thread id of the thread the state should 
    ///                   be modified for.
    /// \param newstate   [in] The new state to be set for the thread 
    ///                   referenced by the \a id parameter.
    /// \param after_duration
    ///                   
    ///
    /// \returns
    ///////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT thread_id_type 
        set_thread_state(thread_id_type id, thread_state newstate, 
            boost::posix_time::time_duration const& after_duration);

}}

#endif
