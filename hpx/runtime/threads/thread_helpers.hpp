//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREAD_HELPERS_NOV_15_2008_0504PM)
#define HPX_THREAD_HELPERS_NOV_15_2008_0504PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/backtrace.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>

#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/exception_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    struct thread_init_data;

    ///////////////////////////////////////////////////////////////////////////
    /// \brief  Set the thread state of the \a thread referenced by the
    ///         thread_id \a id.
    ///
    /// \param id         [in] The thread id of the thread the state should
    ///                   be modified for.
    /// \param state      [in] The new state to be set for the thread
    ///                   referenced by the \a id parameter.
    /// \param state_ex   [in] The new extended state to be set for the
    ///                   thread referenced by the \a id parameter.
    /// \param priority
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note             If the thread referenced by the parameter \a id
    ///                   is in \a thread_state#active state this function
    ///                   schedules a new thread which will set the state of
    ///                   the thread as soon as its not active anymore. The
    ///                   function returns \a thread_state#active in this case.
    ///
    /// \returns          This function returns the previous state of the
    ///                   thread referenced by the \a id parameter. It will
    ///                   return one of the values as defined by the
    ///                   \a thread_state enumeration. If the
    ///                   thread is not known to the thread-manager the
    ///                   return value will be \a thread_state#unknown.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT thread_state set_thread_state(thread_id_type const& id,
        thread_state_enum state = pending,
        thread_state_ex_enum stateex = wait_signaled,
        thread_priority priority = thread_priority_normal,
        hpx::error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////
    /// \brief  Set the thread state of the \a thread referenced by the
    ///         thread_id \a id.
    ///
    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (at the given time)
    ///
    /// \param id         [in] The thread id of the thread the state should
    ///                   be modified for.
    /// \param at_time
    /// \param state      [in] The new state to be set for the thread
    ///                   referenced by the \a id parameter.
    /// \param state_ex   [in] The new extended state to be set for the
    ///                   thread referenced by the \a id parameter.
    /// \param priority
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT thread_id_type set_thread_state(thread_id_type const& id,
        boost::posix_time::ptime const& at_time,
        thread_state_enum state = pending,
        thread_state_ex_enum stateex = wait_timeout,
        thread_priority priority = thread_priority_normal,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief  Set the thread state of the \a thread referenced by the
    ///         thread_id \a id.
    ///
    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (after the given duration)
    ///
    /// \param id         [in] The thread id of the thread the state should
    ///                   be modified for.
    /// \param after_duration
    /// \param state      [in] The new state to be set for the thread
    ///                   referenced by the \a id parameter.
    /// \param state_ex   [in] The new extended state to be set for the
    ///                   thread referenced by the \a id parameter.
    /// \param priority
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT thread_id_type set_thread_state(thread_id_type const& id,
        boost::posix_time::time_duration const& after_duration,
        thread_state_enum state = pending,
        thread_state_ex_enum stateex = wait_timeout,
        thread_priority priority = thread_priority_normal,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// The function get_thread_description is part of the thread related API
    /// allows to query the description of one of the threads known to the
    /// thread-manager.
    ///
    /// \param id         [in] The thread id of the thread being queried.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns the description of the
    ///                   thread referenced by the \a id parameter. If the
    ///                   thread is not known to the thread-manager the return
    ///                   value will be the string "<unknown>".
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT char const* get_thread_description(thread_id_type const& id,
        error_code& ec = throws);
    HPX_API_EXPORT char const* set_thread_description(thread_id_type const& id,
        char const* desc = 0, error_code& ec = throws);

    HPX_API_EXPORT char const* get_thread_lco_description(thread_id_type const& id,
        error_code& ec = throws);
    HPX_API_EXPORT char const* set_thread_lco_description(thread_id_type const& id,
        char const* desc = 0, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// The function get_thread_backtrace is part of the thread related API
    /// allows to query the currently stored thread back trace (which is
    /// captured during thread suspension).
    ///
    /// \param id         [in] The thread id of the thread being queried.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns the currently captured stack
    ///                   back trace of the thread referenced by the \a id
    ///                   parameter. If the thread is not known to the
    ///                   thread-manager the return value will be the zero.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT util::backtrace const* get_thread_backtrace(
        thread_id_type const& id, error_code& ec = throws);
    HPX_API_EXPORT util::backtrace const* set_thread_backtrace(
        thread_id_type const& id, util::backtrace const* bt = 0, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// The function get_thread_state is part of the thread related API. It
    /// queries the state of one of the threads known to the thread-manager.
    ///
    /// \param id         [in] The thread id of the thread the state should
    ///                   be modified for.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns the thread state of the
    ///                   thread referenced by the \a id parameter. If the
    ///                   thread is not known to the thread-manager the return
    ///                   value will be \a terminated.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT thread_state get_thread_state(thread_id_type const& id,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// The function get_thread_phase is part of the thread related API.
    /// It queries the phase of one of the threads known to the thread-manager.
    ///
    /// \param id         [in] The thread id of the thread the phase should
    ///                   be modified for.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns the thread phase of the
    ///                   thread referenced by the \a id parameter. If the
    ///                   thread is not known to the thread-manager the return
    ///                   value will be ~0.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT std::size_t get_thread_phase(thread_id_type const& id,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    // Return the number of the NUMA node the current thread is running on
    HPX_API_EXPORT std::size_t get_numa_node_number();

    ///////////////////////////////////////////////////////////////////////////
    /// Returns whether the given thread can be interrupted at this point.
    ///
    /// \param id         [in] The thread id of the thread which should be
    ///                   queried.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns \a true if the given thread
    ///                   can be interrupted at this point in time. It will
    ///                   return \a false otherwise.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT bool get_thread_interruption_enabled(thread_id_type const& id,
        error_code& ec = throws);

    /// Set whether the given thread can be interrupted at this point.
    ///
    /// \param id         [in] The thread id of the thread which should
    ///                   receive the new value.
    /// \param enable     [in] This value will determine the new interruption
    ///                   enabled status for the given thread.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns the previous value of
    ///                   whether the given thread could have been interrupted.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT bool set_thread_interruption_enabled(thread_id_type const& id,
        bool enable, error_code& ec = throws);

    /// Returns whether the given thread has been flagged for interruption.
    ///
    /// \param id         [in] The thread id of the thread which should be
    ///                   queried.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns \a true if the given thread
    ///                   was flagged for interruption. It will return
    ///                   \a false otherwise.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT bool get_thread_interruption_requested(thread_id_type const& id,
        error_code& ec = throws);

    /// Flag the given thread for interruption.
    ///
    /// \param id         [in] The thread id of the thread which should be
    ///                   interrupted.
    /// \param flag       [in] The flag encodes whether the thread should be
    ///                   interrupted (if it is \a true), or 'uninterrupted'
    ///                   (if it is \a false).
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT void interrupt_thread(thread_id_type const& id, bool flag,
        error_code& ec = throws);

    inline void interrupt_thread(thread_id_type const& id, error_code& ec = throws)
    {
        interrupt_thread(id, true, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Interrupt the current thread at this point if it was canceled. This
    /// will throw a thread_interrupted exception, which will cancel the thread.
    ///
    /// \param id         [in] The thread id of the thread which should be
    ///                   interrupted.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT void interruption_point(thread_id_type const& id,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// Return priority of the given thread
    ///
    /// \param id         [in] The thread id of the thread whose priority
    ///                   is queried.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT threads::thread_priority get_thread_priority(
        thread_id_type const& id, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT void run_thread_exit_callbacks(thread_id_type const& id,
        error_code& ec = throws);

    HPX_API_EXPORT bool add_thread_exit_callback(thread_id_type const& id,
        HPX_STD_FUNCTION<void()> const& f, error_code& ec = throws);

    HPX_API_EXPORT void free_thread_exit_callbacks(thread_id_type const& id,
        error_code& ec = throws);

#if HPX_THREAD_MAINTAIN_THREAD_DATA
    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT std::size_t get_thread_data(thread_id_type const& id,
        error_code& ec = throws);

    HPX_API_EXPORT std::size_t set_thread_data(thread_id_type const& id,
        std::size_t data, error_code& ec = throws);
#endif

    /// Returns a non-null pointer to the executor which was used to create
    /// the given thread.
    ///
    /// \throws If <code>&ec != &throws</code>, never throws, but will set \a ec
    ///         to an appropriate value when an error occurs. Otherwise, this
    ///         function will throw an \a hpx#exception with an error code of
    ///         \a hpx#yield_aborted if it is signaled with \a wait_aborted.
    ///         If called outside of a HPX-thread, this function will throw
    ///         an \a hpx#exception with an error code of \a hpx::null_thread_id.
    ///         If this function is called while the thread-manager is not
    ///         running, it will throw an \a hpx#exception with an error code of
    ///         \a hpx#invalid_status.
    ///
    HPX_API_EXPORT threads::executor get_executor(
        thread_id_type const& id, error_code& ec = throws);
}}

namespace hpx { namespace this_thread
{
    ///////////////////////////////////////////////////////////////////////////
    /// The function \a suspend will return control to the thread manager
    /// (suspends the current thread). It sets the new state of this thread
    /// to the thread state passed as the parameter.
    ///
    /// \note Must be called from within a HPX-thread.
    ///
    /// \throws If <code>&ec != &throws</code>, never throws, but will set \a ec
    ///         to an appropriate value when an error occurs. Otherwise, this
    ///         function will throw an \a hpx#exception with an error code of
    ///         \a hpx#yield_aborted if it is signaled with \a wait_aborted.
    ///         If called outside of a HPX-thread, this function will throw
    ///         an \a hpx#exception with an error code of \a hpx::null_thread_id.
    ///         If this function is called while the thread-manager is not
    ///         running, it will throw an \a hpx#exception with an error code of
    ///         \a hpx#invalid_status.
    ///
    HPX_API_EXPORT threads::thread_state_ex_enum suspend(
        threads::thread_state_enum state = threads::pending,
        char const* description = "this_thread::suspend",
        error_code& ec = throws);

    /// The function \a suspend will return control to the thread manager
    /// (suspends the current thread). It sets the new state of this thread
    /// to \a suspended and schedules a wakeup for this threads at the given
    /// time.
    ///
    /// \note Must be called from within a HPX-thread.
    ///
    /// \throws If <code>&ec != &throws</code>, never throws, but will set \a ec
    ///         to an appropriate value when an error occurs. Otherwise, this
    ///         function will throw an \a hpx#exception with an error code of
    ///         \a hpx#yield_aborted if it is signaled with \a wait_aborted.
    ///         If called outside of a HPX-thread, this function will throw
    ///         an \a hpx#exception with an error code of \a hpx::null_thread_id.
    ///         If this function is called while the thread-manager is not
    ///         running, it will throw an \a hpx#exception with an error code of
    ///         \a hpx#invalid_status.
    ///
    HPX_API_EXPORT threads::thread_state_ex_enum suspend(
        boost::posix_time::ptime const&,
        char const* description = "this_thread::suspend",
        error_code& ec = throws);

    /// The function \a suspend will return control to the thread manager
    /// (suspends the current thread). It sets the new state of this thread
    /// to \a suspended and schedules a wakeup for this threads after the given
    /// duration.
    ///
    /// \note Must be called from within a HPX-thread.
    ///
    /// \throws If <code>&ec != &throws</code>, never throws, but will set \a ec
    ///         to an appropriate value when an error occurs. Otherwise, this
    ///         function will throw an \a hpx#exception with an error code of
    ///         \a hpx#yield_aborted if it is signaled with \a wait_aborted.
    ///         If called outside of a HPX-thread, this function will throw
    ///         an \a hpx#exception with an error code of \a hpx::null_thread_id.
    ///         If this function is called while the thread-manager is not
    ///         running, it will throw an \a hpx#exception with an error code of
    ///         \a hpx#invalid_status.
    ///
    HPX_API_EXPORT threads::thread_state_ex_enum suspend(
        boost::posix_time::time_duration const&,
        char const* description = "this_thread::suspend",
        error_code& ec = throws);

    /// The function \a suspend will return control to the thread manager
    /// (suspends the current thread). It sets the new state of this thread
    /// to \a suspended and schedules a wakeup for this threads after the given
    /// time (specified in milliseconds).
    ///
    /// \note Must be called from within a HPX-thread.
    ///
    /// \throws If <code>&ec != &throws</code>, never throws, but will set \a ec
    ///         to an appropriate value when an error occurs. Otherwise, this
    ///         function will throw an \a hpx#exception with an error code of
    ///         \a hpx#yield_aborted if it is signaled with \a wait_aborted.
    ///         If called outside of a HPX-thread, this function will throw
    ///         an \a hpx#exception with an error code of \a hpx::null_thread_id.
    ///         If this function is called while the thread-manager is not
    ///         running, it will throw an \a hpx#exception with an error code of
    ///         \a hpx#invalid_status.
    ///
    inline threads::thread_state_ex_enum suspend(
        boost::uint64_t ms, char const* description = "this_thread::suspend",
        error_code& ec = throws)
    {
        return suspend(boost::posix_time::milliseconds(ms), description, ec);
    }
}}

///////////////////////////////////////////////////////////////////////////////
// FIXME: the API function below belong into the namespace hpx::threads
namespace hpx { namespace applier
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the work to
    ///        be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_state_ex_enum) and returns a
    ///                   \a threads#thread_state_enum.
    /// \param description [in] A optional string describing the newly created
    ///                   thread. This is useful for debugging and logging
    ///                   purposes as this string will be inserted in the logs.
    /// \param initial_state [in] The thread state the newly created thread
    ///                   should have. If this is not given it defaults to
    ///                   \a threads#pending, which means that the new thread
    ///                   will be scheduled to run as soon as it is created.
    /// \param run_now    [in] If this is set to `true` the thread object will
    ///                   be actually immediately created. Otherwise the
    ///                   thread-manager creates a work-item description, which
    ///                   will result in creating a thread object later (if
    ///                   no work is available any more). The default is to
    ///                   immediately create the thread object.
    /// \param priority   [in] This is the priority the newly created PX-thread
    ///                   should be executed with. The default is \a
    ///                   threads#thread_priority_normal. This parameter is not
    ///                   guaranteed to be taken into account as it depends on
    ///                   the used scheduling policy whether priorities are
    ///                   supported in the first place.
    /// \param os_thread  [in] The number of the shepherd thread the newly
    ///                   created PX-thread should run on. If this is given it
    ///                   will be no more than a hint in any case, mainly
    ///                   because even if the PX-thread gets scheduled on the
    ///                   queue of the requested shepherd thread, it still can
    ///                   be stolen by another shepherd thread. If this is not
    ///                   given, the system will select a shepherd thread.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns This function will return the internal id of the newly created
    ///          PX-thread or threads#invalid_thread_id (if run_now is set to
    ///          `false`).
    ///
    /// \note The value returned by the thread function will be interpreted by
    ///       the thread manager as the new thread state the executed PX-thread
    ///       needs to be switched to. Normally, PX-threads will either return
    ///       \a threads#terminated (if the thread should be destroyed) or
    ///       \a threads#suspended (if the thread needs to be suspended because
    ///       it is waiting for an external event to happen). The external
    ///       event will set the state of the thread back to pending, which
    ///       will re-schedule the PX-thread.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a hpx#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of hpx#exception.
    HPX_API_EXPORT threads::thread_id_type register_thread_plain(
        HPX_STD_FUNCTION<threads::thread_function_type> && func,
        char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1),
        threads::thread_stacksize stacksize = threads::thread_stacksize_default,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the work to
    ///        be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_state_ex_enum). The thread will be
    ///                   terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a threads#register_thread_plain
    ///
    HPX_API_EXPORT threads::thread_id_type register_thread(
        HPX_STD_FUNCTION<void(threads::thread_state_ex_enum)> && func,
        char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1),
        threads::thread_stacksize stacksize = threads::thread_stacksize_default,
        error_code& ec = throws);

    /// \brief Create a new \a thread using the given function as the work to
    ///        be executed. The work item can't be suspended when
    ///        executing.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_state_ex_enum). The thread will be
    ///                   terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a threads#register_thread_plain
    ///
    HPX_API_EXPORT threads::thread_id_type register_non_suspendable_thread(
        HPX_STD_FUNCTION<void(threads::thread_state_ex_enum)> && func,
        char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1),
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the work to
    ///        be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes no arguments. The
    ///                   thread will be terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a threads#register_thread_plain
    ///
    HPX_API_EXPORT threads::thread_id_type register_thread_nullary(
        HPX_STD_FUNCTION<void()> && func, char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1),
        threads::thread_stacksize stacksize = threads::thread_stacksize_default,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given data.
    ///
    /// \note This function is completely equivalent to the first overload
    ///       of threads#register_thread_plain above, except that part of the
    ///       parameters are passed as members of the threads#thread_init_data
    ///       object.
    ///
    HPX_API_EXPORT threads::thread_id_type register_thread_plain(
        threads::thread_init_data& data,
        threads::thread_state_enum initial_state = threads::pending,
        bool run_now = true, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed. This work item will be used to create a
    ///        \a threads#thread instance whenever the shepherd thread runs out
    ///        of work only. The created work descriptions will be queued
    ///        separately, causing them to be converted into actual thread
    ///        objects on a first-come-first-served basis.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_state_ex_enum) and returns a
    ///                   \a threads#thread_state_enum.
    /// \param description [in] A optional string describing the newly created
    ///                   thread. This is useful for debugging and logging
    ///                   purposes as this string will be inserted in the logs.
    /// \param initial_state [in] The thread state the newly created thread
    ///                   should have. If this is not given it defaults to
    ///                   \a threads#pending, which means that the new thread
    ///                   will be scheduled to run as soon as it is created.
    /// \param priority   [in] This is the priority the newly created PX-thread
    ///                   should be executed with. The default is \a
    ///                   threads#thread_priority_normal. This parameter is not
    ///                   guaranteed to be taken into account as it depends on
    ///                   the used scheduling policy whether priorities are
    ///                   supported in the first place.
    /// \param os_thread  [in] The number of the shepherd thread the newly
    ///                   created PX-thread should run on. If this is given it
    ///                   will be no more than a hint in any case, mainly
    ///                   because even if the PX-thread gets scheduled on the
    ///                   queue of the requested shepherd thread, it still can
    ///                   be stolen by another shepherd thread. If this is not
    ///                   given, the system will select a shepherd thread.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note The value returned by the thread function will be interpreted by
    ///       the thread manager as the new thread state the executed PX-thread
    ///       needs to be switched to. Normally, PX-threads will either return
    ///       \a threads#terminated (if the thread should be destroyed) or
    ///       \a threads#suspended (if the thread needs to be suspended because
    ///       it is waiting for an external event to happen). The external
    ///       event will set the state of the thread back to pending, which
    ///       will re-schedule the PX-thread.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    HPX_API_EXPORT void register_work_plain(
        HPX_STD_FUNCTION<threads::thread_function_type> && func,
        char const* description = 0, naming::address::address_type lva = 0,
        threads::thread_state_enum initial_state = threads::pending,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1),
        threads::thread_stacksize stacksize = threads::thread_stacksize_default,
        error_code& ec = throws);

#if !defined(DOXYGEN)
    HPX_API_EXPORT void register_work_plain(
        HPX_STD_FUNCTION<threads::thread_function_type> && func,
        naming::id_type const& target, char const* description = 0,
        naming::address::address_type lva = 0,
        threads::thread_state_enum initial_state = threads::pending,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1),
        threads::thread_stacksize stacksize = threads::thread_stacksize_default,
        error_code& ec = throws);
#endif

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_state_ex_enum). The thread will be
    ///                   terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a threads#register_work_plain
    ///
    HPX_API_EXPORT void register_work(
        HPX_STD_FUNCTION<void(threads::thread_state_ex_enum)> && func,
        char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1),
        threads::thread_stacksize stacksize = threads::thread_stacksize_default,
        error_code& ec = throws);

    /// \brief Create a new work item using the given function as the
    ///        work to be executed. The work item can't be suspended when
    ///        executing.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_state_ex_enum). The thread will be
    ///                   terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a threads#register_work_plain
    ///
    HPX_API_EXPORT void register_non_suspendable_work(
        HPX_STD_FUNCTION<void(threads::thread_state_ex_enum)> && func,
        char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1),
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes no arguments. The
    ///                   thread will be terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a threads#register_work_plain
    ///
    HPX_API_EXPORT void register_work_nullary(
        HPX_STD_FUNCTION<void()> && func, char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1),
        threads::thread_stacksize stacksize = threads::thread_stacksize_default,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed.
    ///
    /// \note This function is completely equivalent to the first overload
    ///       of threads#register_work_plain above, except that part of the
    ///       parameters are passed as members of the threads#thread_init_data
    ///       object.
    ///
    HPX_API_EXPORT void register_work_plain(
        threads::thread_init_data& data,
        threads::thread_state_enum initial_state = threads::pending,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// The \a create function initiates the creation of a new
    /// component instance using the runtime_support as given by targetgid.
    /// This function is non-blocking as it returns a \a lcos#future. The
    /// caller of this create is responsible to call
    /// \a lcos#future#get to obtain the result.
    ///
    /// \param targetgid
    /// \param type
    /// \param count
    ///
    /// \returns    The function returns a \a lcos#future instance
    ///             returning the the global id of the newly created
    ///             component when used to call get.
    ///
    /// \note       For synchronous operation use the function
    ///             \a threads#create_sync.
    HPX_API_EXPORT lcos::unique_future<naming::id_type>
        create(naming::id_type const& targetgid,
            components::component_type type, std::size_t count = 1);

    ///////////////////////////////////////////////////////////////////////////
    /// The \a create_sync function creates a new component instance using the
    /// \a runtime_support as given by targetgid. This function is blocking
    /// for the component to be created and until the global id of the new
    /// component has been returned.
    ///
    /// \param targetgid
    /// \param type
    /// \param count
    ///
    /// \returns    The function returns the global id of the newly created
    ///             component.
    ///
    /// \note       For asynchronous operation use the function
    ///             \a threads#create.
    HPX_API_EXPORT naming::id_type create_sync(naming::id_type const& targetgid,
        components::component_type type, std::size_t count = 1);
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    // Import all thread creation functions into this name space (we will
    // deprecate the functions in namespace applier above at some point).
    using applier::register_thread_plain;
    using applier::register_thread;
    using applier::register_thread_nullary;

    using applier::register_work_plain;
    using applier::register_work;
    using applier::register_work_nullary;
}}

#endif
