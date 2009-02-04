//  (C) Copyright 2008 Anthony Williams 
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_N2561_FUTURE_HPP
#define HPX_N2561_FUTURE_HPP

#include <stdexcept>
#include <boost/thread/detail/move.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <algorithm>
#include <memory>

#include <hpx/lcos/mutex.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    class future_uninitialized:
        public std::logic_error
    {
    public:
        future_uninitialized():
            std::logic_error("Future Uninitialized")
        {}
    };
    class future_moved:
        public std::logic_error
    {
    public:
        future_moved():
            std::logic_error("Future moved")
        {}
    };
    class broken_promise:
        public std::logic_error
    {
    public:
        broken_promise():
            std::logic_error("Broken promise")
        {}
    };
    class future_already_retrieved:
        public std::logic_error
    {
    public:
        future_already_retrieved():
            std::logic_error("Future already retrieved")
        {}
    };
    class promise_already_satisfied:
        public std::logic_error
    {
    public:
        promise_already_satisfied():
            std::logic_error("Promise already satisfied")
        {}
    };

    class task_already_started:
        public std::logic_error
    {
    public:
        task_already_started():
            std::logic_error("Task already started")
        {}
    };

    class task_moved:
        public std::logic_error
    {
    public:
        task_moved():
            std::logic_error("Task moved")
        {}
    };

    namespace future_state
    {
        enum state { uninitialized, waiting, ready, moved };
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct future_object_base
        {
            boost::exception_ptr exception;
            bool done;
            hpx::lcos::mutex mutex;
            boost::condition_variable waiters;

            future_object_base():
                done(false)
            {}
            virtual ~future_object_base()
            {}

            void mark_finished_internal()
            {
                done=true;
                waiters.notify_all();
            }

            void wait_internal(boost::unique_lock<hpx::lcos::mutex>& lock)
            {
                while(!done)
                {
                    waiters.wait(lock);
                }
            }
            

            void wait()
            {
                boost::unique_lock<hpx::lcos::mutex> lock(mutex);
                wait_internal(lock);
            }

            bool timed_wait_until_internal(boost::unique_lock<hpx::lcos::mutex>& lock,boost::system_time const& target_time)
            {
                while(!done)
                {
                    bool const success=waiters.timed_wait(lock,target_time);
                    if(!success && !done)
                    {
                        return false;
                    }
                }
                return true;
            }
            
            bool timed_wait_until(boost::system_time const& target_time)
            {
                boost::unique_lock<hpx::lcos::mutex> lock(mutex);
                return timed_wait_until_internal(lock,target_time);
            }
            
            void mark_exceptional_finish_internal(boost::exception_ptr const& e)
            {
                exception=e;
                mark_finished_internal();
            }
            void mark_exceptional_finish()
            {
                boost::lock_guard<hpx::lcos::mutex> lock(mutex);
                mark_exceptional_finish_internal(boost::current_exception());
            }
        private:
            future_object_base(future_object_base const&);
            future_object_base& operator=(future_object_base const&);
        };

        template<typename T>
        struct future_traits
        {
            typedef boost::scoped_ptr<T> storage_type;
            typedef T const& source_reference_type;
            typedef boost::detail::thread_move_t<T> move_source_type;

            static void init(storage_type& storage,T const& t)
            {
                storage.reset(new T(t));
            }
            
            static void init(storage_type& storage,boost::detail::thread_move_t<T> t)
            {
                storage.reset(new T(t));
            }

            static T move(storage_type& storage)
            {
                T res(*storage);
                cleanup(storage);
                return res;
            }

            static void move(storage_type& storage,T& dest)
            {
                dest=*storage;
                cleanup(storage);
            }

            static void cleanup(storage_type& storage)
            {
                storage.reset();
            }
        };
        
        template<typename T>
        struct future_traits<T&>
        {
            typedef T* storage_type;
            typedef T& source_reference_type;
            struct move_source_type
            {};

            static void init(storage_type& storage,T& t)
            {
                storage=&t;
            }

            static T& move(storage_type& storage)
            {
                T& res=*storage;
                cleanup(storage);
                return res;
            }

            static void cleanup(storage_type& storage)
            {
                storage=0;
            }
        };

        template<>
        struct future_traits<void>
        {
            typedef bool storage_type;

            static void init(storage_type& storage)
            {
                storage=true;
            }

            static void move(storage_type& storage)
            {
                cleanup(storage);
            }

            static void cleanup(storage_type& storage)
            {
                storage=false;
            }

        };

        template<typename T>
        struct future_object:
            detail::future_object_base
        {
            typedef typename future_traits<T>::storage_type storage_type;
            typedef typename future_traits<T>::source_reference_type source_reference_type;
            typedef typename future_traits<T>::move_source_type move_source_type;
            typedef typename boost::add_reference<T>::type reference;
            
            storage_type result;

            future_object():
                result(0)
            {}

            void mark_finished_with_result_internal(source_reference_type result_)
            {
                future_traits<T>::init(result,result_);
                mark_finished_internal();
            }
            void mark_finished_with_result_internal(move_source_type result_)
            {
                future_traits<T>::init(result,result_);
                mark_finished_internal();
            }

            void mark_finished_with_result(source_reference_type result_)
            {
                boost::lock_guard<hpx::lcos::mutex> lock(mutex);
                mark_finished_with_result_internal(result_);
            }
            void mark_finished_with_result(move_source_type result_)
            {
                boost::lock_guard<hpx::lcos::mutex> lock(mutex);
                mark_finished_with_result_internal(result_);
            }

            T move()
            {
                boost::unique_lock<hpx::lcos::mutex> lock(mutex);
                wait_internal(lock);
                if(exception)
                {
                    boost::rethrow_exception(exception);
                }
                if(!result)
                {
                    boost::throw_exception(future_moved());
                }

                return future_traits<T>::move(result);
            }

            bool try_move(reference dest)
            {
                boost::lock_guard<hpx::lcos::mutex> lock(mutex);
                if(!done)
                {
                    return false;
                }
                if(exception)
                {
                    boost::rethrow_exception(exception);
                }
                if(!result)
                {
                    boost::throw_exception(future_moved());
                }

                future_traits<T>::move(result,dest);
                return true;
            }

            bool timed_move_until(reference dest,boost::system_time const& target_time)
            {
                boost::unique_lock<hpx::lcos::mutex> lock(mutex);
                if(!timed_wait_until_internal(lock,target_time))
                {
                    return false;
                }
                if(exception)
                {
                    boost::rethrow_exception(exception);
                }
                if(!result)
                {
                    boost::throw_exception(future_moved());
                }
                
                future_traits<T>::move(result,dest);
                return true;
            }
            
            reference get()
            {
                boost::unique_lock<hpx::lcos::mutex> lock(mutex);
                wait_internal(lock);
                if(exception)
                {
                    boost::rethrow_exception(exception);
                }
                if(!result)
                {
                    boost::throw_exception(future_moved());
                }
                
                return *result;
            }

            bool try_get(reference dest)
            {
                boost::lock_guard<hpx::lcos::mutex> lock(mutex);
                if(!done)
                {
                    return false;
                }
                if(exception)
                {
                    boost::rethrow_exception(exception);
                }
                if(!result)
                {
                    boost::throw_exception(future_moved());
                }
                
                dest=*result;
                return true;
            }

            bool timed_get_until(reference dest,boost::system_time const& target_time)
            {
                boost::unique_lock<hpx::lcos::mutex> lock(mutex);

                if(!timed_wait_until_internal(lock,target_time))
                {
                    return false;
                }
                
                if(exception)
                {
                    boost::rethrow_exception(exception);
                }
                if(!result)
                {
                    boost::throw_exception(future_moved());
                }
                
                dest=*result;
                return true;
            }

            future_state::state get_state()
            {
                boost::lock_guard<hpx::lcos::mutex> guard(mutex);
                if(!done)
                {
                    return future_state::waiting;
                }
                else
                {
                    return (!result && !exception)?future_state::moved:future_state::ready;
                }
            }

        private:
            future_object(future_object const&);
            future_object& operator=(future_object const&);
        };

        template<>
        struct future_object<void>:
            detail::future_object_base
        {
            bool result;

            future_object():
                result(false)
            {}

            void mark_finished_with_result_internal()
            {
                result=true;
                mark_finished_internal();
            }

            void mark_finished_with_result()
            {
                boost::lock_guard<hpx::lcos::mutex> lock(mutex);
                mark_finished_with_result_internal();
            }

            void move()
            {
                boost::unique_lock<hpx::lcos::mutex> lock(mutex);
                wait_internal(lock);
                if(exception)
                {
                    boost::rethrow_exception(exception);
                }
                if(!result)
                {
                    boost::throw_exception(future_moved());
                }
                result=false;
            }

            void get()
            {
                boost::unique_lock<hpx::lcos::mutex> lock(mutex);
                wait_internal(lock);
                if(exception)
                {
                    boost::rethrow_exception(exception);
                }
                if(!result)
                {
                    boost::throw_exception(future_moved());
                }
            }
            
            future_state::state get_state()
            {
                boost::lock_guard<hpx::lcos::mutex> guard(mutex);
                if(!done)
                {
                    return future_state::waiting;
                }
                else
                {
                    return (!result && !exception)?future_state::moved:future_state::ready;
                }
            }

        private:
            future_object(future_object const&);
            future_object& operator=(future_object const&);
        };
    }
    
    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class shared_future;

    template <typename R>
    class unique_future
    {
        unique_future(unique_future & rhs);// = delete;
        unique_future& operator=(unique_future& rhs);// = delete;

        typedef boost::shared_ptr<detail::future_object<R> > future_ptr;
        
        future_ptr future;

        friend class shared_future<R>;

        typedef typename boost::add_reference<R>::type reference;
    public:
        unique_future(future_ptr future_):
            future(future_)
        {}

        typedef future_state::state state;

        unique_future()
        {}
        
//         unique_future(unique_future &&);
        unique_future(boost::detail::thread_move_t<unique_future> other):
            future(other->future)
        {
            other->future.reset();
        }

        ~unique_future()
        {}

//         unique_future& operator=(unique_future &&);
        unique_future& operator=(boost::detail::thread_move_t<unique_future> other)
        {
            future=other->future;
            other->future.reset();
            return *this;
        }

        operator boost::detail::thread_move_t<unique_future>()
        {
            return boost::detail::thread_move_t<unique_future>(*this);
        }

        void swap(unique_future& other)
        {
            future.swap(other.future);
        }

        // retrieving the value
        R move()
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            return future->move();
        }
        
        bool try_move(reference dest)
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            return future->try_move(dest);
        }
        
        template<typename Duration>
        bool timed_move(reference dest, Duration const& rel_time)
        {
            return timed_move_until(dest,boost::get_system_time()+rel_time);
        }
        
        bool timed_move_until(reference dest, boost::system_time const& abs_time)
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            return future->timed_move_until(dest,abs_time);
        }

        reference get()
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            return future->get();
        }
        
        bool try_get(reference dest)
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            return future->try_get(dest);
        }
        
        template<typename Duration>
        bool timed_get(reference dest, Duration const& rel_time)
        {
            return timed_get_until(dest,boost::get_system_time()+rel_time);
        }
        
        bool timed_get_until(reference dest, boost::system_time const& abs_time)
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            return future->timed_get_until(dest,abs_time);
        }
        

        // functions to check state, and wait for ready
        state get_state() const
        {
            if(!future)
            {
                return future_state::uninitialized;
            }
            return future->get_state();
        }
        

        bool is_ready() const
        {
            return get_state()==future_state::ready;
        }
        
        bool has_exception() const
        {
            if(!future)
            {
                return false;
            }
            boost::lock_guard<hpx::lcos::mutex> guard(future->mutex);
            return future->done && future->exception;
        }
        
        bool has_value() const
        {
            if(!future)
            {
                return false;
            }
            boost::lock_guard<hpx::lcos::mutex> guard(future->mutex);
            return future->done && future->result;
        }
        
        bool was_moved() const
        {
            return get_state()==future_state::moved;
        }
        

        void wait() const
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }
            future->wait();
        }
        
        template<typename Duration>
        bool timed_wait(Duration const& rel_time) const
        {
            return timed_wait_until(boost::get_system_time()+rel_time);
        }
        
        bool timed_wait_until(boost::system_time const& abs_time) const
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }
            return future->timed_wait_until(abs_time);
        }
        
    };

    template <>
    class unique_future<void>
    {
        unique_future(unique_future & rhs);// = delete;
        unique_future& operator=(unique_future& rhs);// = delete;

        typedef boost::shared_ptr<detail::future_object<void> > future_ptr;
        
        future_ptr future;

        friend class shared_future<void>;
    public:
        unique_future(future_ptr future_):
            future(future_)
        {}

        typedef future_state::state state;

        unique_future()
        {}
        
//         unique_future(unique_future &&);
        unique_future(boost::detail::thread_move_t<unique_future> other):
            future(other->future)
        {
            other->future.reset();
        }

        ~unique_future()
        {}

//         unique_future& operator=(unique_future &&);
        unique_future& operator=(boost::detail::thread_move_t<unique_future> other)
        {
            future=other->future;
            other->future.reset();
            return *this;
        }

        operator boost::detail::thread_move_t<unique_future>()
        {
            return boost::detail::thread_move_t<unique_future>(*this);
        }

        void swap(unique_future& other)
        {
            future.swap(other.future);
        }

        // retrieving the value
        void move()
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            future->move();
        }
        

        void get()
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            future->get();
        }
        
        // functions to check state, and wait for ready
        state get_state() const
        {
            if(!future)
            {
                return future_state::uninitialized;
            }
            return future->get_state();
        }
        

        bool is_ready() const
        {
            return get_state()==future_state::ready;
        }
        
        bool has_exception() const
        {
            if(!future)
            {
                return false;
            }
            boost::lock_guard<hpx::lcos::mutex> guard(future->mutex);
            return future->done && future->exception;
        }
        
        bool has_value() const
        {
            if(!future)
            {
                return false;
            }
            boost::lock_guard<hpx::lcos::mutex> guard(future->mutex);
            return future->done && future->result;
        }
        
        bool was_moved() const
        {
            return get_state()==future_state::moved;
        }

        void wait() const
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }
            future->wait();
        }
        
        template<typename Duration>
        bool timed_wait(Duration const& rel_time) const
        {
            return timed_wait_until(boost::get_system_time()+rel_time);
        }
        
        bool timed_wait_until(boost::system_time const& abs_time) const
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }
            return future->timed_wait_until(abs_time);
        }
        
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class shared_future
    {
        typedef boost::shared_ptr<detail::future_object<R> > future_ptr;
        
        future_ptr future;

        shared_future(unique_future<R>& other);
        shared_future& operator=(unique_future<R>& other);

        typedef typename boost::add_reference<R>::type reference;
    public:
        shared_future(future_ptr future_):
            future(future_)
        {}

        shared_future(shared_future const& other):
            future(other.future)
        {}

        typedef future_state::state state;

        shared_future()
        {}
        
//         shared_future(shared_future &&);
        shared_future(boost::detail::thread_move_t<shared_future> other):
            future(other->future)
        {
            other->future.reset();
        }

//         shared_future(unique_future<R> &&);
//         shared_future(const unique_future<R> &) = delete;
        shared_future(boost::detail::thread_move_t<unique_future<R> > other):
            future(other->future)
        {
            other->future.reset();
        }

        ~shared_future()
        {}

        shared_future& operator=(shared_future const& other)
        {
            future=other.future;
            return *this;
        }
//         shared_future& operator=(shared_future &&);
        shared_future& operator=(boost::detail::thread_move_t<shared_future> other)
        {
            future.swap(other->future);
            other->future.reset();
            return *this;
        }
        shared_future& operator=(boost::detail::thread_move_t<unique_future<R> > other)
        {
            future.swap(other->future);
            other->future.reset();
            return *this;
        }

        operator boost::detail::thread_move_t<shared_future>()
        {
            return boost::detail::thread_move_t<shared_future>(*this);
        }

        void swap(shared_future& other)
        {
            future.swap(other.future);
        }

        // retrieving the value
        reference get()
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            return future->get();
        }
        
        bool try_get(reference dest)
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            return future->try_get(dest);
        }
        
        template<typename Duration>
        bool timed_get(reference dest, Duration const& rel_time)
        {
            return timed_get_until(dest,boost::get_system_time()+rel_time);
        }
        
        bool timed_get_until(reference dest, boost::system_time const& abs_time)
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            return future->timed_get_until(dest,abs_time);
        }
        

        // functions to check state, and wait for ready
        state get_state() const
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }
            return future->get_state();
        }
        

        bool is_ready() const
        {
            return get_state()==future_state::ready;
        }
        
        bool has_exception() const
        {
            if(!future)
            {
                return false;
            }
            boost::lock_guard<hpx::lcos::mutex> guard(future->mutex);
            return future->done && future->exception;
        }
        
        bool has_value() const
        {
            if(!future)
            {
                return false;
            }
            boost::lock_guard<hpx::lcos::mutex> guard(future->mutex);
            return future->done && future->result;
        }
        
        bool was_moved() const
        {
            return get_state()==future_state::moved;
        }
        

        void wait() const
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }
            future->wait();
        }
        
        template<typename Duration>
        bool timed_wait(Duration const& rel_time) const
        {
            return timed_wait_until(boost::get_system_time()+rel_time);
        }
        
        bool timed_wait_until(boost::system_time const& abs_time) const
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }
            return future->timed_wait_until(abs_time);
        }
        
    };

    template <>
    class shared_future<void>
    {
        typedef boost::shared_ptr<detail::future_object<void> > future_ptr;
        
        future_ptr future;

        shared_future(unique_future<void>& other);
        shared_future& operator=(unique_future<void>& other);
    public:
        shared_future(future_ptr future_):
            future(future_)
        {}

        shared_future(shared_future const& other):
            future(other.future)
        {}

        typedef future_state::state state;

        shared_future()
        {}
        
//         shared_future(shared_future &&);
        shared_future(boost::detail::thread_move_t<shared_future> other):
            future(other->future)
        {
            other->future.reset();
        }

//         shared_future(unique_future<void> &&);
//         shared_future(const unique_future<void> &) = delete;
        shared_future(boost::detail::thread_move_t<unique_future<void> > other):
            future(other->future)
        {
            other->future.reset();
        }

        ~shared_future()
        {}

        shared_future& operator=(shared_future const& other)
        {
            future=other.future;
            return *this;
        }
//         shared_future& operator=(shared_future &&);
        shared_future& operator=(boost::detail::thread_move_t<shared_future> other)
        {
            future.swap(other->future);
            other->future.reset();
            return *this;
        }
        shared_future& operator=(boost::detail::thread_move_t<unique_future<void> > other)
        {
            future.swap(other->future);
            other->future.reset();
            return *this;
        }

        operator boost::detail::thread_move_t<shared_future>()
        {
            return boost::detail::thread_move_t<shared_future>(*this);
        }

        void swap(shared_future& other)
        {
            future.swap(other.future);
        }

        // retrieving the value
        void get()
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }

            future->get();
        }

        // functions to check state, and wait for ready
        state get_state() const
        {
            if(!future)
            {
                return future_state::uninitialized;
            }
            return future->get_state();
        }
        

        bool is_ready() const
        {
            return get_state()==future_state::ready;
        }
        
        bool has_exception() const
        {
            if(!future)
            {
                return false;
            }
            boost::lock_guard<hpx::lcos::mutex> guard(future->mutex);
            return future->done && future->exception;
        }
        
        bool has_value() const
        {
            if(!future)
            {
                return false;
            }
            boost::lock_guard<hpx::lcos::mutex> guard(future->mutex);
            return future->done && future->result;
        }
        
        bool was_moved() const
        {
            return get_state()==future_state::moved;
        }
        

        void wait() const
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }
            future->wait();
        }
        
        template<typename Duration>
        bool timed_wait(Duration const& rel_time) const
        {
            return timed_wait_until(boost::get_system_time()+rel_time);
        }
        
        bool timed_wait_until(boost::system_time const& abs_time) const
        {
            if(!future)
            {
                boost::throw_exception(future_uninitialized());
            }
            return future->timed_wait_until(abs_time);
        }
        
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class promise
    {
        typedef boost::shared_ptr<detail::future_object<R> > future_ptr;
        
        future_ptr future;
        bool future_obtained;
        
        promise(promise & rhs);// = delete;
        promise & operator=(promise & rhs);// = delete;
    public:
//         template <class Allocator> explicit promise(Allocator a);

        promise():
            future(new detail::future_object<R>),future_obtained(false)
        {}
        
//         promise(promise && rhs);
        promise(boost::detail::thread_move_t<promise> rhs):
            future(rhs->future),future_obtained(rhs->future_obtained)
        {
            rhs->future.reset();
        }
        ~promise()
        {
            if(future)
            {
                boost::lock_guard<hpx::lcos::mutex> lock(future->mutex);

                if(!future->done)
                {
                    try
                    {
                        boost::throw_exception(broken_promise());
                    }
                    catch(...)
                    {
                        future->mark_exceptional_finish_internal(boost::current_exception());
                    }
                }
            }
        }

        // Assignment
        promise & operator=(boost::detail::thread_move_t<promise> rhs)
        {
            future=rhs->future;
            future_obtained=rhs->future_obtained;
            rhs->future.reset();
            return *this;
        }
        
        void swap(promise& other)
        {
            future.swap(other.future);
            std::swap(future_obtained,other.future_obtained);
        }

        // Result retrieval
        unique_future<R> get_future()
        {
            if(!future)
            {
                boost::throw_exception(future_moved());
            }
            if(future_obtained)
            {
                boost::throw_exception(future_already_retrieved());
            }
            future_obtained=true;
            return unique_future<R>(future);
        }

        void set_value(typename detail::future_traits<R>::source_reference_type r)
        {
            if(!future)
            {
                boost::throw_exception(future_moved());
            }
            boost::lock_guard<hpx::lcos::mutex> lock(future->mutex);
            if(future->done)
            {
                boost::throw_exception(promise_already_satisfied());
            }
            future->mark_finished_with_result_internal(r);
        }

//         void set_value(R && r);
        void set_value(typename detail::future_traits<R>::move_source_type r)
        {
            if(!future)
            {
                boost::throw_exception(future_moved());
            }
            boost::lock_guard<hpx::lcos::mutex> lock(future->mutex);
            if(future->done)
            {
                boost::throw_exception(promise_already_satisfied());
            }
            future->mark_finished_with_result_internal(r);
        }

        void set_exception(boost::exception_ptr p)
        {
            if(!future)
            {
                boost::throw_exception(future_moved());
            }
            boost::lock_guard<hpx::lcos::mutex> lock(future->mutex);
            if(future->done)
            {
                boost::throw_exception(promise_already_satisfied());
            }
            future->mark_exceptional_finish_internal(p);
        }
        
    };

    template <>
    class promise<void>
    {
        typedef boost::shared_ptr<detail::future_object<void> > future_ptr;
        
        future_ptr future;
        bool future_obtained;
        
        promise(promise & rhs);// = delete;
        promise & operator=(promise & rhs);// = delete;
    public:
//         template <class Allocator> explicit promise(Allocator a);

        promise():
            future(new detail::future_object<void>),future_obtained(false)
        {}
        
//         promise(promise && rhs);
        promise(boost::detail::thread_move_t<promise> rhs):
            future(rhs->future),future_obtained(rhs->future_obtained)
        {
            rhs->future.reset();
        }
        ~promise()
        {
            if(future)
            {
                boost::lock_guard<hpx::lcos::mutex> lock(future->mutex);

                if(!future->done)
                {
                    try
                    {
                        boost::throw_exception(broken_promise());
                    }
                    catch(...)
                    {
                        future->mark_exceptional_finish_internal(boost::current_exception());
                    }
                }
            }
        }

        // Assignment
        promise & operator=(boost::detail::thread_move_t<promise> rhs)
        {
            future=rhs->future;
            future_obtained=rhs->future_obtained;
            rhs->future.reset();
            return *this;
        }
        
        void swap(promise& other)
        {
            future.swap(other.future);
            std::swap(future_obtained,other.future_obtained);
        }

        // Result retrieval
        unique_future<void> get_future()
        {
            if(!future)
            {
                boost::throw_exception(future_moved());
            }
            if(future_obtained)
            {
                boost::throw_exception(future_already_retrieved());
            }
            future_obtained=true;
            return unique_future<void>(future);
        }

        void set_value()
        {
            if(!future)
            {
                boost::throw_exception(future_moved());
            }
            boost::lock_guard<hpx::lcos::mutex> lock(future->mutex);
            if(future->done)
            {
                boost::throw_exception(promise_already_satisfied());
            }
            future->mark_finished_with_result_internal();
        }

        void set_exception(boost::exception_ptr p)
        {
            if(!future)
            {
                boost::throw_exception(future_moved());
            }
            boost::lock_guard<hpx::lcos::mutex> lock(future->mutex);
            if(future->done)
            {
                boost::throw_exception(promise_already_satisfied());
            }
            future->mark_exceptional_finish_internal(p);
        }
        
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template<typename R>
        struct task_base:
            detail::future_object<R>
        {
            bool started;

            task_base():
                started(false)
            {}

            void run()
            {
                {
                    boost::lock_guard<hpx::lcos::mutex> lk(this->mutex);
                    if(started)
                    {
                        boost::throw_exception(task_already_started());
                    }
                    started=true;
                }
                do_run();
            }
            
            virtual void do_run()=0;
        };
        
        
        template<typename R,typename F>
        struct task_object:
            task_base<R>
        {
            F f;
            task_object(F const& f_):
                f(f_)
            {}
            task_object(boost::detail::thread_move_t<F> f_):
                f(f_)
            {}
            
            void do_run()
            {
                try
                {
                    this->mark_finished_with_result(f());
                }
                catch(...)
                {
                    this->mark_exceptional_finish();
                }
            }
        };

        template<typename F>
        struct task_object<void,F>:
            task_base<void>
        {
            F f;
            task_object(F const& f_):
                f(f_)
            {}
            task_object(boost::detail::thread_move_t<F> f_):
                f(f_)
            {}
            
            void do_run()
            {
                try
                {
                    f();
                    this->mark_finished_with_result();
                }
                catch(...)
                {
                    this->mark_exceptional_finish();
                }
            }
        };

    }
    

    template<typename R>
    class packaged_task
    {
        boost::shared_ptr<detail::task_base<R> > task;
        bool future_obtained;

        packaged_task(packaged_task&);// = delete;
        packaged_task& operator=(packaged_task&);// = delete;
        
    public:
        // construction and destruction
        template <class F>
        explicit packaged_task(F const& f):
            task(new detail::task_object<R,F>(f)),future_obtained(false)
        {}
        explicit packaged_task(R(*f)()):
            task(new detail::task_object<R,R(*)()>(f)),future_obtained(false)
        {}
        
        template <class F>
        explicit packaged_task(boost::detail::thread_move_t<F> f):
            task(new detail::task_object<R,F>(f)),future_obtained(false)
        {}

//         template <class F, class Allocator>
//         explicit packaged_task(F const& f, Allocator a);
//         template <class F, class Allocator>
//         explicit packaged_task(F&& f, Allocator a);

//         packaged_task(packaged_task&& other);
        packaged_task(boost::detail::thread_move_t<packaged_task> other):
            future_obtained(other->future_obtained)
        {
            task.swap(other->task);
        }

        ~packaged_task()
        {}

        // assignment
//         packaged_task& operator=(packaged_task&& other);

        packaged_task& operator=(boost::detail::thread_move_t<packaged_task> other)
        {
            packaged_task temp(other);
            swap(temp);
            return *this;
        }

        void swap(packaged_task& other)
        {
            task.swap(other.task);
            std::swap(future_obtained,other.future_obtained);
        }

        // result retrieval
        unique_future<R> get_future()
        {
            if(!task)
            {
                boost::throw_exception(task_moved());
            }
            else if(!future_obtained)
            {
                future_obtained=true;
                return unique_future<R>(task);
            }
            else
            {
                boost::throw_exception(future_already_retrieved());
            }
        }
        

        // execution
        void operator()()
        {
            if(!task)
            {
                boost::throw_exception(task_moved());
            }
            task->run();
        }
        
    };

///////////////////////////////////////////////////////////////////////////////
}}   // namespace hpx::util

#endif

