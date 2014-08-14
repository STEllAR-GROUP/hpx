//  Copyright (c) 2014 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_RECEIVE_BUFFER_MAY_08_2014_1102AM)
#define HPX_LCOS_LOCAL_RECEIVE_BUFFER_MAY_08_2014_1102AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/move.hpp>

#include <utility>
#include <map>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Mutex = lcos::local::spinlock>
    struct receive_buffer
    {
    protected:
        typedef Mutex mutex_type;
        typedef hpx::lcos::local::promise<T> buffer_promise_type;

        struct entry_data
        {
        private:
            HPX_MOVABLE_BUT_NOT_COPYABLE(entry_data)

        public:
            entry_data()
              : can_be_deleted_(false)
            {}

            entry_data(entry_data && rhs)
              : promise_(std::move(rhs.promise_)),
                can_be_deleted_(rhs.can_be_deleted_)
            {}

            hpx::future<T> get_future()
            {
                return promise_.get_future();
            }

            template <typename Val>
            void set_value(Val && val)
            {
                promise_.set_value(std::forward<Val>(val));
            }

            buffer_promise_type promise_;
            bool can_be_deleted_;
        };

        typedef std::map<std::size_t, entry_data> buffer_map_type;
        typedef typename buffer_map_type::iterator iterator;

    private:
        HPX_MOVABLE_BUT_NOT_COPYABLE(receive_buffer)

    public:
        receive_buffer() {}

        receive_buffer(receive_buffer && other)
          : mtx_(std::move(mtx_))
          , buffer_map_(std::move(buffer_map_))
        {}

        ~receive_buffer()
        {
            iterator end = buffer_map_.end();
            for (iterator it = buffer_map_.begin(); it != end; ++it)
            {
                if ((*it).second.promise_.valid())
                    (*it).second.get_future();
            }
        }

        receive_buffer& operator=(receive_buffer && other)
        {
            if(this != &other)
            {
                mtx_ = std::move(other.mtx_);
                buffer_map_ = std::move(other.buffer_map_);
            }
            return *this;
        }

        hpx::future<T> receive(std::size_t step)
        {
            typename mutex_type::scoped_lock l(mtx_);
            return receive_locked(step);
        }

        void store_received(std::size_t step, T && val)
        {
            typename mutex_type::scoped_lock l(mtx_);
            return store_received_locked(step, std::move(val));
        }

    protected:
        struct erase_on_exit
        {
            erase_on_exit(buffer_map_type& buffer_map, iterator it)
              : buffer_map_(buffer_map), it_(it)
            {}

            ~erase_on_exit()
            {
                buffer_map_.erase(it_);
            }

            buffer_map_type& buffer_map_;
            iterator it_;
        };

        hpx::future<T> receive_locked(std::size_t step)
        {
            iterator it = get_buffer_entry(step);
            HPX_ASSERT(it != buffer_map_.end());

            // if the value was already set we delete the entry after
            // retrieving the future
            if (it->second.can_be_deleted_)
            {
                erase_on_exit t(buffer_map_, it);
                return it->second.get_future();
            }

            // otherwise mark the entry as to be deleted once the value was set
            it->second.can_be_deleted_ = true;
            return it->second.get_future();
        }

        void store_received_locked(std::size_t step, T && val)
        {
            iterator it = get_buffer_entry(step);
            HPX_ASSERT(it != buffer_map_.end());

            // set value in promise
            it->second.set_value(std::move(val));

            if (!it->second.can_be_deleted_)
            {
                // if the future was not retrieved yet mark the entry as to be
                // deleted after it was be retrieved
                it->second.can_be_deleted_ = true;
                return;
            }

            // if the future was already retrieved we can delete the entry
            buffer_map_.erase(it);
        }

        iterator get_buffer_entry(std::size_t step)
        {
            iterator it = buffer_map_.find(step);
            if (it == buffer_map_.end())
            {
                std::pair<iterator, bool> res =
                    buffer_map_.insert(std::make_pair(step, entry_data()));
                if (!res.second)
                {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "base_receive_buffer::get_buffer_entry",
                        "couldn't insert a new entry into the receive buffer");
                }
                return res.first;
            }
            return it;
        }

    private:
        mutable mutex_type mtx_;
        buffer_map_type buffer_map_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex>
    struct receive_buffer<void, Mutex>
    {
    protected:
        typedef Mutex mutex_type;
        typedef hpx::lcos::local::promise<void> buffer_promise_type;

        struct entry_data
        {
        private:
            HPX_MOVABLE_BUT_NOT_COPYABLE(entry_data)

        public:
            entry_data()
              : can_be_deleted_(false)
            {}

            entry_data(entry_data && rhs)
              : promise_(std::move(rhs.promise_)),
                can_be_deleted_(rhs.can_be_deleted_)
            {}

            hpx::future<void> get_future()
            {
                return promise_.get_future();
            }

            void set_value()
            {
                promise_.set_value();
            }

            buffer_promise_type promise_;
            bool can_be_deleted_;
        };

        typedef std::map<std::size_t, entry_data> buffer_map_type;
        typedef typename buffer_map_type::iterator iterator;

    private:
        HPX_MOVABLE_BUT_NOT_COPYABLE(receive_buffer)

    public:
        receive_buffer() {}

        receive_buffer(receive_buffer && other)
          : mtx_(std::move(mtx_))
          , buffer_map_(std::move(buffer_map_))
        {}

        ~receive_buffer()
        {
            iterator end = buffer_map_.end();
            for (iterator it = buffer_map_.begin(); it != end; ++it)
            {
                if ((*it).second.promise_.valid())
                    (*it).second.get_future();
            }
        }

        receive_buffer& operator=(receive_buffer && other)
        {
            if(this != &other)
            {
                mtx_ = std::move(other.mtx_);
                buffer_map_ = std::move(other.buffer_map_);
            }
            return *this;
        }

        hpx::future<void> receive(std::size_t step)
        {
            typename mutex_type::scoped_lock l(mtx_);
            return receive_locked(step);
        }

        void store_received(std::size_t step)
        {
            typename mutex_type::scoped_lock l(mtx_);
            store_received_locked(step);
        }

    protected:
        struct erase_on_exit
        {
            erase_on_exit(buffer_map_type& buffer_map, iterator it)
              : buffer_map_(buffer_map), it_(it)
            {}

            ~erase_on_exit()
            {
                buffer_map_.erase(it_);
            }

            buffer_map_type& buffer_map_;
            iterator it_;
        };

        hpx::future<void> receive_locked(std::size_t step)
        {
            iterator it = get_buffer_entry(step);
            HPX_ASSERT(it != buffer_map_.end());

            // if the value was already set we delete the entry after
            // retrieving the future
            if (it->second.can_be_deleted_)
            {
                erase_on_exit t(buffer_map_, it);
                return it->second.get_future();
            }

            // otherwise mark the entry as to be deleted once the value was set
            it->second.can_be_deleted_ = true;
            return it->second.get_future();
        }

        void store_received_locked(std::size_t step)
        {
            iterator it = get_buffer_entry(step);
            HPX_ASSERT(it != buffer_map_.end());

            // set value in promise
            it->second.set_value();

            if (!it->second.can_be_deleted_)
            {
                // if the future was not retrieved yet mark the entry as to be
                // deleted after it was be retrieved
                it->second.can_be_deleted_ = true;
                return;
            }

            // if the future was already retrieved we can delete the entry
            buffer_map_.erase(it);
        }

        iterator get_buffer_entry(std::size_t step)
        {
            iterator it = buffer_map_.find(step);
            if (it == buffer_map_.end())
            {
                std::pair<iterator, bool> res =
                    buffer_map_.insert(std::make_pair(step, entry_data()));
                if (!res.second)
                {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "base_receive_buffer::get_buffer_entry",
                        "couldn't insert a new entry into the receive buffer");
                }
                return res.first;
            }
            return it;
        }

    private:
        mutable mutex_type mtx_;
        buffer_map_type buffer_map_;
    };
}}}

#endif

