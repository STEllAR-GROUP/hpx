//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_IBVERBS_SENDER_HPP
#define HPX_PARCELSET_POLICIES_IBVERBS_SENDER_HPP

#include <hpx/config/defines.hpp>
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)

#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/plugins/parcelport/ibverbs/context.hpp>
#include <hpx/plugins/parcelport/ibverbs/messages.hpp>
#include <hpx/plugins/parcelport/ibverbs/allocator.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/asio/placeholders.hpp>
#include <boost/thread/locks.hpp>

#include <vector>

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    class connection_handler;
    void add_sender(connection_handler & handler,
        boost::shared_ptr<sender> const& sender_connection);

    ibverbs_mr register_buffer(connection_handler & handler,
        ibv_pd * pd, char * buffer, std::size_t size, int access);

    class sender
      : public parcelset::parcelport_connection<
            sender
          , std::vector<char, allocator<message::payload_size> >
        >//data_buffer>
    {
        typedef bool(sender::*next_function_type)();
    public:
        typedef
            util::function_nonser<void(boost::system::error_code const &, std::size_t)>
            handler_function_type;
        typedef
            util::function_nonser<
                void(
                    boost::system::error_code const &
                  , parcelset::locality const&
                  , boost::shared_ptr<sender>
                )
            >
            postprocess_function_type;

        sender(connection_handler & handler, util::memory_chunk_pool & pool,
            parcelset::locality const& there,
            performance_counters::parcels::gatherer& parcels_sent)
          : context_(), parcelport_(handler), there_(there),
            parcels_sent_(parcels_sent), memory_pool_(pool)
        {
        }

        ~sender()
        {
            // gracefully and portably shutdown the socket
            boost::system::error_code ec;
            context_.shutdown(ec); // shut down data connection
            context_.close(ec);    // close the socket to give it back to the OS
        }

        /// Get the window associated with the parcelport_connection.
        client_context& context() { return context_; }

        void verify(parcelset::locality const & parcel_locality_id)
        {
            HPX_ASSERT(parcel_locality_id == there_);
        }

        parcelset::locality const& destination() const
        {
            return there_;
        }

        boost::shared_ptr<parcel_buffer_type> get_buffer(parcel const & p,
            std::size_t arg_size)
        {
            if(!buffer_)
            {
                boost::system::error_code ec;
                buffer_
                    = boost::shared_ptr<parcel_buffer_type>(
                        new parcel_buffer_type(
                            allocator<message::payload_size>(memory_pool_)
                        )
                    );
                buffer_->data_.reserve(arg_size);
            }
            return buffer_;
        }

        /// Asynchronously write a data structure to the socket.
        template <typename Handler, typename ParcelPostprocess>
        void async_write(Handler handler, ParcelPostprocess parcel_postprocess)
        {
            /// Increment sends and begin timer.
            buffer_->data_point_.time_ = timer_.elapsed_nanoseconds();
            HPX_ASSERT(buffer_->num_chunks_.first == 0u);

            handler_ = handler;
            postprocess_ = parcel_postprocess;

            send_size();
            add_sender(parcelport_, shared_from_this());
        }

        bool done()
        {
            next_function_type f = 0;
            {
                boost::lock_guard<hpx::lcos::local::spinlock> l(mtx_);
                f = next_;
            }
            if(f != 0)
            {
                if(((*this).*f)())
                {
                    error_code ec;
                    handler_(ec, buffer_->data_.size());
                    buffer_->data_point_.time_ = timer_.elapsed_nanoseconds()
                        - buffer_->data_point_.time_;
                    parcels_sent_.add_data(buffer_->data_point_);
                    postprocess_function_type pp;
                    std::swap(pp, postprocess_);
                    pp(ec, there_, shared_from_this());
                    mr_.reset();
                    return true;
                }
            }
            return false;
        }

    private:
        bool send_size()
        {
            boost::system::error_code & ec = boost::system::throws;
            std::size_t size = buffer_->data_.size();
            HPX_ASSERT(buffer_->num_chunks_.first == 0u);
            if(size <= message::payload_size)
            {
                context_.connection().send_small_msg(&buffer_->data_[0], size, ec);
                return next(&sender::sent_small_msg);
            }
            else
            {
                context_.send_size(buffer_->data_.size(), ec);
                mr_ = register_buffer(
                    parcelport_
                  , context_.pd_
                  , &buffer_->data_[0]
                  , buffer_->data_.size()
                  , IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

                adapted_mr_ = *mr_.mr_;
                adapted_mr_.addr = &buffer_->data_[0];
                adapted_mr_.length = buffer_->data_.size();

                return next(&sender::sent_size);
            }
        }

        bool sent_small_msg()
        {
            boost::system::error_code & ec = boost::system::throws;
            if(context_.check_wc<false>(MSG_SIZE, ec))
            {
                context_.post_receive(ec);
                return next(&sender::read_ack);
            }
            return false;
        }

        bool sent_size()
        {
            boost::system::error_code & ec = boost::system::throws;
            if(context_.check_wc<false>(MSG_SIZE, ec))
            {
                context_.post_receive(ec);
                return next(&sender::read_mr);
            }
            return false;
        }

        bool read_mr()
        {
            if(context_.check_wc<false>(MSG_MR, boost::system::throws))
            {
                return send_data();
            }
            return false;
        }

        bool send_data()
        {
            context_.connection().write_remote(
                &buffer_->data_[0]
              , &adapted_mr_
              , buffer_->data_.size()
              , boost::system::throws
            );
            return next(&sender::sent_data);
        }

        bool sent_data()
        {
            boost::system::error_code & ec = boost::system::throws;
            if(context_.check_wc<false>(MSG_DATA, ec))
            {
                context_.post_receive(ec);
                return next(&sender::read_ack);
            }
            return false;
        }

        bool read_ack()
        {
            boost::system::error_code & ec = boost::system::throws;
            if(context_.check_wc<false>(MSG_DONE, ec))
            {
                next(0);
                return true;
            }
            return false;
        }

        bool next(next_function_type f)
        {
            boost::lock_guard<hpx::lcos::local::spinlock> l(mtx_);
            next_ = f;
            return false;
        }

        /// Context for the parcelport_connection.
        client_context context_;

        hpx::lcos::local::spinlock mtx_;
        next_function_type next_;

        handler_function_type handler_;
        postprocess_function_type postprocess_;

        connection_handler & parcelport_;

        /// the other (receiving) end of this connection
        parcelset::locality there_;
        /// Counters and their data containers.
        util::high_resolution_timer timer_;
        performance_counters::parcels::gatherer& parcels_sent_;

        util::memory_chunk_pool & memory_pool_;
        ibverbs_mr mr_;
        ibv_mr adapted_mr_;
    };
}}}}

#endif

#endif
