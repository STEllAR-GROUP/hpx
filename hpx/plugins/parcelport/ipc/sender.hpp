//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_IPC_SENDER_HPP
#define HPX_PARCELSET_POLICIES_IPC_SENDER_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_IPC)

#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>
#include <hpx/plugins/parcelport/ipc/data_buffer_cache.hpp>
#include <hpx/plugins/parcelport/ipc/data_window.hpp>
#include <hpx/plugins/parcelport/ipc/locality.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/protect.hpp>

#include <boost/asio/placeholders.hpp>

#include <memory>
#include <string>

namespace hpx { namespace parcelset { namespace policies { namespace ipc
{
    class sender
      : public parcelset::parcelport_connection<sender, data_buffer>
    {
    public:
        /// Construct a sending parcelport_connection with the given io_service.
        sender(boost::asio::io_service& io_service,
            parcelset::locality const& here, parcelset::locality const& there,
            data_buffer_cache& cache, parcelset::parcelport* pp,
            std::size_t connection_count)
          : window_(io_service), there_(there), pp_(pp),
            cache_(cache)
        {
            std::string fullname(here.get<locality>().address() + "." +
                std::to_string(here.get<locality>().port()) + "." +
                std::to_string(connection_count));

            window_.set_option(data_window::bound_to(fullname));
        }

        ~sender()
        {
            // gracefully and portably shutdown the socket
            boost::system::error_code ec;
            window_.shutdown(ec); // shut down data connection
            window_.close(ec);    // close the socket to give it back to the OS
        }

        /// Get the window associated with the sender.
        data_window& window() { return window_; }

        void verify(parcelset::locality const & parcel_locality_id)
        {
            HPX_ASSERT(parcel_locality_id == there_);
        }

        std::shared_ptr<parcel_buffer_type> get_buffer(parcel const & p,
            std::size_t arg_size)
        {
            // generate the name for this data_buffer
            std::string data_buffer_name(p.get_parcel_id().to_string());
            if(!buffer_)
            {
                // clear and preallocate out_buffer_ (or fetch from cache)
                buffer_ = std::make_shared<parcel_buffer_type>(
                    get_data_buffer((arg_size * 12) / 10 + 1024,
                        data_buffer_name)
                );
            }
            else
            {
                buffer_->data_ =
                    get_data_buffer((arg_size * 12) / 10 + 1024,
                        data_buffer_name);
            }
            return buffer_;
        }

        /// Asynchronously write a data structure to the socket.
        template <typename Handler, typename ParcelPostprocess>
        void async_write(Handler handler, ParcelPostprocess parcel_postprocess)
        {
            /// Increment sends and begin timer.
            buffer_->data_point_.time_ = timer_.elapsed_nanoseconds();

            // Write the serialized data to the socket.
            //
            // this additional wrapping of the handler into a bind object is
            // needed to keep  this sender object alive for the whole
            // write operation
            void (sender::*f)(boost::system::error_code const&, std::size_t,
                    Handler, ParcelPostprocess)
                = &sender::handle_write<Handler, ParcelPostprocess>;

            window_.async_write(buffer_->data_,
                util::bind(f, shared_from_this(),
                    boost::asio::placeholders::error, ::_2,
                    util::protect(handler), util::protect(parcel_postprocess)));
        }

        parcelset::locality const& destination() const
        {
            return there_;
        }

    protected:
        /// handle completed write operation
        template <typename Handler, typename ParcelPostprocess>
        void handle_write(boost::system::error_code const& e, std::size_t bytes,
            Handler handler, ParcelPostprocess parcelPostprocess)
        {
            // just call initial handler
            handler(e, bytes);

            // complete data point and push back onto gatherer
            buffer_->data_point_.time_ =
                timer_.elapsed_nanoseconds() - buffer_->data_point_.time_;
            pp->add_sent_data(buffer_->data_point_);

            // now handle the acknowledgment byte which is sent by the receiver
            void (sender::*f)(boost::system::error_code const&,
                      Handler, ParcelPostprocess)
                = &sender::handle_read_ack<Handler, ParcelPostprocess>;

            window_.async_read_ack(util::bind(f, shared_from_this(),
                boost::asio::placeholders::error, util::protect(handler)));
        }

        template <typename Handler, typename ParcelPostprocess>
        void handle_read_ack(boost::system::error_code const& e,
            Handler handler, ParcelPostprocess parcelPostprocess)
        {
            // now we can give this connection back to the cache
            reclaim_data_buffer(buffer_->data_);

            performance_counters::parcels::data_point& data = buffer_->data_point_;
            data.bytes_ = 0;
            data.time_ = 0;
            data.serialization_time_ = 0;
            data.num_parcels_ = 0;

            // Call post-processing handler, which will send remaining pending
            // parcels. Pass along the connection so it can be reused if more
            // parcels have to be sent.
            parcelPostprocess(e, there_, shared_from_this());
        }

    protected:
        data_buffer get_data_buffer(std::size_t size, std::string const& name)
        {
            data_buffer buffer;
            if (cache_.get(size, buffer))
                return buffer;

            return data_buffer(name.c_str(), size);
        }

        void reclaim_data_buffer(data_buffer& buffer)
        {
            cache_.add(buffer.size(), buffer);
            buffer.resize(0);
            buffer.reset();
        }

    private:
        /// Window for the sender.
        data_window window_;

        /// the other (receiving) end of this connection
        parcelset::locality there_;

        /// Counters and their data containers.
        util::high_resolution_timer timer_;
        performance_counters::parcels::gatherer& parcels_sent_;

        // data buffer cache
        data_buffer_cache& cache_;
    };
}}}}

#endif

#endif
