//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_MPI_SENDER_HPP
#define HPX_PARCELSET_MPI_SENDER_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/parcelset/mpi/header.hpp>
#include <hpx/util/high_resolution_clock.hpp>

#include <boost/assert.hpp>
#include <boost/move/move.hpp>
#include <boost/noncopyable.hpp>

#include <vector>
#include <utility>

namespace hpx { namespace parcelset { namespace mpi
{
    struct parcel_buffer
    {
        typedef HPX_STD_FUNCTION<
            void(boost::system::error_code const&, std::size_t)
        > write_handler_type;

        int rank_;
        std::vector<char> buffer_;
        std::vector<write_handler_type> handlers_;
        performance_counters::parcels::data_point send_data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct sender : boost::noncopyable
    {
        typedef HPX_STD_FUNCTION<
            void(boost::system::error_code const&, std::size_t)
        > write_handler_type;

        enum sender_state
        {
            invalid,
            sending_header,
            sent_header,
            sending_data,
            sent_data,
            sender_done,
        };

        sender(
            header const & h,
            boost::shared_ptr<parcel_buffer> buffer,
            MPI_Comm communicator)
          : communicator_(communicator)
          , header_(h)
          , buffer_(buffer)
          , state_(invalid)
        {
            buffer_->send_data_.time_ = util::high_resolution_clock::now();

            header_.assert_valid();
            BOOST_ASSERT(header_.rank() != util::mpi_environment::rank());

            MPI_Isend(
                header_.data(), // Data pointer
                header_.data_size_,     // Size
                header_.type(), // MPI Datatype
                header_.rank(), // Destination
                0,              // Tag
                communicator_,  // Communicator
                &header_request_        // Request
                );
            state_ = sending_header;
        }

        bool done(parcelport & pp)
        {
            switch (state_)
            {
            case sending_header:
                {
                    int completed = 0;
                    MPI_Test(&header_request_, &completed, MPI_STATUS_IGNORE);
                    if(completed)
                    {
                        state_ = sent_header;
                        return done(pp);
                    }
                    break;
                }
            case sent_header:
                {
                    BOOST_ASSERT(static_cast<std::size_t>(header_.size()) ==
                        buffer_->buffer_.size());

                    MPI_Isend(
                        buffer_->buffer_.data(), // Data pointer
                        static_cast<int>(buffer_->buffer_.size()), // Size
                        MPI_CHAR,           // MPI Datatype
                        header_.rank(),     // Destination
                        header_.tag(),      // Tag
                        communicator_,      // Communicator
                        &data_request_      // Request
                        );
                    state_ = sending_data;
                    return done(pp);
                }
            case sending_data:
                {
                    int completed = 0;
                    MPI_Test(&data_request_, &completed, MPI_STATUS_IGNORE);
                    if(completed)
                    {
                        state_ = sent_data;
                        return done(pp);
                    }
                    break;
                }
            case sent_data:
                {
                    error_code ec;
                    BOOST_FOREACH(write_handler_type & f, buffer_->handlers_)
                    {
                        if(!f.empty())
                        {
                            f(ec, header_.size());
                        }
                    }
                    state_ = sender_done;
                    return done(pp);
                }
            case sender_done:
                buffer_->send_data_.time_ = util::high_resolution_clock::now() -
                        buffer_->send_data_.time_;
                pp.add_sent_data(buffer_->send_data_);
                return true;
            default:
            case invalid:
                {
                    BOOST_ASSERT(false);
                }
                return false;
            }
            return false;
        }

        int tag() const
        {
            return header_.tag();
        }

    private:
        MPI_Comm communicator_;
        header header_;
        boost::shared_ptr<parcel_buffer> buffer_;
        sender_state state_;

        MPI_Request header_request_;
        MPI_Request data_request_;
    };
}}}

#endif
