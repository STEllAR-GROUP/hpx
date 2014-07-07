//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_PARCELPORT_MPI)
#include <mpi.h>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/policies/mpi/connection_handler.hpp>
#include <hpx/runtime/parcelset/policies/mpi/sender.hpp>
#include <hpx/runtime/parcelset/policies/mpi/receiver.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/mpi_environment.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/shared_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    std::vector<std::string> connection_handler::runtime_configuration()
    {
        std::vector<std::string> lines;

        using namespace boost::assign;
        lines +=
#if defined(HPX_PARCELPORT_MPI_ENV)
            "env = ${HPX_PARCELPORT_MPI_ENV:" HPX_PARCELPORT_MPI_ENV "}",
#else
            "env = ${HPX_PARCELPORT_MPI_ENV:PMI_RANK,OMPI_COMM_WORLD_SIZE}",
#endif
            "multithreaded = ${HPX_PARCELPORT_MPI_MULTITHREADED:0}",
            "io_pool_size = 1",
            "use_io_pool = 1"
            ;

        return lines;
    }

    connection_handler::connection_handler(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : base_type(ini, on_start_thread, on_stop_thread)
      , stopped_(false)
      , handling_messages_(false)
      , next_tag_(1)
      , use_io_pool_(true)
    {
        if (here_.get_type() != connection_mpi) {
            HPX_THROW_EXCEPTION(network_error, "mpi::parcelport::parcelport",
                "this parcelport was instantiated to represent an unexpected "
                "locality type: " + get_connection_type_name(here_.get_type()));
        }
        std::string use_io_pool =
            ini.get_entry("hpx.parcel.mpi.use_io_pool", "1");
        if(boost::lexical_cast<int>(use_io_pool) == 0)
        {
            use_io_pool_ = false;
        }
    }

    connection_handler::~connection_handler()
    {
    }

    bool connection_handler::do_run()
    {
        MPI_Comm_dup(MPI_COMM_WORLD, &communicator_);
        acceptor_.run(communicator_);
        background_work();      // schedule message handler
        return true;
    }

    void connection_handler::do_stop()
    {
        // Mark stopped state
        stopped_ = true;
        // Wait until message handler returns
        std::size_t k = 0;
        while(handling_messages_)
        {
            hpx::lcos::local::spinlock::yield(k);
            ++k;
        }
    }

    // Make sure all pending requests are handled
    void connection_handler::background_work()
    {
        if (stopped_)
            return;

        // Atomically set handling_messages_ to true, if another work item hasn't
        // started executing before us.
        bool false_ = false;
        if (!handling_messages_.compare_exchange_strong(false_, true))
            return;

        if(!hpx::is_starting() && !use_io_pool_)
        {
            hpx::applier::register_thread_nullary(
                util::bind(&connection_handler::handle_messages, this),
                "mpi::connection_handler::handle_messages",
                threads::pending, true, threads::thread_priority_critical);
        }
        else
        {
            boost::asio::io_service& io_service = io_service_pool_.get_io_service();
            io_service.post(util::bind(&connection_handler::handle_messages, this));
        }
    }

    std::string connection_handler::get_locality_name() const
    {
        return util::mpi_environment::get_processor_name();
    }

    boost::shared_ptr<sender> connection_handler::create_connection(
        naming::locality const& l, error_code& ec)
    {
        boost::shared_ptr<sender> sender_connection(new sender(
            communicator_, get_next_tag(), l,
            *this, this->parcels_sent_));

        return sender_connection;
    }

    void connection_handler::enable_parcel_handling(bool new_state)
    {
        if(enable_parcel_handling_)
        {
            background_work();
        }
        else
        {
            // Wait until message handler returns
            std::size_t k = 0;
            while(handling_messages_)
            {
                hpx::lcos::local::spinlock::yield(k);
                ++k;
            }
        }
    }

    void connection_handler::add_sender(
        boost::shared_ptr<sender> const& sender_connection)
    {
        hpx::lcos::local::spinlock::scoped_lock l(senders_mtx_);
        senders_.push_back(sender_connection);
    }

    void add_sender(connection_handler & handler,
        boost::shared_ptr<sender> const& sender_connection)
    {
        handler.add_sender(sender_connection);
    }

    void connection_handler::close_sender_connection(int tag, int rank)
    {
        {
            hpx::lcos::local::spinlock::scoped_lock l(close_mtx_);
            pending_close_requests_.push_back(std::make_pair(tag, rank));
        }
    }

    void close_sender_connection(connection_handler & handler, int tag, int rank)
    {
        handler.close_sender_connection(tag, rank);
    }

    int connection_handler::get_next_tag()
    {
        int tag = 0;
        {
            hpx::lcos::local::spinlock::scoped_lock l(tag_mtx_);
            if(!free_tags_.empty())
            {
                tag = free_tags_.front();
                free_tags_.pop_front();
                return tag;
            }
        }
        if(next_tag_ + 1 > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        {
            HPX_THROW_EXCEPTION(network_error, "mpi::connection_handler::get_next_tag",
                "there are no free tags available. Consider increasing the cache size");
            return -1;
        }
        tag = static_cast<int>(next_tag_++);
        return tag;
    }

    namespace detail
    {
        struct handling_messages
        {
            handling_messages(boost::atomic<bool>& handling_messages_flag)
              : handling_messages_(handling_messages_flag)
            {}

            ~handling_messages()
            {
                handling_messages_.store(false);
            }

            boost::atomic<bool>& handling_messages_;
        };
    }

    void connection_handler::handle_messages()
    {
        detail::handling_messages hm(handling_messages_);       // reset on exit

        bool bootstrapping = hpx::is_starting();
        bool has_work = true;
        std::size_t k = 0;

        hpx::util::high_resolution_timer t;
        std::list<std::pair<int, MPI_Request> > close_requests;

        // We let the message handling loop spin for another 2 seconds to avoid the
        // costs involved with posting it to asio
        while(bootstrapping || (!stopped_ && has_work) || (!has_work && t.elapsed() < 2.0))
        {
            // break the loop if someone requested to pause the parcelport
            if(!enable_parcel_handling_) break;

            // handle all send requests
            {
                hpx::lcos::local::spinlock::scoped_lock l(senders_mtx_);
                for(
                    senders_type::iterator it = senders_.begin();
                    !stopped_ && enable_parcel_handling_ && it != senders_.end();
                    /**/)
                {
                    if((*it)->done())
                    {
                        it = senders_.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                has_work = !senders_.empty();
            }

            // Send the pending close requests
            {
                hpx::lcos::local::spinlock::scoped_lock l(close_mtx_);
                typedef std::pair<int, int> pair_type;

                BOOST_FOREACH(pair_type p, pending_close_requests_)
                {
                    header close_request = header::close(p.first, p.second);
                    close_requests.push_back(std::make_pair(p.first, MPI_Request()));
                    MPI_Isend(
                        close_request.data(),         // Data pointer
                        close_request.data_size_,     // Size
                        close_request.type(),         // MPI Datatype
                        close_request.rank(),         // Destination
                        0,                            // Tag
                        communicator_,                // Communicator
                        &close_requests.back().second
                    );
                }
                pending_close_requests_.clear();
            }

            // add new receive requests
            std::pair<bool, header> next(acceptor_.next_header());
            if(next.first)
            {
                boost::shared_ptr<receiver> rcv;
                header h = next.second;

                receivers_tag_map_type & tag_map = receivers_map_[h.rank()];

                receivers_tag_map_type::iterator jt = tag_map.find(h.tag());

                if(jt != tag_map.end())
                {
                    rcv = jt->second;
                }
                else
                {
                    rcv = boost::make_shared<receiver>(
                        communicator_
                      , get_next_tag()
                      , h.tag()
                      , h.rank()
                      , *this);
                    tag_map.insert(std::make_pair(h.tag(), rcv));
                }

                if(h.close_request())
                {
                    rcv->close();
                }
                else
                {
                    h.assert_valid();
                    if (static_cast<std::size_t>(h.size()) > this->get_max_message_size())
                    {
                        // report this problem ...
                        HPX_THROW_EXCEPTION(boost::asio::error::operation_not_supported,
                            "mpi::connection_handler::handle_messages",
                            "The size of this message exceeds the maximum inbound data size");
                        return;
                    }
                    if(rcv->async_read(h))
                    {
#ifdef HPX_DEBUG
                        receivers_type::iterator it = std::find(receivers_.begin(), receivers_.end(), rcv);
                        HPX_ASSERT(it == receivers_.end());

#endif
                        receivers_.push_back(rcv);
                    }
                }
            }

            // handle all receive requests
            for(receivers_type::iterator it = receivers_.begin();
                it != receivers_.end(); /**/)
            {
                boost::shared_ptr<receiver> rcv = *it;
                if(rcv->done())
                {
                    HPX_ASSERT(rcv->sender_tag() != -1);
                    if(rcv->closing())
                    {
                        receivers_tag_map_type & tag_map = receivers_map_[rcv->rank()];

                        receivers_tag_map_type::iterator jt = tag_map.find(rcv->sender_tag());
                        HPX_ASSERT(jt != tag_map.end());
                        tag_map.erase(jt);
                        {
                            hpx::lcos::local::spinlock::scoped_lock l(tag_mtx_);
                            free_tags_.push_back(rcv->tag());
                        }
                    }
                    it = receivers_.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            if(!has_work) has_work = !receivers_.empty();

            // handle completed close requests
            for(
                std::list<std::pair<int, MPI_Request> >::iterator it = close_requests.begin();
                !stopped_ && enable_parcel_handling_ && it != close_requests.end();
            )
            {
                int completed = 0;
                MPI_Status status;
                int ret = 0;
                ret = MPI_Test(&it->second, &completed, &status);
                HPX_ASSERT(ret == MPI_SUCCESS);
                if(completed && status.MPI_ERROR != MPI_ERR_PENDING)
                {
                    hpx::lcos::local::spinlock::scoped_lock l(tag_mtx_);
                    free_tags_.push_back(it->first);
                    it = close_requests.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            if(!has_work)
                has_work = !close_requests.empty();

            if (bootstrapping)
                bootstrapping = hpx::is_starting();

            if(has_work)
            {
                t.restart();
                k = 0;
            }
            else
            {
                if(enable_parcel_handling_)
                {
                    hpx::lcos::local::spinlock::yield(k);
                    ++k;
                }
            }
        }

        if(stopped_ == true)
        {
            MPI_Comm communicator = communicator_;
            communicator_ = 0;
            MPI_Comm_free(&communicator);
        }
    }
}}}}

#endif
