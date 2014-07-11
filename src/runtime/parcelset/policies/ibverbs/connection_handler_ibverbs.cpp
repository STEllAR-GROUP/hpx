//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_PARCELPORT_IBVERBS)

#include <hpx/exception_list.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/connection_handler.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/acceptor.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/sender.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/receiver.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/shared_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    std::vector<std::string> connection_handler::runtime_configuration()
    {
        std::vector<std::string> lines;

        using namespace boost::assign;
        lines +=
            "ifname = ${HPX_PARCEL_IBVERBS_IFNAME:" HPX_PARCELPORT_IBVERBS_IFNAME "}",
            "memory_chunk_size = ${HPX_PARCEL_IBVERBS_MEMORY_CHUNK_SIZE:"
                BOOST_PP_STRINGIZE(HPX_PARCELPORT_IBVERBS_MEMORY_CHUNK_SIZE) "}",
            "max_memory_chunks = ${HPX_PARCEL_IBVERBS_MAX_MEMORY_CHUNKS:"
                BOOST_PP_STRINGIZE(HPX_PARCELPORT_IBVERBS_MAX_MEMORY_CHUNKS) "}",
            "zero_copy_optimization = 0",
            "io_pool_size = 2",
            "use_io_pool = 1"
            ;

        return lines;
    }

    std::size_t connection_handler::memory_chunk_size(util::runtime_configuration const& ini)
    {
        std::string memory_chunk_size =
            ini.get_entry("hpx.parcel.ibverbs.memory_chunk_size", HPX_PARCELPORT_IBVERBS_MEMORY_CHUNK_SIZE);
        return boost::lexical_cast<std::size_t>(memory_chunk_size);
    }

    std::size_t connection_handler::max_memory_chunks(util::runtime_configuration const& ini)
    {
        std::string max_memory_chunks =
            ini.get_entry("hpx.parcel.ibverbs.max_memory_chunks", HPX_PARCELPORT_IBVERBS_MAX_MEMORY_CHUNKS);
        return boost::lexical_cast<std::size_t>(max_memory_chunks);
    }

    connection_handler::connection_handler(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread)
      : base_type(ini, on_start_thread, on_stop_thread)
      , memory_pool_(memory_chunk_size(ini), max_memory_chunks(ini))
      , stopped_(false)
      , handling_messages_(false)
      , handling_accepts_(false)
      , use_io_pool_(true)
    {
        // we never do zero copy optimization for this parcelport
        allow_zero_copy_optimizations_ = false;

        std::string use_io_pool =
            ini.get_entry("hpx.parcel.ibverbs.use_io_pool", "1");
        if(boost::lexical_cast<int>(use_io_pool) == 0)
        {
            use_io_pool_ = false;
        }

        device_list_ = 0;
        int num_devices = 0;
        device_list_ = ibv_get_device_list(&num_devices);
        for(int i = 0; i < num_devices; ++i)
        {
            ibv_context *ctx = ibv_open_device(device_list_[i]);
            get_pd(ctx, boost::system::throws);
            context_list_.push_back(ctx);
        }
    }

    connection_handler::~connection_handler()
    {
        boost::system::error_code ec;
        acceptor_.close(ec);
        BOOST_FOREACH(pd_map_type::value_type & pd_pair, pd_map_)
        {
            ibv_dealloc_pd(pd_pair.second);
        }
        BOOST_FOREACH(ibv_context * ctx, context_list_)
        {
            ibv_close_device(ctx);
        }
        ibv_free_device_list(device_list_);

    }

    bool connection_handler::do_run()
    {
        // initialize network
        std::size_t tried = 0;
        exception_list errors;
        naming::locality::iterator_type end = accept_end(here_);
        for (naming::locality::iterator_type it =
                accept_begin(here_, io_service_pool_.get_io_service(0), true);
             it != end; ++it, ++tried)
        {
            try {
                boost::asio::ip::tcp::endpoint ep = *it;

                acceptor_.bind(ep, boost::system::throws);
                ++tried;
                break;
            }
            catch (boost::system::system_error const&) {
                errors.add(boost::current_exception());
                continue;
            }
        }

        if (errors.size() == tried) {
            // all attempts failed
            HPX_THROW_EXCEPTION(network_error,
                "ibverbs::connection_handler::run", errors.get_message());
            return false;
        }
        time_send = 0;
        time_recv = 0;
        time_acct = 0;

        handling_accepts_ = true;
        boost::asio::io_service& io_service = io_service_pool_.get_io_service(1);
        io_service.post(util::bind(&connection_handler::handle_accepts, this));

        background_work();
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
        k = 0;
        while(handling_accepts_)
        {
            hpx::lcos::local::spinlock::yield(k);
            ++k;
        }

        std::cout << "Time for accepting: " << time_acct << "\n";
        std::cout << "Time for sending: " << time_send << "\n";
        std::cout << "Time for receiving: " << time_recv << "\n";

        // cancel all pending accept operations
        boost::system::error_code ec;
        acceptor_.close(ec);
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
                "ibverbs::connection_handler::handle_messages",
                threads::pending, true, threads::thread_priority_critical);
        }
        else
        {
            boost::asio::io_service& io_service = io_service_pool_.get_io_service(0);
            io_service.post(util::bind(&connection_handler::handle_messages, this));
        }
    }

    std::string connection_handler::get_locality_name() const
    {
        return "ibverbs";
    }

    boost::shared_ptr<sender> connection_handler::create_connection(
        naming::locality const& l, error_code& ec)
    {
        boost::asio::io_service& io_service = io_service_pool_.get_io_service(0);
        boost::shared_ptr<sender> sender_connection(new sender(*this, memory_pool_, l, parcels_sent_));

        // Connect to the target locality, retry if needed
        boost::system::error_code error = boost::asio::error::try_again;
        for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
        {
            try {
                naming::locality::iterator_type end = connect_end(l);
                for (naming::locality::iterator_type it =
                        connect_begin(l, io_service, true);
                      it != end; ++it)
                {
                    boost::asio::ip::tcp::endpoint const& ep = *it;

                    client_context& ctx = sender_connection->context();
                    ctx.close(ec);
                    ctx.connect(*this, ep, error);
                    if (!error)
                        break;
                }
                if (!error)
                    break;

                // wait for a really short amount of time
                if (hpx::threads::get_self_ptr()) {
                    this_thread::suspend(hpx::threads::pending,
                        "connection_handler(ibverbs)::create_connection");
                }
                else {
                    boost::this_thread::sleep(boost::get_system_time() +
                        boost::posix_time::milliseconds(
                            HPX_NETWORK_RETRIES_SLEEP));
                }
            }
            catch (boost::system::system_error const& e) {
                sender_connection->context().close(ec);
                sender_connection.reset();

                HPX_THROWS_IF(ec, network_error,
                    "ibverbs::parcelport::create_connection", e.what());
                return sender_connection;
            }
        }

        if (error) {
            sender_connection->context().close(ec);
            sender_connection.reset();

            hpx::util::osstream strm;
            strm << error.message() << " (while trying to connect to: "
                  << l << ")";

            HPX_THROWS_IF(ec, network_error,
                "ibverbs::parcelport::create_connection",
                hpx::util::osstream_get_string(strm));
            return sender_connection;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return sender_connection;
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

    ibv_pd *connection_handler::get_pd(ibv_context *context, boost::system::error_code & ec)
    {
        hpx::lcos::local::spinlock::scoped_lock l(pd_map_mtx_);
        typedef pd_map_type::iterator iterator;

        iterator it = pd_map_.find(context);
        if(it == pd_map_.end())
        {
            ibv_pd *pd = ibv_alloc_pd(context);
            if(pd == 0)
            {
                int verrno = errno;
                boost::system::error_code err(verrno, boost::system::system_category());
                HPX_IBVERBS_THROWS_IF(
                    ec
                  , err
                );
                return 0;
            }
            memory_pool_.register_chunk(pd);
            pd_map_.insert(std::make_pair(context, pd));
            mr_map_.insert(std::make_pair(pd, mr_cache_type()));
            return pd;
        }
        return it->second;

    }

    ibv_mr register_buffer(connection_handler & conn, ibv_pd * pd, char * buffer, std::size_t size, int access)
    {
        return conn.register_buffer(pd, buffer, size, access);
    }

    ibv_mr connection_handler::register_buffer(ibv_pd * pd, char * buffer, std::size_t size, int access)
    {
        ibv_mr result = memory_pool_.get_mr(pd, buffer, size);
        if(result.addr == 0)
        {
            hpx::lcos::local::spinlock::scoped_lock l(mr_map_mtx_);
            typedef mr_map_type::iterator iterator;
            iterator it = mr_map_.find(pd);
            HPX_ASSERT(it != mr_map_.end());

            typedef mr_cache_type::iterator jterator;
            jterator jt = it->second.find(buffer);
            if(jt == it->second.end())
            {
                ibverbs_mr mr
                    = ibverbs_mr(
                        pd
                      , buffer
                      , size
                      , access
                    );
                it->second.insert(std::make_pair(buffer, mr));
                return *mr.mr_;
            }
            return *jt->second.mr_;
        }
        return result;
    }

    ibv_mr get_mr(connection_handler & conn, ibv_pd * pd, char * buffer, std::size_t size)
    {
        return conn.get_mr(pd, buffer, size);
    }

    ibv_mr connection_handler::get_mr(ibv_pd * pd, char * buffer, std::size_t size)
    {
        ibv_mr result = memory_pool_.get_mr(pd, buffer, size);
        if(result.addr == 0)
        {
            hpx::lcos::local::spinlock::scoped_lock l(mr_map_mtx_);
            typedef mr_map_type::iterator iterator;
            iterator it = mr_map_.find(pd);
            HPX_ASSERT(it != mr_map_.end());

            typedef mr_cache_type::iterator jterator;
            jterator jt = it->second.find(buffer);
            HPX_ASSERT(jt != it->second.end());
            return *jt->second.mr_;
        }
        return result;
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
        // We let the message handling loop spin for another 2 seconds to avoid the
        // costs involved with posting it to asio
        while(bootstrapping || (!stopped_ && has_work) || (!has_work && t.elapsed() < 2.0))
        {
            // handle all sends ...
            has_work = do_sends();
            // handle all receives ...
            if(do_receives())
            {
                has_work = true;
            }

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
    }

    bool connection_handler::do_sends()
    {
        hpx::util::high_resolution_timer t;
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
        time_send += t.elapsed();
        return !senders_.empty();
    }

    bool connection_handler::do_receives()
    {
        hpx::util::high_resolution_timer t;
        hpx::lcos::local::spinlock::scoped_try_lock l(receivers_mtx_);

        for(
            receivers_type::iterator it = receivers_.begin();
            !stopped_ && enable_parcel_handling_ && it != receivers_.end();
            /**/)
        {
                boost::system::error_code ec;
                if((*it)->done(*this, ec))
                {
                    if(!ec)
                    {
                        HPX_IBVERBS_RESET_EC(ec)
                        (*it)->async_read(ec);
                    }
                }
                if(ec == boost::asio::error::eof
                || ec == boost::asio::error::operation_aborted)
                {
                    it = receivers_.erase(it);
                    continue;
                }
                ++it;
        }
        time_recv += t.elapsed();
        return !receivers_.empty();
    }

    void connection_handler::handle_accepts()
    {
        detail::handling_messages hm(handling_accepts_);       // reset on exit
        std::size_t k = 64;
        while(!stopped_)
        {
            hpx::util::high_resolution_timer t;
            boost::shared_ptr<receiver> rcv = acceptor_.accept(*this, memory_pool_, boost::system::throws);
            if(rcv)
            {
                rcv->async_read(boost::system::throws);
                {
                    hpx::lcos::local::spinlock::scoped_lock l(receivers_mtx_);
                    receivers_.push_back(rcv);
                }
            }
            time_acct += t.elapsed();

            hpx::lcos::local::spinlock::yield(k);
        }
    }
}}}}

#endif
