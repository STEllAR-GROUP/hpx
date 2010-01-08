//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <algorithm>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>

#include <boost/version.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/lambda/lambda.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    // A parcel is submitted for transport at the source locality site to 
    // the parcel set of the locality with the put-parcel command
    // This function is synchronous.

    struct wait_for_put_parcel
    {
        typedef boost::mutex mutex_type;
        typedef boost::condition condition_type;

        wait_for_put_parcel(mutex_type& mtx, condition_type& cond,
              boost::system::error_code& saved_error, 
              bool& waiting, bool& finished)
          : mtx_(mtx), cond_(cond), saved_error_(saved_error),
            waiting_(waiting), finished_(finished)
        {}

        void operator()(boost::system::error_code const& e, std::size_t size)
        {
            mutex_type::scoped_lock l(mtx_);
            if (e) 
                saved_error_ = e;

            if (waiting_)
                cond_.notify_one();
            finished_ = true;
        }

        bool wait()
        {
            mutex_type::scoped_lock l(mtx_);

            if (finished_) 
                return true;

            boost::xtime xt;
            boost::xtime_get(&xt, boost::TIME_UTC);
            xt.sec += 5;        // wait for max. 5sec

            waiting_ = true;
            return cond_.timed_wait(l, xt);
        }

        mutex_type& mtx_;
        condition_type& cond_;
        boost::system::error_code& saved_error_;
        bool& waiting_;
        bool& finished_;
    };

    parcel_id parcelhandler::sync_put_parcel(parcel& p)
    {
        wait_for_put_parcel::mutex_type mtx;
        wait_for_put_parcel::condition_type cond;
        boost::system::error_code saved_error;
        bool waiting = false, finished = false;

        wait_for_put_parcel wfp (mtx, cond, saved_error, waiting, finished);
        parcel_id id = put_parcel(p, wfp);  // schedule parcel send
        if (!wfp.wait())                    // wait for the parcel being sent
            throw exception(network_error, "timeout");

        if (saved_error) 
            throw exception(network_error, saved_error.message());
        return id;
    }

    void parcelhandler::parcel_sink(parcelport& pp, 
        boost::shared_ptr<std::vector<char> > const& parcel_data)
    {
//         decode_parcel(parcel_data);
        if (NULL == tm_) {
            // this is supported for debugging purposes mainly, it results in
            // the direct execution of the parcel decoding
            decode_parcel(parcel_data);
        }
        else {
            // create a new thread which decodes and handles the parcel
            threads::thread_init_data data(
                boost::bind(&parcelhandler::decode_parcel, this, parcel_data),
                "decode_parcel");
            tm_->register_thread(data);
        }
    }

    threads::thread_state parcelhandler::decode_parcel(
        boost::shared_ptr<std::vector<char> > const& parcel_data)
    {
        parcel p;
        {
            // create a special io stream on top of in_buffer_
            typedef util::container_device<std::vector<char> > io_device_type;
            boost::iostreams::stream<io_device_type> io (*parcel_data.get());

            // De-serialize the parcel data
#if HPX_USE_PORTABLE_ARCHIVES != 0
            util::portable_binary_iarchive archive(io);
#else
            boost::archive::binary_iarchive archive(io);
#endif
            archive >> p;
        }

        // add parcel to incoming parcel queue
        parcels_.add_parcel(p);
        return threads::terminated;
    }

    bool parcelhandler::get_remote_prefixes(
        std::vector<naming::id_type>& prefixes) const
    {
        std::vector<naming::id_type> allprefixes;
        bool result = resolver_.get_prefixes(allprefixes);
        if (!result) return false;

        using boost::lambda::_1;
        std::remove_copy_if(allprefixes.begin(), allprefixes.end(), 
            std::back_inserter(prefixes), _1 == prefix_);
        return !prefixes.empty();
    }

///////////////////////////////////////////////////////////////////////////////
}}
