//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/util/polymorphic_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/serialize_intrusive_ptr.hpp>

#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/shared_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    // generate unique parcel id
    naming::gid_type parcel::generate_unique_id(boost::uint32_t locality_id_default)
    {
        static boost::atomic<boost::uint64_t> id(0);

        error_code ec(lightweight);        // ignore all errors
        boost::uint32_t locality_id = hpx::get_locality_id(ec);
        if (locality_id == naming::invalid_locality_id)
            locality_id = locality_id_default;

        naming::gid_type result = naming::get_gid_from_locality_id(locality_id);
        result.set_lsb(++id);
        return result;
    }

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////////
        template <typename Archive>
        void parcel_data::save(Archive& ar, bool has_source_id, bool has_continuation) const
        {
            // If we have a source id, serialize it.
            if (has_source_id)
                ar << source_id_;

            std::string action_name = action_->get_action_name();
            ar.save(action_name);

            action_->save(ar);

            // If we have a continuation, serialize it.
            if (has_continuation) {
                std::string continuation_name = continuation_->get_continuation_name();
                ar.save(continuation_name);

                continuation_->save(ar);
            }
        }

        template <typename Archive>
        void parcel_data::load(Archive& ar, bool has_source_id, bool has_continuation)
        {
            // Check for a source id.
            if (has_source_id)
                ar >> source_id_;

            std::string action_name;
            ar.load(action_name);

            action_ = util::polymorphic_factory<
                actions::base_action>::create(action_name);
            action_->load(ar);

            // handle continuation.
            if (has_continuation) {
                std::string continuation_name;
                ar.load(continuation_name);

                continuation_ = util::polymorphic_factory<
                    actions::continuation>::create(continuation_name);
                continuation_->load(ar);
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        template <typename Archive>
        void single_destination_parcel_data::save_optimized(Archive& ar) const
        {
            data_.has_source_id_ = source_id_ != naming::invalid_id;

            ar.save(data_);
            ar << dest_ << addr_;

            this->parcel_data::save(ar, data_.has_source_id_ != 0,
                data_.has_continuation_ != 0);
        }

        template <typename Archive>
        void single_destination_parcel_data::save_normal(Archive& ar) const
        {
            data_.has_source_id_ = source_id_ != naming::invalid_id;

            ar << data_.parcel_id_;
            ar << data_.start_time_ << data_.creation_time_;
            ar << data_.dest_size_;
            ar << data_.has_source_id_ << data_.has_continuation_;

            ar << dest_ << addr_;

            this->parcel_data::save(ar, data_.has_source_id_ != 0,
                data_.has_continuation_ != 0);
        }

        template <typename Archive>
        void single_destination_parcel_data::save(Archive& ar) const
        {
            if (ar.flags() & util::disable_array_optimization)
                save_normal(ar);
            else
                save_optimized(ar);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Archive>
        void single_destination_parcel_data::load_optimized(Archive & ar)
        {
            ar.load(data_);
            ar >> dest_ >> addr_;

            this->parcel_data::load(ar, data_.has_source_id_ != 0,
                data_.has_continuation_ != 0);
        }

        template <typename Archive>
        void single_destination_parcel_data::load_normal(Archive & ar)
        {
            ar >> data_.parcel_id_;
            ar >> data_.start_time_ >> data_.creation_time_;
            ar >> data_.dest_size_;
            ar >> data_.has_source_id_ >> data_.has_continuation_;

            ar >> dest_ >> addr_;

            this->parcel_data::load(ar, data_.has_source_id_ != 0,
                data_.has_continuation_ != 0);
        }

        template <typename Archive>
        void single_destination_parcel_data::load(Archive& ar)
        {
            if (ar.flags() & util::disable_array_optimization)
                load_normal(ar);
            else
                load_optimized(ar);
        }

        ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        template <typename Archive>
        void multi_destination_parcel_data::save_optimized(Archive& ar) const
        {
            data_.has_source_id_ = source_id_ != naming::invalid_id;

            ar.save(data_);
            ar << dests_ << addrs_;

            this->parcel_data::save(ar, data_.has_source_id_ != 0,
                data_.has_continuation_ != 0);
        }

        template <typename Archive>
        void multi_destination_parcel_data::save_normal(Archive& ar) const
        {
            data_.has_source_id_ = source_id_ != naming::invalid_id;

            ar << data_.parcel_id_;
            ar << data_.start_time_ << data_.creation_time_;
            ar << data_.dest_size_;
            ar << data_.has_source_id_ << data_.has_continuation_;
            ar << dests_ << addrs_;

            this->parcel_data::save(ar, data_.has_source_id_ != 0,
                data_.has_continuation_ != 0);
        }

        template <typename Archive>
        void multi_destination_parcel_data::save(Archive& ar) const
        {
            if (ar.flags() & util::disable_array_optimization)
                save_normal(ar);
            else
                save_optimized(ar);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Archive>
        void multi_destination_parcel_data::load_optimized(Archive& ar)
        {
            ar.load(data_);
            ar >> dests_ >> addrs_;

            this->parcel_data::load(ar, data_.has_source_id_ != 0,
                data_.has_continuation_ != 0);
        }

        template <typename Archive>
        void multi_destination_parcel_data::load_normal(Archive& ar)
        {
            ar >> data_.parcel_id_;
            ar >> data_.start_time_ >> data_.creation_time_;
            ar >> data_.dest_size_;
            ar >> data_.has_source_id_ >> data_.has_continuation_;
            ar >> dests_ >> addrs_;

            this->parcel_data::load(ar, data_.has_source_id_ != 0,
                data_.has_continuation_ != 0);
        }

        template <typename Archive>
        void multi_destination_parcel_data::load(Archive& ar)
        {
            if (ar.flags() & util::disable_array_optimization)
                load_normal(ar);
            else
                load_optimized(ar);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        // explicit instantiation for the correct archive types
        template HPX_EXPORT void
        single_destination_parcel_data::save(util::portable_binary_oarchive&) const;

        template HPX_EXPORT void
        single_destination_parcel_data::load(util::portable_binary_iarchive&);

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        template HPX_EXPORT void
        multi_destination_parcel_data::save(util::portable_binary_oarchive&) const;

        template HPX_EXPORT void
        multi_destination_parcel_data::load(util::portable_binary_iarchive&);
#endif

        ///////////////////////////////////////////////////////////////////////////
        std::ostream& operator<< (std::ostream& os, single_destination_parcel_data const& p)
        {
            os << "(" << p.dest_ << ":" << p.addr_ << ":";
            os << p.action_->get_action_name() << ")";
            return os;
        }

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        std::ostream& operator<< (std::ostream& os, multi_destination_parcel_data const& p)
        {
            os << "(";
            if (!p.dests_.empty())
                os << p.dests_[0] << ":" << p.addrs_[0] << ":";

            os << p.action_->get_action_name() << ")";
            return os;
        }
#endif
    }

    template <typename Archive>
    void parcel::save(Archive& ar, const unsigned int version) const
    {
        HPX_ASSERT(data_.get() != 0);

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        bool is_multi_destination = data_->is_multi_destination();
        ar.save(is_multi_destination);

        if (is_multi_destination) {
            boost::static_pointer_cast<
                detail::multi_destination_parcel_data
            >(data_)->save(ar);
        }
        else
#endif
        {
            boost::static_pointer_cast<
                detail::single_destination_parcel_data
            >(data_)->save(ar);
        }
    }

    template <typename Archive>
    void parcel::load(Archive& ar, const unsigned int version)
    {
        if (version > HPX_PARCEL_VERSION)
        {
            HPX_THROW_EXCEPTION(version_too_new,
                "parcel::load",
                "trying to load parcel with unknown version");
        }

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        bool is_multi_destination;
        ar.load(is_multi_destination);

        if (is_multi_destination) {
            boost::intrusive_ptr<detail::parcel_data> data(
                new detail::multi_destination_parcel_data);

            boost::static_pointer_cast<
                detail::multi_destination_parcel_data
            >(data)->load(ar);
            std::swap(data_, data);
        }
        else
#endif
        {
            boost::intrusive_ptr<detail::parcel_data> data(
                new detail::single_destination_parcel_data);

            boost::static_pointer_cast<
                detail::single_destination_parcel_data
            >(data)->load(ar);
            std::swap(data_, data);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
    template HPX_EXPORT void
    parcel::save(util::portable_binary_oarchive&, const unsigned int) const;

    template HPX_EXPORT void
    parcel::load(util::portable_binary_iarchive&, const unsigned int);

    ///////////////////////////////////////////////////////////////////////////
    std::ostream& operator<< (std::ostream& os, parcel const& p)
    {
        HPX_ASSERT(p.data_.get() != 0);
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        if (p.data_->is_multi_destination()) {
            os << *static_cast<detail::multi_destination_parcel_data const*>(p.data_.get());
        }
        else
#endif
        {
            os << *static_cast<detail::single_destination_parcel_data const*>(p.data_.get());
        }
        return os;
    }
}}

