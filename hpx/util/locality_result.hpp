//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_LOCALITY_RESULT_MAY_23_2012_1254PM)
#define HPX_UTIL_LOCALITY_RESULT_MAY_23_2012_1254PM

#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/traits/promise_remote_result.hpp>
#include <hpx/traits/promise_local_result.hpp>

#include <boost/make_shared.hpp>
#include <boost/iterator_adaptors.hpp>
#include <boost/foreach.hpp>

#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT locality_result_iterator;

    ///////////////////////////////////////////////////////////////////////////
    // exposed functionality of this component
    struct remote_locality_result
    {
        typedef std::vector<naming::gid_type>::iterator iterator;
        typedef std::vector<naming::gid_type>::const_iterator const_iterator;
        typedef std::vector<naming::gid_type>::value_type value_type;

        remote_locality_result()
        {}

        remote_locality_result(naming::gid_type const& prefix,
                components::component_type type)
          : prefix_(prefix), type_(type)
        {}

        naming::gid_type prefix_;             ///< prefix of the locality
        std::vector<naming::gid_type> gids_;  ///< gids of the created components
        components::component_type type_;     ///< type of created components

        iterator begin() { return gids_.begin(); }
        const_iterator begin() const { return gids_.begin(); }
        iterator end() { return gids_.end(); }
        const_iterator end() const { return gids_.end(); }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & prefix_ & gids_ & type_;
        }
    };

    // same as remote_locality_result, except it stores id_type's
    struct locality_result
    {
        typedef std::vector<naming::id_type>::iterator iterator;
        typedef std::vector<naming::id_type>::const_iterator const_iterator;
        typedef std::vector<naming::id_type>::value_type value_type;

        locality_result()
        {}

        locality_result(remote_locality_result const& results)
          : prefix_(results.prefix_), type_(results.type_)
        {
            BOOST_FOREACH(naming::gid_type const& gid, results.gids_)
            {
                gids_.push_back(naming::id_type(gid, naming::id_type::managed));
            }
        }

        iterator begin() { return gids_.begin(); }
        const_iterator begin() const { return gids_.begin(); }
        iterator end() { return gids_.end(); }
        const_iterator end() const { return gids_.end(); }

        naming::gid_type prefix_;             ///< prefix of the locality
        std::vector<naming::id_type> gids_;   ///< gids of the created components
        components::component_type type_;     ///< type of created components
    };


    ///////////////////////////////////////////////////////////////////////////
    /// Special segmented iterator allowing to iterate over all gids referenced
    /// by an instance of a \a distributing_factory#result_type
    class HPX_EXPORT locality_result_iterator
      : public boost::iterator_facade<
            locality_result_iterator, naming::id_type,
            boost::forward_traversal_tag, naming::id_type const&>
    {
    private:
        typedef std::vector<util::locality_result> result_type;
        typedef result_type::value_type locality_result_type;

        struct HPX_EXPORT data
        {
            data();
            data(result_type::const_iterator begin, result_type::const_iterator end);

            void increment();
            bool equal(data const& rhs) const;
            naming::id_type const& dereference() const;

            result_type::const_iterator current_;
            result_type::const_iterator end_;
            locality_result_type::const_iterator current_gid_;

            bool is_at_end_;
        };

    public:
        /// construct begin iterator
        locality_result_iterator(result_type const& results)
          : data_(new data(results.begin(), results.end()))
        {}

        /// construct end iterator
        locality_result_iterator()
          : data_(boost::make_shared<data>())
        {}

    private:
        boost::shared_ptr<data> data_;

        /// support functions needed for a forward iterator
        friend class boost::iterator_core_access;

        void increment()
        {
            data_->increment();
        }

        void decrement() {}
        void advance(difference_type) {}

        bool equal(locality_result_iterator const& rhs) const
        {
            return data_->equal(*rhs.data_);
        }

        naming::id_type const& dereference() const
        {
            return data_->dereference();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT std::pair<locality_result_iterator, locality_result_iterator>
        locality_results(std::vector<util::locality_result> const& v);
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    // we need to specialize this template to allow for automatic conversion of
    // the vector<remote_locality_result> to a vector<locality_result>
    template <>
    struct get_remote_result<
        std::vector<util::locality_result>,
        std::vector<util::remote_locality_result> >
    {
        typedef std::vector<util::locality_result> result_type;
        typedef std::vector<util::remote_locality_result>
            remote_result_type;

        static result_type call(remote_result_type const& rhs)
        {
            result_type result;
            BOOST_FOREACH(remote_result_type::value_type const& r, rhs)
            {
                result.push_back(result_type::value_type(r));
            }
            return result;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct promise_remote_result<std::vector<util::locality_result> >
      : boost::mpl::identity<std::vector<util::remote_locality_result> >
    {};

    template <>
    struct promise_local_result<std::vector<util::remote_locality_result> >
      : boost::mpl::identity<std::vector<util::locality_result> >
    {};
}}

#include <hpx/config/warnings_suffix.hpp>

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::base_lco_with_value<
        std::vector<hpx::util::remote_locality_result>
    >::set_value_action
  , set_value_action_factory_locality_result)

#endif
