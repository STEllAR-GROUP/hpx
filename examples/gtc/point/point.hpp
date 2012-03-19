//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AA0F78C3_B11C_4173_8FA8_C1A6073FB9BA)
#define HPX_AA0F78C3_B11C_4173_8FA8_C1A6073FB9BA

#include <hpx/runtime/components/client_base.hpp>

#include "stubs/point.hpp"

namespace gtc
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a \a gtc::server::point
    /// components.
    class point
      : public hpx::components::client_base<point, stubs::point>
    {
        typedef hpx::components::client_base<point, stubs::point>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        point()
        {}

        /// Create a client side representation for the existing
        /// \a gtc::server::point instance with the given GID.
        point(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the \a gtc::server::point instance with the
        /// given point file.
        hpx::lcos::future<void> init_async(std::size_t objectid,parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_,objectid,par);
        }

        /// Initialize the \a gtc::server::point instance with the
        /// given point file.
        void init(std::size_t objectid,parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init_async(gid_,objectid,par);
        }

        hpx::lcos::future<void> load_async(std::size_t objectid,parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::load_async(gid_,objectid,par);
        }

        void load(std::size_t objectid,parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::load_async(gid_,objectid,par);
        }

        hpx::lcos::future<void> chargei_async(std::size_t istep,
            std::vector<hpx::naming::id_type> const& point_components,
            parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::chargei_async(gid_,istep,point_components,par);
        }

        void chargei(std::size_t istep,
                    std::vector<hpx::naming::id_type> const& point_components,
                    parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::chargei(gid_,istep,point_components,par);
        }

        hpx::lcos::future< std::valarray<double> > get_densityi_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_densityi_async(gid_);
        }

        std::valarray<double> get_densityi()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_densityi(gid_);
        }

        hpx::lcos::future< std::vector<double> > get_zonali_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_zonali_async(gid_);
        }

        std::vector<double> get_zonali()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_zonali(gid_);
        }

        hpx::lcos::future<void> smooth_async(std::size_t iflag,
            std::vector<hpx::naming::id_type> const& point_components,
            std::size_t idiag,
            parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::smooth_async(gid_,iflag,point_components,idiag,par);
        }

        void smooth(std::size_t iflag,
                    std::vector<hpx::naming::id_type> const& point_components,
                    std::size_t idiag,
                    parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::smooth(gid_,iflag,point_components,idiag,par);
        }

        hpx::lcos::future< std::valarray<double> > get_phi_async(std::size_t depth)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_phi_async(gid_,depth);
        }

        std::valarray<double> get_phi(std::size_t depth)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_phi(gid_,depth);
        }

        hpx::lcos::future< std::vector<double> > get_eachzeta_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_eachzeta_async(gid_);
        }

        std::vector<double> get_eachzeta()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_eachzeta(gid_);
        }

        hpx::lcos::future<void> field_async(
            std::vector<hpx::naming::id_type> const& point_components,
            parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::field_async(gid_,point_components,par);
        }

        void field( std::vector<hpx::naming::id_type> const& point_components,
                    parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::field(gid_,point_components,par);
        }

        hpx::lcos::future< std::valarray<double> > get_evector_async(std::size_t depth,std::size_t extent)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_evector_async(gid_,depth,extent);
        }

        std::valarray<double> get_evector(std::size_t depth,std::size_t extent)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_evector(gid_,depth,extent);
        }

        hpx::lcos::future<void> pushi_async(
            std::size_t irk,
            std::size_t istep,
            std::size_t idiag,
            std::vector<hpx::naming::id_type> const& point_components,
            parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::pushi_async(gid_,irk,istep,idiag,point_components,par);
        }

        void pushi( std::size_t irk,
                    std::size_t istep,
                    std::size_t idiag,
                    std::vector<hpx::naming::id_type> const& point_components,
                    parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::pushi(gid_,irk,istep,idiag,point_components,par);
        }

        hpx::lcos::future< std::vector<double> > get_dden_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_dden_async(gid_);
        }

        std::vector<double> get_dden()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_dden(gid_);
        }

        hpx::lcos::future< std::vector<double> > get_dtem_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_dtem_async(gid_);
        }

        std::vector<double> get_dtem()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_dtem(gid_);
        }

        hpx::lcos::future<void> shifti_async(
            std::vector<hpx::naming::id_type> const& point_components,
            parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::shifti_async(gid_,point_components,par);
        }

        void shifti( std::vector<hpx::naming::id_type> const& point_components,
                    parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::shifti(gid_,point_components,par);
        }

        hpx::lcos::future< std::size_t > get_msend_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_msend_async(gid_);
        }

        std::size_t get_msend()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_msend(gid_);
        }

        hpx::lcos::future< std::vector<std::size_t> > get_msendright_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_msendright_async(gid_);
        }

        std::vector<std::size_t> get_msendright()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_msendright(gid_);
        }

        hpx::lcos::future< array<double> > get_sendright_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_sendright_async(gid_);
        }

        array<double> get_sendright()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_sendright(gid_);
        }

        hpx::lcos::future< std::vector<std::size_t> > get_msendleft_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_msendleft_async(gid_);
        }

        std::vector<std::size_t> get_msendleft()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_msendleft(gid_);
        }

        hpx::lcos::future< array<double> > get_sendleft_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_sendleft_async(gid_);
        }

        array<double> get_sendleft()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_sendleft(gid_);
        }

        hpx::lcos::future<void> poisson_async(
            std::size_t iflag,
            std::size_t istep,
            std::size_t irk,
            std::vector<hpx::naming::id_type> const& point_components,
            parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::poisson_async(gid_,iflag,istep,irk,
                                                   point_components,par);
        }

        void poisson( std::size_t iflag,
                      std::size_t istep,
                      std::size_t irk,
                      std::vector<hpx::naming::id_type> const& point_components,
                    parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::poisson(gid_,iflag,istep,irk,point_components,par);
        }


    };
}

#endif

