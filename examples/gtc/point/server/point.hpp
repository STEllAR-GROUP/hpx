//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_5EC73F4A_D9BD_4B9B_A02B_87A9BA04C043)
#define HPX_5EC73F4A_D9BD_4B9B_A02B_87A9BA04C043

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include "../../parameter.hpp"
#include "../../array1d.hpp"

using hpx::components::gtc::parameter;

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
{

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT point
      : public hpx::components::detail::managed_component_base<point>
    {
    public:
        enum actions
        {
            point_init = 0,
            point_load = 1,
            point_chargei = 2,
            point_get_densityi = 3,
            point_get_zonali = 4,
            point_smooth = 5
        };

        point()
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the point with the given point file. 
        void init(std::size_t objectid, parameter const& par);
            
        void load(std::size_t objectid,parameter const& par);

        void chargei(std::size_t istep, std::vector<hpx::naming::id_type> const& point_components, parameter const& par);

        void smooth(std::size_t iflag, std::vector<hpx::naming::id_type> const& point_components, parameter const& par);

        bool chargei_callback(std::size_t i,std::valarray<double> const& density);

        bool chargei_zonali_callback(std::size_t i,
                           std::vector<double> const& zonali);

        std::valarray<double> get_densityi();

        std::vector<double> get_zonali();

       ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action2<
            // Component server type.
            point,
            // Action code.
            point_init,
            // Arguments of this action.
            std::size_t,
            parameter const&,
            // Method bound to this action. 
            &point::init
        > init_action;

        typedef hpx::actions::action2<
            // Component server type.
            point,
            // Action code.
            point_load,
            // Arguments of this action.
            std::size_t,
            parameter const&,
            // Method bound to this action. 
            &point::load
        > load_action;

        typedef hpx::actions::action3<
            // Component server type.
            point,
            // Action code.
            point_chargei,
            // Arguments of this action.
            std::size_t,
            std::vector<hpx::naming::id_type> const&,
            parameter const&,
            // Method bound to this action. 
            &point::chargei
        > chargei_action;

        typedef hpx::actions::action3<
            // Component server type.
            point,
            // Action code.
            point_smooth,
            // Arguments of this action.
            std::size_t,
            std::vector<hpx::naming::id_type> const&,
            parameter const&,
            // Method bound to this action. 
            &point::smooth
        > smooth_action;

        typedef hpx::actions::result_action0<
            // Component server type.
            point,
            // Return type.
            std::valarray<double>,
            // Action code.
            point_get_densityi,
            // Method bound to this action.
            &point::get_densityi
        > get_densityi_action;

        typedef hpx::actions::result_action0<
            // Component server type.
            point,
            // Return type.
            std::vector<double>,
            // Action code.
            point_get_zonali,
            // Method bound to this action.
            &point::get_zonali
        > get_zonali_action;

    private:
        std::size_t idx_;
        double tauii_;
        std::size_t mzeta_;
        double deltaz_;
        double deltar_;
        double zetamin_,zetamax_;
        std::size_t toroidal_domain_location_;
        double pi_;
        std::size_t mi_; // # of ions per proc
        std::size_t me_; // # of electrons per proc

        std::size_t left_pe_;
        std::size_t right_pe_;
        std::size_t particle_domain_location_;
        std::size_t myrank_toroidal_;

        std::vector<std::size_t> itran_,mtheta_,igrid_;
        std::vector<double> qtinv_,deltat_,rtemi_,rteme_; 
        std::vector<double> rden_,pmarki_,pmarke_,phi00_,phip00_;
        std::vector<double> hfluxpsi_,zonali_,zonale_,gradt_;
        array<double> eigenmode_;
        array<double> pgyro_,tgyro_,markeri_,densityi_,phi_,evector_;
        array<double> jtp1_,jtp2_,wtp1_,wtp2_,dtemper_,heatflux_;
        array<double> zion_,zion0_,jtion0_,jtion1_,wpion_,wtion0_,wtion1_;
        std::vector<double> kzion_,wzion_;

        std::vector<double> pfluxpsi_,rdtemi_,rdteme_;

        std::valarray<double> recvr_;
        std::vector<double> adum_;

        std::size_t mtdiag_;
        std::size_t mgrid_;
    };
}}

HPX_REGISTER_ACTION_DECLARATION_EX(
              gtc::server::point::init_action,
              gtc_point_init_action)

HPX_REGISTER_ACTION_DECLARATION_EX(
              gtc::server::point::load_action,
              gtc_point_load_action)

HPX_REGISTER_ACTION_DECLARATION_EX(
              gtc::server::point::chargei_action,
              gtc_point_chargei_action)

HPX_REGISTER_ACTION_DECLARATION_EX(
              gtc::server::point::get_densityi_action,
              gtc_point_get_densityi_action)

HPX_REGISTER_ACTION_DECLARATION_EX(
              gtc::server::point::get_zonali_action,
              gtc_point_get_zonali_action)

HPX_REGISTER_ACTION_DECLARATION_EX(
              gtc::server::point::smooth_action,
              gtc_point_smooth_action)

#endif

