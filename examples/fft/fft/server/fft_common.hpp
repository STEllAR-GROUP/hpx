#if !defined(HPX_SGdMWM8umRRplhE6artIzPnYioCV0cnrf2ZOremX)
#define HPX_SGdMWM8umRRplhE6artIzPnYioCV0cnrf2ZOremX
namespace fft
{
    //////////////////////////////////////////////////////////////////////////
    typedef std::vector<std::pair<hpx::naming::gid_type, std::size_t > >
        loc_rank_type;
    typedef std::pair<hpx::naming::id_type, std::size_t> comp_rank_pair_type;
    typedef std::vector<comp_rank_pair_type> comp_rank_vec_type;
    //////////////////////////////////////////////////////////////////////////
    struct HPX_COMPONENT_EXPORT complex_type
        //struct HPX_COMPONENT_EXPORT complex_type
    {
        complex_type():re(0), im(0)
        {}
        complex_type(double real, double imag): re(real), im(imag)
        {}
        ~complex_type(){}
        double re, im;
    };

    struct HPX_COMPONENT_EXPORT config_data
    {
        config_data() : num_workers_(0), num_localities_(0){}

        std::string data_filename_;
        std::string symbolic_name_;
        std::size_t num_workers_;
        std::size_t num_localities_;
        hpx::naming::id_type comp_id_;
        std::size_t comp_cardinality_;
        comp_rank_vec_type comp_rank_vec_;
        /// < a locality that is ellibigle to pull >
        bool valid_;
        bool complete_;
        std::size_t current_level_;
    };
}

////////////////////////////////////////////////////////////////////////////////
//Non intrusive serialization
namespace boost { namespace serialization
{
    template <typename Archive>
    void serialize(Archive& ar, fft::complex_type& type, unsigned int const);

   template <typename Archive>                                                  
   void serialize(Archive&, fft::config_data&, unsigned int const);         
}}

#endif //HPX_SGdMWM8umRRplhE6artIzPnYioCV0cnrf2ZOremX
