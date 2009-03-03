//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_BASE_MAR_01_2009_0134PM)
#define HPX_PERFORMANCE_COUNTERS_BASE_MAR_01_2009_0134PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <boost/cstdint.hpp>
#include <boost/serialization/serialization.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters 
{
    ///////////////////////////////////////////////////////////////////////////
    enum counter_type
    {
        /// \a counter_text shows a variable-length text string. It does not 
        /// deliver calculated values.
        ///
        /// Formula:  None
        /// Average:  None
        /// Type:     Text
        counter_text,

        /// \a counter_raw shows the last observed value only. It does 
        /// not deliver an average.
        ///
        /// Formula:  None. Shows raw data as collected.
        /// Average:  SUM(N) / x
        /// Type:     Instantaneous
        counter_raw,

        /// \a counter_average_base is used as the base data (denominator) in the 
        /// computation of time or count averages for the \a counter_average_count 
        /// and \a counter_average_timer counter types. This counter type 
        /// collects the last observed value only.
        ///
        /// Formula:  None. This counter uses raw data in factional calculations 
        ///           without delivering an output.
        /// Average:  SUM (N) / x
        /// Type:     Instantaneous
        counter_average_base,

        /// \a counter_average_count shows how many items are processed, on 
        /// average, during an operation. Counters of this type display a ratio 
        /// of the items processed (such as bytes sent) to the number of 
        /// operations completed. The ratio is calculated by comparing the 
        /// number of items processed during the last interval to the number of 
        /// operations completed during the last interval.
        ///
        /// Formula:  (N1 - N0) / (D1 - D0), where the numerator (N) represents 
        ///           the number of items processed during the last sample 
        ///           interval, and the denominator (D) represents the number 
        ///           of operations completed during the last two sample 
        ///           intervals.
        /// Average:  (Nx - N0) / (Dx - D0) 
        /// Type:     Average
        counter_average_count,

        /// \a counter_average_timer measures the average time it takes to 
        /// complete a process or operation. Counters of this type display a 
        /// ratio of the total elapsed time of the sample interval to the 
        /// number of processes or operations completed during that time. This 
        /// counter type measures time in ticks of the system clock. The 
        /// variable F represents the number of ticks per second. The value of 
        /// F is factored into the equation so that the result is displayed in 
        /// seconds.
        ///
        /// Formula:  ((N1 - N0) / F) / (D1 - D0), where the numerator (N) 
        ///           represents the number of ticks counted during the last 
        ///           sample interval, the variable F represents the frequency 
        ///           of the ticks, and the denominator (D) represents the 
        ///           number of operations completed during the last sample 
        ///           interval.
        /// Average:  ((Nx - N0) / F) / (Dx - D0)
        /// Type:     Average
        counter_average_timer,

        /// \a counter_elapsed_time shows the total time between when the 
        /// component or process started and the time when this value is 
        /// calculated. The variable F represents the number of time units that 
        /// elapse in one second. The value of F is factored into the equation 
        /// so that the result is displayed in seconds.
        ///
        /// Formula:  (D0 - N0) / F, where the denominator (D) represents the 
        ///           current time, the numerator (N) represents the time the 
        ///           object was started, and the variable F represents the 
        ///           number of time units that elapse in one second.
        /// Average:  (Dx - N0) / F
        /// Type:     Difference
        counter_elapsed_time,
    };

    ///////////////////////////////////////////////////////////////////////////
    enum counter_status
    {
        status_valid_data,      ///< No error occurred, data is valid
        status_new_data,        ///< Data is valid and different from last call
        status_invalid_data,    ///< Some error occurred, data is not value
    };

    ///////////////////////////////////////////////////////////////////////////
    /// A counter_path holds the elements of a full name for a counter instance
    /// Generally, a full name of a counter has the structure:
    ///
    ///    /objectname(parentinstancename/instancename#instanceindex)/countername
    ///
    /// i.e.
    ///    /threadmanager(localityprefix/thread#2)/queuelength
    ///
    struct counter_path_elements
    {
        std::string objectname_;          ///< the name of the performance object 
        std::string parentinstancename_;  ///< the name of the parent instance 
        std::string instancename_;        ///< the name of the object instance 
        boost::uint32_t instanceindex_;   ///< the instance index 
        std::string countername_;         ///< contains the counter name

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & objectname_ & parentinstancename_ & instancename_ & 
                 instanceindex_ & countername_;
        }
    };

    /// \brief Create a full name of a counter from the contents of the given 
    ///        \a counter_path_elements instance.
    HPX_API_EXPORT counter_status get_counter_name(
        counter_path_elements const& path, std::string& result);

    /// \brief Fill the given \a counter_path_elements instance from the given 
    ///        full name of a counter
    HPX_API_EXPORT counter_status get_counter_path_elements(
        std::string const& name, counter_path_elements& path);

    ///////////////////////////////////////////////////////////////////////////
    struct counter_info
    {
        counter_type type_;         ///< The type of the described counter
        boost::uint32_t version_;   ///< The version of the described counter
                                    ///< using the 0xMMmmSSSS scheme
        counter_status status_;     ///< The status of the counter object
        std::string fullname_;      ///< The full name of this counter 
        std::string helptext_;      ///< The full descriptive text for this counter

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & type_ & version_ & status_ & fullname_ & helptext_;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct counter_value
    {
        counter_status status_;     ///< The status of the counter value
        boost::int64_t time_;       ///< The local time when data was collected
        boost::int64_t value_;      ///< The current counter value

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & status_ & time_ & value_;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // parcel action code: the action to be performed on the destination 
    // object 
    enum actions
    {
        performance_counter_get_counter_info = 0,
        performance_counter_get_counter_value = 1,
    };

    class base_performance_counter 
    {
    protected:
        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_performance_counter() {}

        virtual void get_counter_info(counter_info& info) = 0;
        virtual void get_counter_value(counter_value& value) = 0;

    public:
        // components must contain a typedef for wrapping_type defining the
        // simple_component type used to encapsulate instances of this 
        // component
        typedef components::managed_component<base_performance_counter> wrapping_type;

        /// \brief finalize() will be called just before the instance gets 
        ///        destructed
        void finalize() {}

        // This is the component id. Every component needs to have a function
        // \a get_component_type() which is used by the generic action 
        // implementation to associate this component with a given action.
        static components::component_type get_component_type() 
        { 
            return components::component_performance_counter; 
        }
        static void set_component_type(components::component_type) 
        { 
        }

        ///////////////////////////////////////////////////////////////////////
        counter_info get_counter_info_nonvirt()
        {
            counter_info info;
            get_counter_info(info);
            return info;
        }

        counter_value get_counter_value_nonvirt()
        {
            counter_value value;
            get_counter_value(value);
            return value;
        }

        /// Each of the exposed functions needs to be encapsulated into an action
        /// type, allowing to generate all required boilerplate code for threads,
        /// serialization, etc.

        /// The \a get_counter_info_action may be used to ...
        typedef hpx::actions::result_action0<
            base_performance_counter, counter_info, 
            performance_counter_get_counter_info, 
            &base_performance_counter::get_counter_info_nonvirt
        > get_counter_info_action;

        /// The \a get_counter_value_action may be used to ...
        typedef hpx::actions::result_action0<
            base_performance_counter, counter_value, 
            performance_counter_get_counter_value, 
            &base_performance_counter::get_counter_value_nonvirt
        > get_counter_value_action;
    };

}}

#endif

