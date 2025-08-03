module;

// Include all system headers in global module fragment to prevent ODR violations
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <boost/config.hpp>
#include <boost/version.hpp>

// Define module-specific macros before including config
#define HPX_BUILD_MODULE
#include "config/include/hpx/config.hpp"

export module HPX.Core;

#include "version/include/hpx/version.hpp" 