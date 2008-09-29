

#include <boost/any.hpp>
#include <map>
#include <utility>

#include <boost/plugin/export_plugin.hpp>

#include "weapon.hpp"

class Missile 
:   public Weapon 
{
public:
    Missile(std::string const& s, int i = 10) : name(s)
    {
        std::cout << "Created missile '" << s << "'\n";
        std::cout << "Fuel load is " << i << "\n";
    }
    Missile(std::string const& s, int i, int j) : name(s)
    {
        std::cout << "Created missile '" << s << "'\n";
        std::cout << "Fuel load is " << i << "\n";
        std::cout << "Speed is " << j << "\n";
    }

    void fire() { std::cout << "Fire " << name << "!\n"; }
    
    std::string name;
};

BOOST_PLUGIN_EXPORT(Weapon, Missile, "Missile");
BOOST_PLUGIN_EXPORT_LIST();
