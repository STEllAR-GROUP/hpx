

#include <string>
#include <iostream>

#include <boost/plugin.hpp>

#include "weapon.hpp"

using namespace std;

int main()
{

#if ! defined (BOOST_WINDOWS)
  string lib ("./library.so");
#else
  string lib ("./library.dll");
#endif

  try {
    /* get the handle of the library */
    boost::plugin::dll d (lib);
    boost::plugin::plugin_factory <Weapon> pf (d);

    cout << "*** Creating an instance of plugin class\n";
    std::auto_ptr <Weapon> w1 (pf.create ("Missile", "biz", 13)); 

    cout << "*** Creating an instance of plugin class\n";
    std::auto_ptr <Weapon> w2 (pf.create ("Missile", "wush")); 

    cout << "*** Creating an instance of plugin class\n";
    std::auto_ptr <Weapon> w3 (pf.create ("Missile", "wish", 10, 20)); 

    cout << "*** Calling method of the created instance\n";
    w1->fire();
    w2->fire();
    w3->fire();

  }
  catch ( std::logic_error const & e ) 
  {
    /* report error, and skip the library */
    std::cerr << "Could not load weapon: " << e.what () << std::endl;
  }
}

