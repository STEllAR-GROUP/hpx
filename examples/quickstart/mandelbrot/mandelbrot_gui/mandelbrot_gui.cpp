// mandelbrot_gui.cpp : Defines the class behaviors for the application.
//

#include "stdafx.h"
#include "mandelbrot_gui.h"
#include "mandelbrotDlg.h"

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage: mandelbrot [options]");
        desc_cmdline.add_options()
            ("help,h", "print out program usage (this message)")
            ("run_agas_server,r", "run AGAS server as part of this runtime instance")
            ("worker,w", "run this instance in worker (non-console) mode")
            ("agas,a", po::value<std::string>(), 
                "the IP address the AGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx,x", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
            ("threads,t", po::value<int>(), 
                "the number of operating system threads to spawn for this"
                "HPX locality")
            ("sizex,X", po::value<int>(), 
                "the horizontal (X) size of the generated image (default is 20)")
            ("sizey,Y", po::value<int>(), 
                "the vertical (Y) size of the generated image (default is 20)")
            ("iterations,i", po::value<int>(), 
                "the nmber of iterations to use for the mandelbrot set calculations"
                " (default is 100")
        ;

        po::store(po::command_line_parser(argc, argv)
            .options(desc_cmdline).run(), vm);
        po::notify(vm);

        // print help screen
        if (vm.count("help")) {
            std::cout << desc_cmdline;
            return false;
        }
    }
    catch (std::exception const& e) {
        std::cerr << "mandelbrot: exception caught: " << e.what() << std::endl;
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
inline void 
split_ip_address(std::string const& v, std::string& addr, boost::uint16_t& port)
{
    std::string::size_type p = v.find_first_of(":");
    try {
        if (p != std::string::npos) {
            addr = v.substr(0, p);
            port = boost::lexical_cast<boost::uint16_t>(v.substr(p+1));
        }
        else {
            addr = v;
        }
    }
    catch (boost::bad_lexical_cast const& /*e*/) {
        std::cerr << "mandelbrot: illegal port number given: " << v.substr(p+1) << std::endl;
        std::cerr << "           using default value instead: " << port << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////
// helper class for AGAS server initialization
class agas_server_helper
{
public:
    agas_server_helper(std::string host, boost::uint16_t port)
      : agas_pool_(), agas_(agas_pool_, host, port)
    {
        agas_.run(false);
    }

private:
    hpx::util::io_service_pool agas_pool_; 
    hpx::naming::resolver_server agas_;
};

///////////////////////////////////////////////////////////////////////////////
// CMandelbrotApp

BEGIN_MESSAGE_MAP(CMandelbrotApp, CWinApp)
    ON_COMMAND(ID_HELP, &CWinApp::OnHelp)
END_MESSAGE_MAP()


// CMandelbrotApp construction

CMandelbrotApp::CMandelbrotApp()
{
    // TODO: add construction code here,
    // Place all significant initialization in InitInstance
}

// The one and only CMandelbrotApp object

CMandelbrotApp theApp;

// CMandelbrotApp initialization

///////////////////////////////////////////////////////////////////////////////
// this is the runtime type we use in this application
typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> runtime_type;

BOOL CMandelbrotApp::InitInstance()
{
    // InitCommonControlsEx() is required on Windows XP if an application
    // manifest specifies use of ComCtl32.dll version 6 or later to enable
    // visual styles.  Otherwise, any window creation will fail.
    INITCOMMONCONTROLSEX InitCtrls;
    InitCtrls.dwSize = sizeof(InitCtrls);
    // Set this to include all the common control classes you want to use
    // in your application.
    InitCtrls.dwICC = ICC_WIN95_CLASSES;
    InitCommonControlsEx(&InitCtrls);

    CWinApp::InitInstance();

    // Standard initialization
    // If you are not using these features and wish to reduce the size
    // of your final executable, you should remove from the following
    // the specific initialization routines you do not need
    // Change the registry key under which our settings are stored
    // TODO: You should modify this string to be something appropriate
    // such as the name of your company or organization
    SetRegistryKey(_T("ParalleX"));

    // analyze the command line
    po::variables_map vm;
    if (!parse_commandline(__argc, __argv, vm))
        return FALSE;

    std::string hpx_host("localhost"), agas_host;
    boost::uint16_t hpx_port = HPX_PORT, agas_port = 0;
    int num_threads = 1;
    int size_x = 20;
    int size_y = 20;
    int iterations = 100;

    // extract IP address/port arguments
    if (vm.count("agas")) 
        split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

    if (vm.count("hpx")) 
        split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

    if (vm.count("threads"))
        num_threads = vm["threads"].as<int>();

    if (vm.count("sizex"))
        size_x = vm["sizex"].as<int>();
    if (vm.count("sizey"))
        size_y = vm["sizey"].as<int>();
    if (vm.count("iterations"))
        iterations = vm["iterations"].as<int>();

    // initialize and run the AGAS service, if appropriate
    std::auto_ptr<agas_server_helper> agas_server;
    if (vm.count("run_agas_server"))    // run the AGAS server instance here
        agas_server.reset(new agas_server_helper(agas_host, agas_port));

    // initialize and start the HPX runtime
    runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, hpx::runtime::console);

    // show dialog
    CMandelbrotDlg dlg (rt);
    m_pMainWnd = &dlg;

    // register error routine and start runtime
    boost::signals::scoped_connection conn;
    rt.register_error_sink(
        boost::bind(&CMandelbrotDlg::error_sink, &dlg, _1, _2), conn);
    rt.start(num_threads);

    INT_PTR nResponse = dlg.DoModal();
    if (nResponse == IDOK)
    {
        // TODO: Place code here to handle when the dialog is
        //  dismissed with OK
    }
    else if (nResponse == IDCANCEL)
    {
        // TODO: Place code here to handle when the dialog is
        //  dismissed with Cancel
    }

    // initiate shutdown of the runtime systems on all localities
    hpx::components::stubs::runtime_support::shutdown_all();

    // wait for everything to finish
    rt.wait();
    rt.stop();

    // Since the dialog has been closed, return FALSE so that we exit the
    //  application, rather than start the application's message pump.
    return FALSE;
}

