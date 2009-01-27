// mandelbrotDlg.cpp : implementation file
//

#include "stdafx.h"
#include "mandelbrot_gui.h"
#include "mandelbrotDlg.h"

#include "../mandelbrot_component/mandelbrot.hpp"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
    CAboutDlg();

// Dialog Data
    enum { IDD = IDD_ABOUTBOX };

    protected:
    virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
    DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
    CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()


// CMandelbrotDlg dialog

CMandelbrotDlg::CMandelbrotDlg(hpx::runtime& rt, CWnd* pParent /*=NULL*/)
    : CDialog(CMandelbrotDlg::IDD, pParent)
    , rt(rt)
    , created_bitmap(false)
{
    m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMandelbrotDlg::DoDataExchange(CDataExchange* pDX)
{
    CDialog::DoDataExchange(pDX);
    DDX_Control(pDX, IDC_PICTURE, m_bitmapframe);
    DDX_Control(pDX, ID_RENDER, m_render);
    DDX_Control(pDX, IDC_X, m_x);
    DDX_Control(pDX, IDC_Y, m_y);
    DDX_Control(pDX, IDC_SPINX, m_spinx);
    DDX_Control(pDX, IDC_SPINY, m_spiny);
}

BEGIN_MESSAGE_MAP(CMandelbrotDlg, CDialog)
    //{{AFX_MSG_MAP
    ON_WM_SYSCOMMAND()
    ON_WM_PAINT()
    ON_WM_QUERYDRAGICON()
    ON_BN_CLICKED(ID_RENDER, &CMandelbrotDlg::OnBnClickedRender)
    ON_EN_CHANGE(IDC_X, &CMandelbrotDlg::OnEnChangeX)
    ON_EN_CHANGE(IDC_Y, &CMandelbrotDlg::OnEnChangeY)
    //}}AFX_MSG_MAP
END_MESSAGE_MAP()


inline int get_value(CEdit& edit)
{
    CString text;
    edit.GetWindowText(text);
    return boost::lexical_cast<int>(text);
}

inline void set_value(CEdit& edit, int value)
{
    std::string text (boost::lexical_cast<std::string>(value));
    edit.SetWindowText(text.c_str());
}

// CMandelbrotDlg message handlers

BOOL CMandelbrotDlg::OnInitDialog()
{
    CDialog::OnInitDialog();

    // Add "About..." menu item to system menu.

    // IDM_ABOUTBOX must be in the system command range.
    ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
    ASSERT(IDM_ABOUTBOX < 0xF000);

    CMenu* pSysMenu = GetSystemMenu(FALSE);
    if (pSysMenu != NULL) {
        CString strAboutMenu;
        strAboutMenu.LoadString(IDS_ABOUTBOX);
        if (!strAboutMenu.IsEmpty()) {
            pSysMenu->AppendMenu(MF_SEPARATOR);
            pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
        }
    }

    // Set the icon for this dialog.  The framework does this automatically
    //  when the application's main window is not a dialog
    SetIcon(m_hIcon, TRUE);         // Set big icon
    SetIcon(m_hIcon, FALSE);        // Set small icon

    CRect rcframe;
    m_bitmapframe.GetClientRect(rcframe);
    m_spinx.SetRange(0, rcframe.Width());
    m_spiny.SetRange(0, rcframe.Height());

    set_value(m_x, (std::min)(100, rcframe.Width()));
    set_value(m_y, (std::min)(100, rcframe.Height()));

    return TRUE;  // return TRUE  unless you set the focus to a control
}

void CMandelbrotDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
    if ((nID & 0xFFF0) == IDM_ABOUTBOX) {
        CAboutDlg dlgAbout;
        dlgAbout.DoModal();
    }
    else {
        CDialog::OnSysCommand(nID, lParam);
    }
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CMandelbrotDlg::OnPaint()
{
    CPaintDC dc(this); // device context for painting

    if (IsIconic()) {
        SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

        // Center icon in client rectangle
        int cxIcon = GetSystemMetrics(SM_CXICON);
        int cyIcon = GetSystemMetrics(SM_CYICON);
        CRect rect;
        GetClientRect(&rect);
        int x = (rect.Width() - cxIcon + 1) / 2;
        int y = (rect.Height() - cyIcon + 1) / 2;

        // Draw the icon
        dc.DrawIcon(x, y, m_hIcon);
    }
    else {
        CDC memDC;
        memDC.CreateCompatibleDC(&dc);

        if (!created_bitmap) {
            m_mandelbrot.DeleteObject();    // re-create the bitmap, if needed
            m_mandelbrot.CreateCompatibleBitmap(&dc, get_value(m_x), get_value(m_y));

            // initialize bitmap to white
            CBitmap* oldbitmap = memDC.SelectObject((CBitmap*)&m_mandelbrot);
            memDC.PatBlt(0, 0, get_value(m_x), get_value(m_y), WHITENESS);
            memDC.SelectObject(oldbitmap);

            created_bitmap = true;
        }

        CRect rcframe;
        m_bitmapframe.GetWindowRect(rcframe);

        ScreenToClient(rcframe);
        InflateRect(rcframe, 
            -(rcframe.Width()-get_value(m_x)) / 2, 
            -(rcframe.Height()-get_value(m_y)) / 2);

        CBitmap* oldbitmap = memDC.SelectObject((CBitmap*)&m_mandelbrot);
        dc.BitBlt(rcframe.left, rcframe.top, 
            rcframe.left+rcframe.Width(), rcframe.top+rcframe.Height(), 
            &memDC, 0, 0, SRCCOPY);
        memDC.SelectObject(oldbitmap);
    }
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CMandelbrotDlg::OnQueryDragIcon()
{
    return static_cast<HCURSOR>(m_hIcon);
}

///////////////////////////////////////////////////////////////////////////////
void mandelbrot_callback(hpx::lcos::counting_semaphore& sem,
    int x, int y, int iterations)
{
    sem.signal();
}

void calculate_mandelbrot_set(int sizex, int sizey, CMandelbrotDlg* dlg)
{
    using namespace hpx;

    // get list of all known localities
    applier::applier& appl = applier::get_applier();

    // get prefixes of all remote localities (if any)
    std::vector<naming::id_type> prefixes;
    appl.get_remote_prefixes(prefixes);

    // execute the mandelbrot() functions remotely only, if any, otherwise
    // locally
    if (prefixes.empty())
        prefixes.push_back(appl.get_runtime_support_gid());

    std::size_t prefix_count = prefixes.size();

    hpx::lcos::counting_semaphore sem;
    typedef lcos::eager_future<mandelbrot_action> future_type;
    typedef lcos::future_callback<future_type> future_callback_type;

    // initialize the worker threads, one for each of the pixels
    std::vector<future_callback_type> futures;
    futures.reserve(sizex*sizey);   // preallocate vector

    double deltax = 1.0 / sizex;
    double deltay = 1.0 / sizey;

    for (int x = 0, i = 0; x < sizex; ++x) {
        for (int y = 0; y < sizey; ++y, ++i) {
            futures.push_back(future_callback_type(
                    future_type(prefixes[i % prefix_count], x * deltax, 
                        y * deltay, 100), 
                    boost::bind(mandelbrot_callback, boost::ref(sem), x, y, _1)
                ));
        }
    }

    // wait for the calculation to finish
    sem.wait(sizex*sizey);

    dlg->OnDoneRendering();
}

void CMandelbrotDlg::OnBnClickedRender()
{
    m_render.EnableWindow(FALSE);

    int sizex = get_value(m_x), sizey = get_value(m_y);
    hpx::applier::register_work(
        boost::bind(calculate_mandelbrot_set, sizex, sizey, this),
        "calculate_mandelbrot_set");
}

void CMandelbrotDlg::OnDoneRendering()
{
    m_render.EnableWindow(TRUE);
}

void CMandelbrotDlg::OnEnChangeX()
{
    if (IsWindow(m_x.m_hWnd))
        set_value(m_y, get_value(m_x));

    created_bitmap = false;   // re-create bitmap with new size
    Invalidate();
}

void CMandelbrotDlg::OnEnChangeY()
{
    created_bitmap = false;   // re-create bitmap with new size
    Invalidate();
}

void CMandelbrotDlg::error_sink(boost::uint32_t src, std::string const& msg)
{
    AfxMessageBox(msg.c_str());
}


