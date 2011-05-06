//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#include "afxwin.h"
#include "../mandelbrot_component/mandelbrot.hpp"

// CMandelbrotDlg dialog
class CMandelbrotDlg : public CDialog
{
// Construction
public:
    CMandelbrotDlg(hpx::runtime& rt, CWnd* pParent = NULL);     // standard constructor

// Dialog Data
    enum { IDD = IDD_MANDELBROT_GUI_DIALOG };

    protected:
    virtual void DoDataExchange(CDataExchange* pDX); // DDX/DDV support


// Implementation
protected:
    HICON m_hIcon;

    // Generated message map functions
    virtual BOOL OnInitDialog();
    afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
    afx_msg void OnPaint();
    afx_msg HCURSOR OnQueryDragIcon();
    afx_msg void OnBnClickedRender();
    afx_msg void OnEnChangeX();
    afx_msg void OnEnChangeY();

    DECLARE_MESSAGE_MAP()

public:
    CStatic m_bitmapframe;
    CButton m_render;
    CEdit m_x;
    CEdit m_y;
    CSpinButtonCtrl m_spinx;
    CSpinButtonCtrl m_spiny;

    void OnDoneRendering();
    void error_sink(boost::uint32_t, std::string const&);
    void mandelbrot_callback(hpx::lcos::counting_semaphore& sem,
        mandelbrot::result const& result);

    void InvalidateBitmap(BOOL erase = FALSE);

private:
    boost::mutex mtx_;
    CBitmap m_mandelbrot;
    bool created_bitmap;
    hpx::runtime& rt;
    boost::atomic<long> referesh_counter;
};
