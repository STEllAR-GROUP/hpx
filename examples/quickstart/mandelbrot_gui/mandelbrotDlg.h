// mandelbrotDlg.h : header file
//

#pragma once
#include "afxwin.h"


// CMandelbrotDlg dialog
class CMandelbrotDlg : public CDialog
{
// Construction
public:
    CMandelbrotDlg(hpx::runtime& rt, CWnd* pParent = NULL);     // standard constructor

// Dialog Data
    enum { IDD = IDD_MANDELBROT_GUI_DIALOG };

    protected:
    virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


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

private:
    CBitmap m_mandelbrot;
    bool created_bitmap;
    hpx::runtime& rt;
};
