//! Native Windows System Tray (Activity Bar / Notification Area) icon & menu,
//! plus Enio brand RGBA icon rasterization via `resvg`.
//!
//! Closing the main window minimizes to the system tray so `Ctrl+Shift+Space`
//! system-wide dictation keeps running silently in the background.

use std::sync::mpsc;

const APP_ICON_SVG: &str = include_str!("../assets/icons/app-icon-dark.svg");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrayEvent {
    ShowMainWindow,
    ToggleDictation,
    Quit,
}

/// Rasterize Enio's framed app icon SVG into an RGBA8 pixel buffer of `(size, size)`.
pub fn rasterize_brand_rgba(size: u32) -> Option<Vec<u8>> {
    let opt = resvg::usvg::Options::default();
    let tree = resvg::usvg::Tree::from_str(APP_ICON_SVG, &opt).ok()?;
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size, size)?;
    let sx = size as f32 / tree.size().width();
    let sy = size as f32 / tree.size().height();
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(sx, sy),
        &mut pixmap.as_mut(),
    );
    // Un-premultiply alpha for winit::window::Icon / Win32 icon creation
    let mut out = pixmap.take();
    for px in out.chunks_exact_mut(4) {
        let a = px[3];
        if a > 0 && a < 255 {
            let af = f32::from(a) / 255.0;
            px[0] = (f32::from(px[0]) / af).min(255.0) as u8;
            px[1] = (f32::from(px[1]) / af).min(255.0) as u8;
            px[2] = (f32::from(px[2]) / af).min(255.0) as u8;
        }
    }
    Some(out)
}

pub fn create_winit_window_icon() -> Option<winit::window::Icon> {
    let rgba = rasterize_brand_rgba(32)?;
    winit::window::Icon::from_rgba(rgba, 32, 32).ok()
}

/// Spawn a native Windows System Tray (Activity Bar) icon with left-click restore
/// and right-click context menu (`Open Agente Dictate`, `Start / Stop Dictation`, `Quit`).
pub fn spawn_tray_icon() -> mpsc::Receiver<TrayEvent> {
    let (tx, rx) = mpsc::channel();

    #[cfg(windows)]
    std::thread::spawn(move || {
        windows_tray_loop(tx);
    });

    #[cfg(not(windows))]
    let _ = tx;

    rx
}

#[cfg(windows)]
fn windows_tray_loop(tx: mpsc::Sender<TrayEvent>) {
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
    use windows_sys::Win32::Graphics::Gdi::{
        CreateBitmap, CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, BITMAPINFO,
        BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS,
    };
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::Shell::{
        Shell_NotifyIconW, NIF_ICON, NIF_MESSAGE, NIF_TIP, NIM_ADD, NIM_DELETE, NOTIFYICONDATAW,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        AppendMenuW, CreateIconIndirect, CreatePopupMenu, CreateWindowExW, DefWindowProcW,
        DestroyIcon, DestroyMenu, DispatchMessageW, GetCursorPos, GetMessageW, PostQuitMessage,
        RegisterClassW, SetForegroundWindow, TrackPopupMenu, TranslateMessage, HICON, ICONINFO,
        MF_SEPARATOR, MF_STRING, MSG, TPM_BOTTOMALIGN, TPM_LEFTALIGN, TPM_RETURNCMD,
        WM_LBUTTONDBLCLK, WM_LBUTTONUP, WM_RBUTTONUP, WM_USER, WNDCLASSW,
    };

    const WM_TRAYICON: u32 = WM_USER + 42;
    const CMD_OPEN: u32 = 1001;
    const CMD_DICTATE: u32 = 1002;
    const CMD_QUIT: u32 = 1003;

    static TRAY_TX: OnceLock<mpsc::Sender<TrayEvent>> = OnceLock::new();
    let _ = TRAY_TX.set(tx);

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    unsafe fn create_hicon_from_rgba(rgba: &[u8], size: i32) -> HICON {
        let mut bi: BITMAPINFO = std::mem::zeroed();
        bi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bi.bmiHeader.biWidth = size;
        bi.bmiHeader.biHeight = -size; // top-down DIB
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;

        let hdc = CreateCompatibleDC(std::ptr::null_mut());
        let mut bits: *mut std::ffi::c_void = std::ptr::null_mut();
        let hbm_color = CreateDIBSection(hdc, &bi, DIB_RGB_COLORS, &mut bits, std::ptr::null_mut(), 0);
        if !hbm_color.is_null() && !bits.is_null() {
            let dst = std::slice::from_raw_parts_mut(bits as *mut u8, (size * size * 4) as usize);
            for (s, d) in rgba.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
                // BGRA premultiplied
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            }
        }
        let hbm_mask = CreateBitmap(size, size, 1, 1, std::ptr::null());
        let ii = ICONINFO {
            fIcon: 1,
            xHotspot: 0,
            yHotspot: 0,
            hbmMask: hbm_mask,
            hbmColor: hbm_color,
        };
        let hicon = CreateIconIndirect(&ii);
        if !hbm_color.is_null() {
            DeleteObject(hbm_color);
        }
        if !hbm_mask.is_null() {
            DeleteObject(hbm_mask);
        }
        if !hdc.is_null() {
            DeleteDC(hdc);
        }
        hicon
    }

    unsafe extern "system" fn wnd_proc(
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        if msg == WM_TRAYICON {
            let ev = (lparam & 0xffff) as u32;
            if ev == WM_LBUTTONUP || ev == WM_LBUTTONDBLCLK {
                if let Some(tx) = TRAY_TX.get() {
                    let _ = tx.send(TrayEvent::ShowMainWindow);
                }
                return 0;
            }
            if ev == WM_RBUTTONUP {
                let menu = CreatePopupMenu();
                let s_open = wide("Open Enio");
                let s_dict = wide("Start / Stop Dictation (Ctrl+Shift+Space)");
                let s_quit = wide("Quit");
                AppendMenuW(menu, MF_STRING, CMD_OPEN as usize, s_open.as_ptr());
                AppendMenuW(menu, MF_STRING, CMD_DICTATE as usize, s_dict.as_ptr());
                AppendMenuW(menu, MF_SEPARATOR, 0, std::ptr::null());
                AppendMenuW(menu, MF_STRING, CMD_QUIT as usize, s_quit.as_ptr());

                let mut pt: POINT = std::mem::zeroed();
                GetCursorPos(&mut pt);
                SetForegroundWindow(hwnd);
                let cmd = TrackPopupMenu(
                    menu,
                    TPM_LEFTALIGN | TPM_BOTTOMALIGN | TPM_RETURNCMD,
                    pt.x,
                    pt.y,
                    0,
                    hwnd,
                    std::ptr::null(),
                ) as u32;
                DestroyMenu(menu);

                if let Some(tx) = TRAY_TX.get() {
                    match cmd {
                        CMD_OPEN => {
                            let _ = tx.send(TrayEvent::ShowMainWindow);
                        }
                        CMD_DICTATE => {
                            let _ = tx.send(TrayEvent::ToggleDictation);
                        }
                        CMD_QUIT => {
                            let _ = tx.send(TrayEvent::Quit);
                            PostQuitMessage(0);
                        }
                        _ => {}
                    }
                }
                return 0;
            }
        }
        DefWindowProcW(hwnd, msg, wparam, lparam)
    }

    unsafe {
        let hinst = GetModuleHandleW(std::ptr::null());
        let class_name = wide("EnioTrayWindow");
        let wc = WNDCLASSW {
            lpfnWndProc: Some(wnd_proc),
            hInstance: hinst,
            lpszClassName: class_name.as_ptr(),
            ..std::mem::zeroed()
        };
        RegisterClassW(&wc);
        let hwnd = CreateWindowExW(
            0,
            class_name.as_ptr(),
            class_name.as_ptr(),
            0,
            0,
            0,
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            hinst,
            std::ptr::null(),
        );
        if hwnd.is_null() {
            return;
        }

        let hicon = rasterize_brand_rgba(32)
            .map(|rgba| create_hicon_from_rgba(&rgba, 32))
            .unwrap_or(std::ptr::null_mut());

        let mut nid: NOTIFYICONDATAW = std::mem::zeroed();
        nid.cbSize = std::mem::size_of::<NOTIFYICONDATAW>() as u32;
        nid.hWnd = hwnd;
        nid.uID = 1;
        nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
        nid.uCallbackMessage = WM_TRAYICON;
        nid.hIcon = hicon;
        let tip = wide("Enio — Ctrl+Shift+Space");
        let n = tip.len().min(nid.szTip.len());
        nid.szTip[..n].copy_from_slice(&tip[..n]);

        Shell_NotifyIconW(NIM_ADD, &nid);

        let mut msg: MSG = std::mem::zeroed();
        while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        Shell_NotifyIconW(NIM_DELETE, &nid);
        if !hicon.is_null() {
            DestroyIcon(hicon);
        }
    }
}
