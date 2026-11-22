//! System-wide global hotkey listener (Press & Release detection for `Ctrl+Shift+Space`),
//! foreground window focus preservation, non-activating always-on-top overlay styling,
//! single-instance clipboard manager, and Win32 Unicode typing / `Ctrl+V` paste injection.
//!
//! Adapted from `crates/agente-computer/src/windows.rs` and `crates/agente-tui/src/clipboard.rs`.

use crate::config::DeliveryMode;
use anyhow::{bail, Result};
use parking_lot::Mutex;
use std::sync::{mpsc, OnceLock};
use std::time::{Duration, Instant};

static CLIPBOARD: OnceLock<Mutex<Option<arboard::Clipboard>>> = OnceLock::new();

pub fn copy_to_clipboard(text: &str) -> Result<()> {
    let lock = CLIPBOARD.get_or_init(|| Mutex::new(arboard::Clipboard::new().ok()));
    let mut guard = lock.lock();
    if guard.is_none() {
        *guard = arboard::Clipboard::new().ok();
    }
    if let Some(cb) = guard.as_mut() {
        cb.set_text(text.to_string())
            .map_err(|e| anyhow::anyhow!("Clipboard error: {e}"))?;
        Ok(())
    } else {
        bail!("System clipboard is unavailable")
    }
}

pub fn paste_from_clipboard() -> Option<String> {
    let lock = CLIPBOARD.get_or_init(|| Mutex::new(arboard::Clipboard::new().ok()));
    let mut guard = lock.lock();
    if guard.is_none() {
        *guard = arboard::Clipboard::new().ok();
    }
    guard.as_mut().and_then(|cb| cb.get_text().ok())
}

#[cfg(windows)]
pub fn capture_foreground_window() -> isize {
    use windows_sys::Win32::UI::WindowsAndMessaging::GetForegroundWindow;
    unsafe { GetForegroundWindow() as isize }
}

#[cfg(not(windows))]
pub fn capture_foreground_window() -> isize {
    0
}

#[cfg(windows)]
pub fn restore_foreground_window(hwnd: isize) {
    if hwnd == 0 {
        return;
    }
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, SetForegroundWindow};
    unsafe {
        if GetForegroundWindow() as isize == hwnd {
            return;
        }
        let _ = SetForegroundWindow(hwnd as HWND);
    }
}

#[cfg(not(windows))]
pub fn restore_foreground_window(_hwnd: isize) {}

/// Style the floating speech overlay so it sits on top of any input in the system
/// without stealing keyboard focus (`WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST`),
/// and clip its window region to the exact rounded pill shape so zero black corners leak.
#[cfg(windows)]
pub fn configure_overlay_window_no_activate(window: &winit::window::Window) {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::Graphics::Gdi::{CreateRoundRectRgn, SetWindowRgn};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetWindowLongPtrW, SetWindowLongPtrW, SetWindowPos, GWL_EXSTYLE, HWND_TOPMOST,
        SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST,
    };

    if let Ok(handle) = window.window_handle() {
        if let RawWindowHandle::Win32(win32) = handle.as_raw() {
            let hwnd = win32.hwnd.get() as HWND;
            let phys = window.inner_size();
            let pw = phys.width as i32;
            let ph = phys.height as i32;
            unsafe {
                let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
                let next = ex
                    | (WS_EX_NOACTIVATE as isize)
                    | (WS_EX_TOOLWINDOW as isize)
                    | (WS_EX_TOPMOST as isize);
                SetWindowLongPtrW(hwnd, GWL_EXSTYLE, next);
                if pw > 0 && ph > 0 {
                    let rgn = CreateRoundRectRgn(0, 0, pw + 1, ph + 1, ph, ph);
                    if !rgn.is_null() {
                        // Windows takes ownership of the region handle on success.
                        SetWindowRgn(hwnd, rgn, 1);
                    }
                }
                SetWindowPos(
                    hwnd,
                    HWND_TOPMOST,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
                );
            }
        }
    }
}

#[cfg(not(windows))]
pub fn configure_overlay_window_no_activate(_window: &winit::window::Window) {}

/// Show or hide the floating overlay window WITHOUT stealing keyboard focus from the active app.
#[cfg(windows)]
pub fn set_overlay_visible_no_activate(window: &winit::window::Window, visible: bool) {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::Graphics::Gdi::InvalidateRect;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        SetWindowPos, ShowWindow, HWND_TOPMOST, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE,
        SWP_SHOWWINDOW, SW_HIDE, SW_SHOWNOACTIVATE,
    };

    if let Ok(handle) = window.window_handle() {
        if let RawWindowHandle::Win32(win32) = handle.as_raw() {
            let hwnd = win32.hwnd.get() as HWND;
            unsafe {
                if visible {
                    configure_overlay_window_no_activate(window);
                    ShowWindow(hwnd, SW_SHOWNOACTIVATE);
                    SetWindowPos(
                        hwnd,
                        HWND_TOPMOST,
                        0,
                        0,
                        0,
                        0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
                    );
                    InvalidateRect(hwnd, std::ptr::null(), 0);
                } else {
                    ShowWindow(hwnd, SW_HIDE);
                }
            }
            return;
        }
    }
    window.set_visible(visible);
}

#[cfg(not(windows))]
pub fn set_overlay_visible_no_activate(window: &winit::window::Window, visible: bool) {
    window.set_visible(visible);
}

#[cfg(windows)]
fn wait_for_modifiers_released(max_wait: Duration) {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VK_CONTROL, VK_MENU, VK_SHIFT, VK_SPACE,
    };
    let is_down = |vk: u16| -> bool {
        unsafe { (GetAsyncKeyState(i32::from(vk)) as u16 & 0x8000) != 0 }
    };
    let start = Instant::now();
    while start.elapsed() < max_wait {
        if !is_down(VK_CONTROL) && !is_down(VK_SHIFT) && !is_down(VK_MENU) && !is_down(VK_SPACE) {
            break;
        }
        std::thread::sleep(Duration::from_millis(12));
    }
}

#[cfg(not(windows))]
fn wait_for_modifiers_released(_max_wait: Duration) {}

/// Deliver transcribed text according to the user's selected `DeliveryMode`,
/// preserving or restoring focus to `target_hwnd` before pasting/typing.
pub fn deliver_transcript(text: &str, mode: DeliveryMode, target_hwnd: Option<isize>) -> Result<String> {
    if text.trim().is_empty() {
        return Ok("Empty transcript".to_string());
    }
    copy_to_clipboard(text)?;

    match mode {
        DeliveryMode::ClipboardOnly => Ok("Copied to clipboard".to_string()),
        DeliveryMode::PasteAndClipboard => {
            if let Some(hwnd) = target_hwnd {
                restore_foreground_window(hwnd);
            }
            wait_for_modifiers_released(Duration::from_millis(180));
            std::thread::sleep(Duration::from_millis(40));
            send_ctrl_v()?;
            Ok("Copied & pasted (Ctrl+V) into active window".to_string())
        }
        DeliveryMode::TypeAndClipboard => {
            if let Some(hwnd) = target_hwnd {
                restore_foreground_window(hwnd);
            }
            wait_for_modifiers_released(Duration::from_millis(180));
            std::thread::sleep(Duration::from_millis(40));
            type_unicode_text(text)?;
            Ok("Copied & typed into active window".to_string())
        }
    }
}

#[cfg(windows)]
fn make_key_input(vk: u16, scan: u16, flags: u32) -> windows_sys::Win32::UI::Input::KeyboardAndMouse::INPUT {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT,
    };
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: vk,
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

#[cfg(windows)]
fn send_key_input(vk: u16, scan: u16, flags: u32) -> Result<()> {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{SendInput, INPUT};
    let input = make_key_input(vk, scan, flags);
    let sent = unsafe { SendInput(1, &input, std::mem::size_of::<INPUT>() as i32) };
    if sent != 1 {
        bail!("SendInput was blocked by an elevated window (text is still on your clipboard)");
    }
    Ok(())
}

#[cfg(windows)]
pub fn send_ctrl_v() -> Result<()> {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, MapVirtualKeyW, KEYEVENTF_KEYUP, MAPVK_VK_TO_VSC, VK_CONTROL,
        VK_LMENU, VK_LSHIFT, VK_RMENU, VK_RSHIFT,
    };
    const VK_V: u16 = 0x56;

    // Release any stuck Shift/Alt modifiers so Ctrl+V is never misinterpreted as Ctrl+Shift+V.
    for mod_vk in [VK_LSHIFT, VK_RSHIFT, VK_LMENU, VK_RMENU] {
        let down = unsafe { (GetAsyncKeyState(i32::from(mod_vk)) as u16 & 0x8000) != 0 };
        if down {
            let scan = unsafe { MapVirtualKeyW(u32::from(mod_vk), MAPVK_VK_TO_VSC) as u16 };
            let _ = send_key_input(mod_vk, scan, KEYEVENTF_KEYUP);
        }
    }

    let ctrl_scan = unsafe { MapVirtualKeyW(u32::from(VK_CONTROL), MAPVK_VK_TO_VSC) as u16 };
    let v_scan = unsafe { MapVirtualKeyW(u32::from(VK_V), MAPVK_VK_TO_VSC) as u16 };

    // Timed keystroke sequence so Word, Office, Electron, Chrome, and WinUI apps reliably receive Ctrl+V.
    send_key_input(VK_CONTROL, ctrl_scan, 0)?;
    std::thread::sleep(Duration::from_millis(15));
    let v_down = send_key_input(VK_V, v_scan, 0);
    std::thread::sleep(Duration::from_millis(15));
    let v_up = send_key_input(VK_V, v_scan, KEYEVENTF_KEYUP);
    std::thread::sleep(Duration::from_millis(15));
    let ctrl_up = send_key_input(VK_CONTROL, ctrl_scan, KEYEVENTF_KEYUP);
    v_down?;
    v_up?;
    ctrl_up?;
    Ok(())
}

#[cfg(not(windows))]
pub fn send_ctrl_v() -> Result<()> {
    Ok(())
}

#[cfg(windows)]
pub fn type_unicode_text(text: &str) -> Result<()> {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, VK_RETURN, VK_TAB,
    };
    for unit in text.encode_utf16() {
        let (vk, scan, flags) = if unit == 10 {
            (VK_RETURN, 0, 0)
        } else if unit == 9 {
            (VK_TAB, 0, 0)
        } else {
            (0, unit, KEYEVENTF_UNICODE)
        };
        send_key_input(vk, scan, flags)?;
        send_key_input(vk, scan, flags | KEYEVENTF_KEYUP)?;
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

#[cfg(not(windows))]
pub fn type_unicode_text(_text: &str) -> Result<()> {
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalHotkeyEvent {
    Pressed { target_hwnd: isize },
    Released { held_ms: u64 },
}

/// Spawn a background thread detecting both Press and Release of `Ctrl+Shift+Space`
/// globally across all applications.
pub fn spawn_global_hotkey_listener() -> mpsc::Receiver<GlobalHotkeyEvent> {
    let (tx, rx) = mpsc::channel();

    #[cfg(windows)]
    std::thread::spawn(move || {
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
            GetAsyncKeyState, VK_CONTROL, VK_SHIFT, VK_SPACE,
        };

        let is_down = |vk: u16| -> bool {
            unsafe { (GetAsyncKeyState(i32::from(vk)) as u16 & 0x8000) != 0 }
        };

        let mut was_pressed = false;
        let mut pressed_at = Instant::now();

        loop {
            let combo_down = is_down(VK_CONTROL) && is_down(VK_SHIFT) && is_down(VK_SPACE);
            if combo_down && !was_pressed {
                was_pressed = true;
                pressed_at = Instant::now();
                let target_hwnd = capture_foreground_window();
                if tx.send(GlobalHotkeyEvent::Pressed { target_hwnd }).is_err() {
                    break;
                }
            } else if !combo_down && was_pressed {
                was_pressed = false;
                let held_ms = pressed_at.elapsed().as_millis() as u64;
                if tx.send(GlobalHotkeyEvent::Released { held_ms }).is_err() {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(16));
        }
    });

    #[cfg(not(windows))]
    let _ = tx;

    rx
}
