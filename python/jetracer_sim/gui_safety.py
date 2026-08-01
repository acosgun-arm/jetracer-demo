"""Safety policy for viewers that may touch the native desktop session."""

from __future__ import annotations


class UnsafeGuiRequestError(ValueError):
    """Raised before a tool attempts an unsafe native GUI operation."""


def validate_gui_request(
    *,
    headless: bool,
    viewer: str,
    open_browser: bool,
    allow_native_gui: bool,
) -> None:
    """Reject GUI combinations that are unsafe for background execution.

    OpenCV HighGUI can abort the whole process while macOS is locked, before
    Python has an opportunity to catch an exception. Native windows therefore
    require a separate acknowledgement; browser launching is also opt-in.
    """

    if viewer not in {"browser", "opencv"}:
        raise UnsafeGuiRequestError(f"unknown viewer {viewer!r}")
    if headless and open_browser:
        raise UnsafeGuiRequestError(
            "--headless cannot be combined with --open-browser"
        )
    if open_browser and viewer != "browser":
        raise UnsafeGuiRequestError(
            "--open-browser requires --viewer browser"
        )
    if not headless and viewer == "opencv" and not allow_native_gui:
        raise UnsafeGuiRequestError(
            "native OpenCV windows are disabled by default because macOS may "
            "abort while locked; use --viewer browser, --headless, or "
            "explicitly acknowledge the risk with --allow-native-gui"
        )
