import launch

if not launch.is_installed("imageio"):
    launch.run_pip("install imageio", "requirements 0 for Infinite-Zoom")
if not launch.is_installed("imageio_ffmpeg"):
    launch.run_pip("install imageio-ffmpeg", "requirements 1 for Infinite-Zoom")
