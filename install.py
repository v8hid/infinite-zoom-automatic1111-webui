import launch

if not launch.is_installed("imageio"):
    if not launch.is_installed("imageio.plugins.ffmpeg"):
        launch.run_pip("install imageio[ffmpeg]", "requirements for Infinite-Zoom")
