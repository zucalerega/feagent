"""Runs inside Isaac Sim / any Kit app:  isaac-sim.sh --exec /twin/open_in_kit.py   (open.sh does this)
Opens $TWIN_USD (default ~/feagent-viz/twin.usdc) once the app has started, looks through /World/Camera,
turns on RTX auto-exposure (the stage's UsdLux intensities are moderate) and loops the deformation timeline."""
import asyncio
import os

import omni.kit.app
import omni.timeline
import omni.usd

PATH = os.environ.get("TWIN_USD", os.path.expanduser("~/feagent-viz/twin.usdc"))


async def _frames(n):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


async def _open():
    await _frames(120)  # let the app finish its own startup (it may create a new empty stage)
    ctx = omni.usd.get_context()
    for attempt in range(3):
        await ctx.open_stage_async(PATH)
        await _frames(60)
        if (ctx.get_stage_url() or "").endswith(os.path.basename(PATH)):
            break
        print(f"[open_in_kit] stage was replaced during startup, reopening ({attempt + 1})")
    try:
        import carb.settings
        carb.settings.get_settings().set("/rtx/post/histogram/enabled", True)  # auto exposure
    except Exception as e:
        print(f"[open_in_kit] auto exposure not set: {e}")
    try:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = "/World/Camera"
    except Exception as e:
        print(f"[open_in_kit] could not switch to /World/Camera: {e}")
    tl = omni.timeline.get_timeline_interface()
    tl.set_looping(True)
    tl.play()
    print(f"[open_in_kit] opened {PATH}. Fields: select /World, Property panel > Variants > field. "
          "Deformation: timeline (t=0 undeformed, last frame = full scale).")

asyncio.ensure_future(_open())
