"""Reference-view rendering (runs inside the Unreal Editor with a real RHI).

Reproduces the benchmark's reference capture: a temporary SceneCapture2D at
the view's location looking at its target, an unshadowed fill point light at
the camera and a top point light above the target, final-colour LDR output,
several captures to let streaming settle, then the render target is exported
as PNG. Nothing is saved to the level; the temporary actors are destroyed.

Pixel-exact equality with the benchmark's reference images is not expected
(GPU, driver and texture streaming differ); the pose, field of view,
lighting and resolution are identical.
"""

import os

import unreal

TEMP_PREFIX = "C4S_TempCapture"


def _world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def _cleanup(subsystem):
    for actor in list(subsystem.get_all_level_actors()):
        try:
            if str(actor.get_actor_label()).startswith(TEMP_PREFIX):
                subsystem.destroy_actor(actor)
        except Exception:
            pass


def _point_light(subsystem, location, intensity, radius):
    light = subsystem.spawn_actor_from_class(unreal.PointLight, location, unreal.Rotator())
    light.set_actor_label(TEMP_PREFIX + "_Light")
    component = light.get_editor_property("point_light_component")
    component.set_intensity(float(intensity))
    component.set_attenuation_radius(float(radius))
    component.set_editor_property("cast_shadows", False)
    return light


def render_view(capture, view, output_png):
    world = _world()
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    _cleanup(subsystem)
    commands = list(capture.get("console") or [])
    pool = view.get("streaming_pool_mb")
    if pool:
        commands.append("r.Streaming.PoolSize {}".format(int(pool)))
    for command in commands:
        unreal.SystemLibrary.execute_console_command(world, command)
    location = unreal.Vector(*[float(v) for v in view["location_cm"]])
    target = unreal.Vector(*[float(v) for v in view["target_cm"]])
    rotation = unreal.MathLibrary.find_look_at_rotation(location, target)
    camera = subsystem.spawn_actor_from_class(unreal.SceneCapture2D, location, rotation)
    camera.set_actor_label(TEMP_PREFIX + "_" + view["name"])
    lighting = view.get("lighting") or {}
    temporary = [camera]
    fill = capture.get("fill_light") or {}
    top = capture.get("top_light") or {}
    if fill.get("enabled", True):
        temporary.append(_point_light(subsystem, location, lighting.get("fill_intensity", 700.0),
                                      fill.get("attenuation_radius_cm", 3200.0)))
    if top.get("enabled", True):
        top_location = unreal.Vector(target.x, target.y, target.z + float(top.get("offset_z_cm", 320.0)))
        temporary.append(_point_light(subsystem, top_location, lighting.get("top_intensity", 450.0),
                                      top.get("attenuation_radius_cm", 2600.0)))
    component = camera.get_editor_property("capture_component2d")
    component.set_editor_property("capture_every_frame", False)
    component.set_editor_property("capture_on_movement", False)
    component.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    component.set_editor_property("fov_angle", float(view["fov_deg"]))
    try:
        post = component.get_editor_property("post_process_settings")
        post.set_editor_property("override_auto_exposure_bias", True)
        post.set_editor_property("auto_exposure_bias", float(lighting.get("exposure_bias", 0.0)))
        component.set_editor_property("post_process_settings", post)
        component.set_editor_property("post_process_blend_weight", 1.0)
    except Exception:
        pass
    target_rt = unreal.RenderingLibrary.create_render_target2d(
        world, int(capture["width"]), int(capture["height"]), unreal.TextureRenderTargetFormat.RTF_RGBA8)
    component.set_editor_property("texture_target", target_rt)
    for _ in range(int(capture.get("capture_count", 4))):
        component.capture_scene()
    output_png = os.path.abspath(output_png)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    unreal.RenderingLibrary.export_render_target(world, target_rt, os.path.dirname(output_png),
                                                 os.path.basename(output_png))
    unreal.RenderingLibrary.release_render_target2d(target_rt)
    for actor in temporary:
        subsystem.destroy_actor(actor)
    return {"view": view["name"], "png": output_png, "exists": os.path.isfile(output_png)}
