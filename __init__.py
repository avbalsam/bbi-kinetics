import time

bl_info = {
    # required
    'name': 'Blender Bioluminescence Imaging',
    'blender': (2, 93, 0),
    'category': 'Object',
    # optional
    'version': (1, 0, 0),
    'author': 'Avi Balsam',
    'description': 'Blender addon for practical synthetic data generation',
}

"""
File: __init__.py

Author 1: A. Balsam
Author 2: J. Kuehne
Date: Summer 2022
Project: ISSI Weizmann Institute 2022
Label: Utility

Summary:
    This blender addon is an effective tool for synthetic BLI data generation.
    It creates any number of light emitting blob with custom kinetics, and
    sets the light intensity of the blobs at each frame based on custom user-submitted
    parameters. Additionally, the script creates a mouse which 

Requirements: The python library scipy has to be installed. As blender uses its
    own and not the system python installation, the package has to be installed
    specifically for this installation. If this causes difficulties, this can
    be achieved by the following console commands:
        (there is somewhere in the blender installation folder the python file,
        the commands have to be executed relative to that file)
        PATH_TO_BLENDER_INSTALLATION/.../python3.x -m ensurepip --upgrade
        PATH_TO_BLENDER_INSTALLATION/.../python3.x -m pip install scipy
"""
import textwrap
import logging

import bpy
from bpy.utils import resource_path
from pathlib import Path

import random
import math
from scipy import interpolate


def set_up_scene():
    global BLENDER_FILE_IMPORT_PATH

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.data.scenes["Scene"].render.engine = "CYCLES"

    # Set background color to black
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = [0.000000, 0.000000, 0.000000,
                                                                                      1.000000]
    # register()

    bpy.ops.wm.append(directory=f"{BLENDER_FILE_IMPORT_PATH}/Material/", filename="blob")
    bpy.ops.wm.append(directory=f"{BLENDER_FILE_IMPORT_PATH}/Material/", filename="mouse")

    bpy.ops.wm.append(directory=f"{BLENDER_FILE_IMPORT_PATH}/Object/", filename="mouse")
    bpy.ops.wm.append(directory=f"{BLENDER_FILE_IMPORT_PATH}/Object/", filename="blob")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.wm.append(directory=f"{BLENDER_FILE_IMPORT_PATH}/Camera/", filename="Camera.001")
    camera = bpy.context.selected_objects[0]
    camera.location = [0, 0, 75]
    bpy.context.scene.camera = camera

    # bpy.ops.wm.append(directory="/Users/avbalsam/Downloads/data_gen_discrete.blend/Light/", filename="Light.001")


def _label_multiline(context, text, parent):
    chars = int(context.region.width / 15)  # 7 pix on 1 character
    wrapper = textwrap.TextWrapper(width=chars)
    text_lines = wrapper.wrap(text=text)
    if len(text_lines) == 0:
        parent.label(text="")
    else:
        for text_line in text_lines:
            parent.label(text=text_line)


def process_props(context, props, parent):
    """Updates Blender UI based on props passed."""
    for (prop_name, prop) in props:
        if prop_name == 'label':
            _label_multiline(
                context=context,
                text=prop,
                parent=parent
            )
        else:
            row = parent.row()
            row.prop(context.scene, prop_name)


def scale_object(obj, scaler_x, scaler_y, scaler_z):
    obj.select_set(True)
    bpy.ops.transform.resize(value=(scaler_x, scaler_y, scaler_z))
    obj.select_set(False)


class BlobGeneratorPanel(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_object_renamer'
    bl_label = 'Bioluminescence Blender'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        col = self.layout.column()

        process_props(context, ADD_BLOB_PROPS, col)
        col.operator('opr.object_add_blob_operator', text='Add Blob')

        # process_props(context, ADD_MOUSE_PROPS, col)
        col.operator('opr.object_add_mouse_operator', text='Add Mouse')

        col.label(text="")

        process_props(context, RENDER_SCENE_PROPS, col)
        col.operator('opr.object_render_scene_operator', text='Render Scene')


class BlobGeneratorOperator(bpy.types.Operator):
    bl_idname = 'opr.object_add_blob_operator'
    bl_label = 'Blob Adder'

    def execute(self, context):
        global IS_SCENE_CONFIGURED
        global DATA_GEN
        global BLOB_MATERIAL
        global MOUSE_MATERIAL
        global BLENDER_FILE_IMPORT_PATH

        if not IS_SCENE_CONFIGURED:
            set_up_scene()
            IS_SCENE_CONFIGURED = True

        if len(DATA_GEN) == 0:
            DATA_GEN.append(DataGen())

        if DATA_GEN[0].mouse is None:
            bpy.ops.wm.append(directory=f"{BLENDER_FILE_IMPORT_PATH}/Object/", filename="mouse")
            mouse = bpy.context.selected_objects[0]
            mouse.select_set(False)
            DATA_GEN[0].set_mouse()

        blob = DATA_GEN[0].mouse.add_blob()
        BLOB_MATERIAL = bpy.data.materials["blob"]
        MOUSE_MATERIAL = bpy.data.materials["mouse"]

        if context.scene.randomize_blob_position:
            blob.randomize_position(DATA_GEN[0].mouse.get_x(), DATA_GEN[0].mouse.get_y(), DATA_GEN[0].mouse.get_z())
        else:
            x = 0  # context.scene.blob_position_x
            y = 0  # context.scene.blob_position_y
            z = 0  # context.scene.blob_position_z

            blob.set_position(x, y, z)

        if context.scene.randomize_blob_scale:
            blob.randomize_scale()
        else:
            x = 1  # context.scene.blob_scale_x
            y = 1  # context.scene.blob_scale_y
            z = 1  # context.scene.blob_scale_z

            blob.set_scale(x, y, z)

        if context.scene.randomize_blob_intensity:
            blob.randomize_spline()
        else:
            try:
                blob.set_spline_by_peak_intensity(highest_intensity_frame=context.scene.blob_highest_intensity_frame,
                                                  peak_intensity_blob=context.scene.blob_highest_intensity)
            except ValueError:
                self.report({"ERROR"}, "Invalid spline parameters. Try decreasing highest intensity frame.")
                blob.delete_model()
                return {'CANCELLED'}
        # self.report({"INFO"}, str(blob.spline))
        return {'FINISHED'}


class RenderSceneOperator(bpy.types.Operator):
    bl_idname = 'opr.object_render_scene_operator'
    bl_label = 'Scene Renderer'

    def execute(self, context):
        global DATA_GEN

        if len(DATA_GEN) == 0:
            DATA_GEN.append(DataGen())

        DATA_GEN[0].set_num_frames(context.scene.num_frames)
        DATA_GEN[0].set_num_samples(context.scene.num_samples)
        DATA_GEN[0].set_output_path(context.scene.output_path)

        if context.scene.randomize_blobs:
            self.report({"INFO"}, "Randomizing blobs...")
            DATA_GEN[0].mouse.randomize_blobs()
        DATA_GEN[0].render_data(self)
        DATA_GEN.pop(0)

        return {'FINISHED'}


class Blob:
    def __init__(self,
                 blob=None,
                 num_frames=100,

                 original_scaling_blob=4.20,
                 min_scaling_blob=0.7,
                 max_scaling_blob=1.1,
                 max_shift_blob_x=1.5,
                 max_shift_blob_y=5.0,
                 min_intensity_blob=0.2,
                 max_intensity_blob=0.4,
                 ):
        """
        :param blob: Blender object representing a light-emitting Blob
        :param num_frames: Number of frames in each render
        :param original_scaling_blob: Original scale of Blob
        :param min_scaling_blob: Minimum final scale of Blob
        :param max_scaling_blob: Maximum final scale of Blob
        :param max_shift_blob_x: Maximum position change on the x axis (in the positive and negative directions)
        :param max_shift_blob_y: Maximum position change on the y axis (in the positive and negative directions)
        :param min_intensity_blob: Minimum peak intensity of Blob
        :param max_intensity_blob: Maximum peak intensity of Blob
        """
        global BLOB_MATERIAL

        self.blob = blob

        BLOB_MATERIAL = bpy.data.materials["blob"]
        self.blob.active_material = BLOB_MATERIAL.copy()

        self.light_emit_mesh_blob = blob.active_material.node_tree.nodes["Emission"].inputs[1]

        self.max_shift_blob_x = max_shift_blob_x
        self.max_shift_blob_y = max_shift_blob_y
        self.original_scaling_blob = original_scaling_blob
        self.min_scaling_blob = min_scaling_blob
        self.max_scaling_blob = max_scaling_blob

        self.min_intensity_blob = min_intensity_blob
        self.max_intensity_blob = max_intensity_blob
        self.num_frames = num_frames

        self.spline = None

    def get_name(self):
        try:
            name = self.blob.name
            return name
        except ReferenceError:
            return False

    def set_spline_by_frame(self, x, y):
        """
        Sets the spline based on a list of points. This function can be called manually to circumvent default points of spline.

        :param x: List of points on the x-axis
        :param y: Corresponding points on the y-axis
        :return:
        """
        self.spline = interpolate.splrep(x, y, s=0)

    def delete_model(self):
        self.blob.select_set(True)
        bpy.ops.object.delete()

    def set_spline_by_peak_intensity(self, highest_intensity_frame, peak_intensity_blob):
        """
        Sets spline based on peak intensity. This function can be called manually to circumvent random generation of the spline.

        :param highest_intensity_frame: Number frame with highest intensity
        :param peak_intensity_blob: Maximum intensity of the blob

        :return: None, sets spline
        """
        x = [0,
             highest_intensity_frame / 100,
             highest_intensity_frame / 4,
             highest_intensity_frame / 2,
             highest_intensity_frame,
             (3 / 2) * highest_intensity_frame,
             (self.num_frames + highest_intensity_frame) / 2,
             (3 / 4) * self.num_frames + highest_intensity_frame / 4,
             self.num_frames - 1,
             self.num_frames]

        # y - values
        y = [0.0,
             0.0,
             peak_intensity_blob / 5,
             (4 / 5) * peak_intensity_blob,
             peak_intensity_blob,
             (7 / 10) * peak_intensity_blob,
             (3 / 10) * peak_intensity_blob,
             (1 / 10) * peak_intensity_blob,
             0.0,
             0.0]

        self.set_spline_by_frame(x, y)

    def randomize_spline(self):
        """
        Randomize the shape of the spline curve for this Blob.

        :return: This Blob object (for sequencing)
        """
        # frame with highest light intensity blob -> randomized at around 0.3 * num_frames +/- 0.05 * num_frames
        highest_intensity_frame = int(
            (self.num_frames * 0.3) + (random.uniform(-1.0, 1.0) * 0.05 * self.num_frames))

        peak_intensity_blob = random.uniform(self.min_intensity_blob, self.max_intensity_blob)
        # create spline for light emission of blob
        # x - values
        self.set_spline_by_peak_intensity(highest_intensity_frame, peak_intensity_blob)

        # For sequencing
        return self

    def get_blob(self):
        return self.blob

    def set_blob(self, blob):
        self.blob = blob

    def set_position(self, x, y, z):
        self.blob.location = [x, y, z]

    def randomize_position(self, x_mouse, y_mouse, z_mouse):
        """
        Randomize the position of this Blob within the mouse. Note: This method depends on mouse position,
        so make sure to set the position of the mouse before calling this method.

        :param x_mouse: X-coordinate of mouse position
        :param y_mouse: Y-coordinate of mouse position
        :param z_mouse: Z-coordinate of mouse position
        :return: This Blob object (for sequencing)
        """
        # placement of blob (lighting source) -> randomized -> relative to mouse placement
        x_pos = random.uniform(-1.0, 1.0) * self.max_shift_blob_x + x_mouse
        y_pos = random.uniform(-1.0, 1.0) * self.max_shift_blob_y + y_mouse
        z_pos = 0 + z_mouse
        self.set_position(x_pos, y_pos, z_pos)

        return self

    def set_scale(self, x, y, z):
        scale_object(self.blob, x, y, z)

    def randomize_scale(self):
        """
        Randomize scale of blob.

        :return: This Blob object (for sequencing)
        """
        # Set position and scale of blob
        self.blob.scale[0] = self.original_scaling_blob
        self.blob.scale[1] = self.original_scaling_blob
        self.blob.scale[2] = self.original_scaling_blob

        scaler_x_blob = random.uniform(self.min_scaling_blob, self.max_scaling_blob)
        scaler_y_blob = random.uniform(self.min_scaling_blob, self.max_scaling_blob)
        scaler_z_blob = random.uniform(self.min_scaling_blob, self.max_scaling_blob)

        self.set_scale(scaler_x_blob, scaler_y_blob, scaler_z_blob)

        return self

    def interpolate(self, frame):
        """
        Sets this blob's intensity based on previously generated spline function

        :param frame: Number frame to interpolate for
        :return: None, sets intensity of blob
        """
        self.light_emit_mesh_blob = self.blob.active_material.node_tree.nodes["Emission"].inputs[1]
        intensity = interpolate.splev(frame, self.spline)
        self.light_emit_mesh_blob.default_value = intensity
        return intensity


class Mouse:
    def __init__(self,
                 mouse,
                 light_source,
                 num_frames=100,
                 min_init_intensity_mouse=0.025,
                 max_init_intensity_mouse=0.045,
                 min_end_intensity_mouse=0.005,
                 max_end_intensity_mouse=0.015,
                 min_intensity_blob=0.2,
                 max_intensity_blob=0.4,
                 original_scaling_mouse=0.420,
                 min_scaling_mouse=0.9,
                 max_scaling_mouse=1.1,
                 max_shift_mouse_x=2,
                 max_shift_mouse_y=2,
                 ):
        """
        :param mouse: Blender object representing the mouse
        :param light_source: Blender object representing light source
        :param num_frames: Number of frames to render (full kinematics will always be rendered)
        :param min_init_intensity_mouse: Minimum intensity of initial light emitted from mouse
        :param max_init_intensity_mouse: Maximum intensity of initial light emitted from mouse
        :param min_end_intensity_mouse: Minimum intensity of final light emitted from mouse (should be less than initial intensity)
        :param max_end_intensity_mouse: Maximum intensity of final light emitted from mouse (should be less than initial intensity)
        :param min_intensity_blob: Minimum intensity of blobs at highest-intensity frame
        :param max_intensity_blob: Minimum intensity of blobs at highest-intensity frame
        :param original_scaling_mouse: Original scale of mouse
        :param min_scaling_mouse: Minimum final scale of mouse
        :param max_scaling_mouse: Maximum final scale of mouse
        :param max_shift_mouse_x: Maximum shift in position on the x axis
        :param max_shift_mouse_y: Maximum shift in position on the y axis
        """
        global MOUSE_MATERIAL
        global MOUSE_PATH

        self.num_frames = num_frames
        self.blobs = list()

        self.init_intensity_mouse = random.uniform(min_init_intensity_mouse, max_init_intensity_mouse)
        self.end_intensity_mouse = random.uniform(min_end_intensity_mouse, max_end_intensity_mouse)

        self.min_intensity_blob = min_intensity_blob
        self.max_intensity_blob = max_intensity_blob

        self.original_scaling_mouse = original_scaling_mouse
        self.min_scaling_mouse = min_scaling_mouse
        self.max_scaling_mouse = max_scaling_mouse
        self.max_shift_mouse_x = max_shift_mouse_x
        self.max_shift_mouse_y = max_shift_mouse_y

        self.x_mouse = 0
        self.y_mouse = 0
        self.z_mouse = 0

        # TODO Allow the user to input their own file
        time.sleep(0.5)
        self.mouse = bpy.data.objects["mouse"]
        self.light_emit_mesh = self.mouse.active_material.node_tree.nodes["Principled BSDF"].inputs[20]
        self.mouse.select_set(False)

    def add_blob(self,
                 original_scaling_blob=4.20,
                 min_scaling_blob=0.7,
                 max_scaling_blob=1.1,
                 max_shift_blob_x=1.5,
                 max_shift_blob_y=5.0,
                 min_intensity_blob=0.2,
                 max_intensity_blob=0.4,
                 ):
        """
        Add a blob to this mouse.

        :param blob_model_path: Path to blender model of blob (if left blank, default blob will be used).
        :param reporter: Blender UI panel for logging
        :param original_scaling_blob:
        :param min_scaling_blob:
        :param max_scaling_blob:
        :param max_shift_blob_x:
        :param max_shift_blob_y:
        :param min_intensity_blob:
        :param max_intensity_blob:

        :returns: None
        """
        global BLENDER_FILE_IMPORT_PATH

        bpy.ops.wm.append(directory=f"{BLENDER_FILE_IMPORT_PATH}/Object/", filename="blob")

        new_blob = bpy.context.selected_objects[0]
        new_blob.select_set(False)

        blob_obj = Blob(
            new_blob,
            min_intensity_blob=min_intensity_blob,
            max_intensity_blob=max_intensity_blob,
            original_scaling_blob=original_scaling_blob,
            min_scaling_blob=min_scaling_blob,
            max_scaling_blob=max_scaling_blob,
            max_shift_blob_x=max_shift_blob_x,
            max_shift_blob_y=max_shift_blob_y,
            num_frames=self.num_frames
        )
        self.blobs.append(blob_obj)

        return blob_obj

    def get_blobs(self):
        return self.blobs

    def validate_blobs(self):
        """
        Loops through this mouse's list of blobs and deletes any that are no longer present in the scene.

        :return: None
        """
        b = 0
        while b < len(self.blobs):
            if self.blobs[b].get_name():
                b += 1
            else:
                self.blobs.pop(b)

    def randomize_blobs(self):
        """
        Randomizes the position, scale and spline of all blobs connected to this mouse.

        :return: None
        """
        for blob in self.blobs:
            blob.randomize_position(self.x_mouse, self.y_mouse, self.z_mouse) \
                .randomize_scale()

    def randomize_scale(self):
        """
        Randomizes the scale of this mouse.

        :return: None
        """
        # After initializing values, set position and scale of mouse
        self.mouse.scale[0] = self.original_scaling_mouse
        self.mouse.scale[1] = self.original_scaling_mouse
        self.mouse.scale[2] = self.original_scaling_mouse

        scaler_x_mouse = random.uniform(self.min_scaling_mouse, self.max_scaling_mouse)
        scaler_y_mouse = random.uniform(self.min_scaling_mouse, self.max_scaling_mouse)
        scaler_z_mouse = random.uniform(self.min_scaling_mouse, self.max_scaling_mouse)

        # scale
        scale_object(self.mouse, scaler_x_mouse, scaler_y_mouse, scaler_z_mouse)

        return self

    def randomize_position(self):
        """
        Randomizes the position of this mouse.

        :return: None
        """
        self.x_mouse = random.uniform(-1.0, 1.0) * self.max_shift_mouse_x
        self.y_mouse = random.uniform(-1.0, 1.0) * self.max_shift_mouse_y
        self.z_mouse = 0

        self.mouse.location = [self.x_mouse, self.y_mouse, self.z_mouse]

        return self

    def delete_model(self):
        self.mouse.select_set(True)
        bpy.ops.object.delete()
        self.mouse = None

    def interpolate(self, frame):
        """
        Sets intensity of mouse based on exponential decay

        :param frame: Number frame to set intensity for
        :return: None, sets intensity of mouse
        """
        # illumination mouse -> decrease linearly (approx)
        intensity_delta = self.init_intensity_mouse - self.end_intensity_mouse
        intensity = self.init_intensity_mouse - intensity_delta * (frame / self.num_frames)
        # intensity = 0.030
        self.light_emit_mesh.default_value = intensity
        return intensity

    def get_x(self):
        return self.x_mouse

    def get_y(self):
        return self.y_mouse

    def get_z(self):
        return self.z_mouse


class DataGen:
    def __init__(self,
                 img_size=128,
                 num_frames=100,
                 num_samples=10,
                 output_path=None,
                 ):
        """
        :param img_size: Pixel size of images to generate
        :param num_frames: Number of frames to render in each sample
        :param num_samples: Number of samples to render
        """
        self.img_size = img_size
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.output_path = output_path

        self.mouse = None

    def set_mouse(self,

                  min_init_intensity_mouse=0.25,
                  max_init_intensity_mouse=0.35,
                  min_end_intensity_mouse=0.005,
                  max_end_intensity_mouse=0.15,
                  original_scaling_mouse=0.420,
                  min_scaling_mouse=0.9,
                  max_scaling_mouse=1.1,
                  max_shift_mouse_x=2,
                  max_shift_mouse_y=2,
                  ):
        if not self.mouse:
            self.mouse = Mouse(
                mouse=None,
                light_source=None,

                min_init_intensity_mouse=min_init_intensity_mouse,
                max_init_intensity_mouse=max_init_intensity_mouse,
                min_end_intensity_mouse=min_end_intensity_mouse,
                max_end_intensity_mouse=max_end_intensity_mouse,
                original_scaling_mouse=original_scaling_mouse,
                min_scaling_mouse=min_scaling_mouse,
                max_scaling_mouse=max_scaling_mouse,
                max_shift_mouse_x=max_shift_mouse_x,
                max_shift_mouse_y=max_shift_mouse_y,
            )

    def set_num_frames(self, num_frames):
        self.num_frames = num_frames

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

    def set_output_path(self, output_path):
        self.output_path = output_path

    def render_data(self, panel):
        bpy.data.scenes["Scene"].render.image_settings.file_format = 'TIFF'
        bpy.data.scenes["Scene"].render.use_overwrite = False
        bpy.data.scenes["Scene"].render.image_settings.color_mode = 'RGB'
        bpy.data.scenes["Scene"].render.resolution_x = self.img_size
        bpy.data.scenes["Scene"].render.resolution_y = self.img_size

        self.mouse.validate_blobs()

        for sample in range(self.num_samples):
            # Randomize position and scale of mouse and blobs. Make sure to call in this order -- position of blobs
            # is relative to position of mouse.
            self.mouse.randomize_position().randomize_scale()
            # self.mouse.randomize_blobs()

            for frame in range(self.num_frames):
                bpy.context.scene.render.filepath = f"{self.output_path}/{str(sample)}/frame{str(frame)}"

                for blob in self.mouse.get_blobs():
                    panel.report({"INFO"}, f"Blob Intensity: {blob.interpolate(frame)}")

                panel.report({"INFO"}, f"Mouse Intensity: {self.mouse.interpolate(frame)}")

                # save frame to file
                bpy.ops.render.render(write_still=True, use_viewport=True)

        for blob in self.mouse.get_blobs():
            blob.blob.select_set(True)
            bpy.ops.object.delete()

        self.mouse.delete_model()
        self.mouse = None


CLASSES = [
    BlobGeneratorPanel,
    BlobGeneratorOperator,
    RenderSceneOperator,
]

DATA_GEN = list()
# Globals
IS_SCENE_CONFIGURED = False

BLOB_MATERIAL = None
MOUSE_MATERIAL = None

ADDON = "BBI_Addon"
BLOB_MODEL = "blob.usdc"
MOUSE_MODEL = "mouse.usdc"
BLENDER_FILE = "data_gen_discrete.blend"

USER = Path(resource_path('USER'))

BLOB_PATH = USER / "scripts/addons" / ADDON / "models" / BLOB_MODEL
BLOB_PATH = str(BLOB_PATH)

MOUSE_PATH = USER / "scripts/addons" / ADDON / "models" / MOUSE_MODEL
MOUSE_PATH = str(MOUSE_PATH)

BLENDER_FILE_IMPORT_PATH = USER / "scripts/addons" / ADDON / "models" / BLENDER_FILE
BLENDER_FILE_IMPORT_PATH = str(BLENDER_FILE_IMPORT_PATH)

ADD_BLOB_PROPS = [
    ('label', "Blob Menu:"),
    ('label', "Path to .usdc file containg light emitting blob. To use default model, leave blank."),
    ('blob_model_path', bpy.props.StringProperty(name="Blob Path", default=BLOB_PATH)),
    ('randomize_blob_intensity', bpy.props.BoolProperty(name='Randomize Blob Intensity?', default=True)),
    ('blob_highest_intensity_frame', bpy.props.IntProperty(name='Peak Intensity Frame', default=30)),
    ('blob_highest_intensity', bpy.props.FloatProperty(name='Peak Intensity Val', default=0.3)),
    ('randomize_blob_position', bpy.props.BoolProperty(name='Randomize Blob Position?', default=True)),
    # ('blob_position_x', bpy.props.FloatProperty(name='Blob Pos X', default=0.0)),
    # ('blob_position_y', bpy.props.FloatProperty(name='Blob Pos Y', default=0.0)),
    # ('blob_position_z', bpy.props.FloatProperty(name='Blob Pos Z', default=0.0)),
    ('randomize_blob_scale', bpy.props.BoolProperty(name='Randomize Blob Scale?', default=True)),
    # ('blob_scale_x', bpy.props.FloatProperty(name='Blob Scale X', default=0.0)),
    # ('blob_scale_y', bpy.props.FloatProperty(name='Blob Scale Y', default=0.0)),
    # ('blob_scale_z', bpy.props.FloatProperty(name='Blob Scale Z', default=0.0)),
    ('label', ""),
]

ADD_MOUSE_PROPS = [
    ('label', ""),
    ('label', "Mouse Menu:"),
    ('randomize_mouse_position', bpy.props.BoolProperty(name='Randomize Mouse Position?', default=False)),
    ('randomize_mouse_scale', bpy.props.BoolProperty(name='Randomize Mouse Scale?', default=False)),
    ('label', "Path to .usdc file containing mouse (or other object). "
              "For default mouse model, leave blank."),
    ('mouse_model_path', bpy.props.StringProperty(name="Mouse Path", default=MOUSE_PATH)),
]

RENDER_SCENE_PROPS = [
    ('label', ""),
    ('label', "Render Scene Menu:"),
    ('num_frames', bpy.props.IntProperty(name='Num Frames', default=100)),
    ('num_samples', bpy.props.IntProperty(name='Num Samples', default=10)),
    ('randomize_blobs', bpy.props.BoolProperty(name='Randomize Blobs?')),
    ('output_path', bpy.props.StringProperty(name='Output Path',
                                             default="/Users/avbalsam/Desktop/blender_animations"
                                                     "/training_data_multiblob")),
]

PROPS = ADD_BLOB_PROPS + ADD_MOUSE_PROPS + RENDER_SCENE_PROPS


def register():
    for (prop_name, prop_value) in PROPS:
        if prop_name != 'label':
            setattr(bpy.types.Scene, prop_name, prop_value)

    for klass in CLASSES:
        bpy.utils.register_class(klass)


def unregister():
    for (prop_name, _) in PROPS:
        if prop_name != 'label':
            delattr(bpy.types.Scene, prop_name)

    for klass in CLASSES:
        bpy.utils.unregister_class(klass)


if __name__ == "__main__":
    register()


"""
Mouse light emission emission setting:
material.node_tree.nodes["Principled BDSF"].inputs[19].default_value = [1.000000, 1.000000, 1.000000, 1.000000]
material.node_tree.nodes["Principled BDSF"].inputs[20].default_value = 0.05



"""