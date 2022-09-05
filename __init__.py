bl_info = {
    # required
    'name': 'Blender Bioluminescence Imaging',
    'blender': (2, 93, 0),
    'category': 'Object',
    # optional
    'version': (1, 0, 0),
    'author': 'Avi Balsam',
    'description': 'Blender addon for practical synthetic bioluminescence data generation',
    'link': 'https://github.com/avbalsam/bbi-kinetics',
}

"""
File: __init__.py

Author 1: A. Balsam
Author 2: V. Kalchenko
Date: Summer 2022
Project: ISSI Weizmann Institute 2022

Summary:
    This blender addon is an effective tool for synthetic BLI data generation.
    It creates any number of light emitting blob with custom kinetics, and
    sets the light intensity of the blobs at each frame based on custom user-submitted
    parameters.
"""
import time
import textwrap
import math

import bpy
from bpy.utils import resource_path
from pathlib import Path

import random


def pharma_func(frame, num_frames, ka=0.1, ke=0.03):
    """
    The magic formula of pharmacokinetics in case of oral administration. Uses framedata, absorption rate and
    elimination rate to to find relative intensity of pharma function at selected frame.

    Args:
        frame: Current frame
        num_frames: Total number of frames
        ka: Absorption rate
        ke: Elimination rate

    Returns:
        Relative intensity value for selected frame
    """
    x = frame / num_frames * 100
    # The magic formula of pharmacokinetics in case of oral administration
    y = math.exp(x * (-ke)) - math.exp(x * (-ka))

    return y


def set_up_scene():
    """
    Sets up any blender file so that it can be used by the rest of the program. Imports blob and mouse objects
    from BLENDER_FILE_IMPORT_PATH.

    Returns:
        None
    """
    global BLENDER_FILE_IMPORT_PATH

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.data.scenes["Scene"].render.engine = "CYCLES"

    # Set background color to black
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = [0.000000, 0.000000, 0.000000,
                                                                                      1.000000]

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


def process_props(context, props: list, parent) -> None:
    """
    Updates Blender UI based on props passed.

    Args:
        context: Current scene context
        props (list): List of tuples to add to "parent" as props
        parent: Instance of a subclass of bpy.types.Panel

    Returns:
        None
    """
    for (prop_name, text) in props:
        if prop_name == 'label':
            # Create a multiline label with text of prop
            chars = int(context.region.width / 15)  # 7 pix on 1 character
            wrapper = textwrap.TextWrapper(width=chars)
            text_lines = wrapper.wrap(text=text)
            if len(text_lines) == 0:
                parent.label(text="")
            else:
                for text_line in text_lines:
                    parent.label(text=text_line)
        else:
            row = parent.row()
            row.prop(context.scene, prop_name)


def scale_object(obj, scaler_x: float, scaler_y: float, scaler_z: float):
    """
    Scales "obj" (a blender object) using the bpy python module

    Args:
        obj: Blender object to scale
        scaler_x (float): Scale factor in the X direction
        scaler_y (float): Scale factor in the Y direction
        scaler_z (float): Scale factor in the Z direction

    Returns:
        None
    """
    obj.select_set(True)
    bpy.ops.transform.resize(value=(scaler_x, scaler_y, scaler_z))
    obj.select_set(False)


class BlobGeneratorPanel(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_object_renamer'
    bl_label = 'Bioluminescence Blender'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        """
        Creates a Blender UI panel with various options.

        Args:
            context: Current scene context

        Returns:
            None
        """
        col = self.layout.column()

        process_props(context, ADD_BLOB_PROPS, col)
        col.operator('opr.object_add_blob_operator', text='Add Blob')

        col.label(text="")

        process_props(context, RENDER_SCENE_PROPS, col)
        col.operator('opr.object_render_scene_operator', text='Render Scene')


class BlobGeneratorOperator(bpy.types.Operator):
    bl_idname = 'opr.object_add_blob_operator'
    bl_label = 'Blob Adder'

    def execute(self, context):
        """
        Defines a button in the blender UI which, when pressed, adds a blob and a mouse to the scene
        (unless there is already a mouse, in which case it just adds a blob).

        Args:
            context: Current scene context

        Returns:
            None
        """
        global IS_SCENE_CONFIGURED
        global DATA_GEN
        global BLOB_MATERIAL
        global MOUSE_MATERIAL
        global BLENDER_FILE_IMPORT_PATH

        if context.scene.import_path != '':
            BLENDER_FILE_IMPORT_PATH = context.scene.import_path

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
            x = 0
            y = 0
            z = 0

            blob.set_position(x, y, z)

        x = 1
        y = 1
        z = 1
        blob.scale(x, y, z)

        if context.scene.randomize_blob_scale:
            blob.randomize_scale()

        blob.set_peak_intensity_value(context.scene.blob_highest_intensity)

        if context.scene.randomize_blob_intensity:
            blob.randomize_intensity()

        blob.fit_kinetics()

        absorption_rate = context.scene.absorption_rate
        elimination_rate = context.scene.elimination_rate

        if absorption_rate or elimination_rate != 0:
            try:
                blob.set_kinetic_func_params((absorption_rate, elimination_rate))
            except Exception as e:
                self.report({"ERROR", f"Could not convert terms into equation: {e}"})
                blob.delete_model()

        return {'FINISHED'}


class RenderSceneOperator(bpy.types.Operator):
    bl_idname = 'opr.object_render_scene_operator'
    bl_label = 'Scene Renderer'

    def execute(self, context):
        """
        Defines a Blender UI button which, when pressed, renders the scene by interpolating for each frame.

        Args:
            context: Current scene context

        Returns:
            None
        """
        global DATA_GEN

        if len(DATA_GEN) == 0:
            DATA_GEN.append(DataGen())

        DATA_GEN[0].set_num_frames(context.scene.num_frames)
        DATA_GEN[0].set_num_samples(context.scene.num_samples)
        DATA_GEN[0].set_output_path(context.scene.output_path)

        DATA_GEN[0].render_data(randomize_blobs=context.scene.randomize_blobs, panel=self)
        DATA_GEN.pop(0)

        return {'FINISHED'}


class Blob:
    def __init__(self,
                 kinetic_func=pharma_func,
                 kinetic_func_params=(0.1, 0.03),
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
        Args:
            kinetic_func: Function to find intensity of blob at a given frame
            kinetic_func_params: Params to pass to function
            blob: Blender object representing a light-emitting Blob
            num_frames: Number of frames in each render
            original_scaling_blob: Original scale of Blob (deprecated)
            min_scaling_blob: Minimum scale of Blob
            max_scaling_blob: Maximum scale of Blob
            max_shift_blob_x: Maximum position change on the x axis (in the positive and negative directions)
            max_shift_blob_y: Maximum position change on the y axis (in the positive and negative directions)
            min_intensity_blob: Minimum peak intensity of Blob
            max_intensity_blob: Maximum peak intensity of Blob
        """
        global BLOB_MATERIAL

        self.blob = blob
        self.blob.scale[0] = original_scaling_blob
        self.blob.scale[1] = original_scaling_blob
        self.blob.scale[2] = original_scaling_blob

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

        self.peak_intensity_value = None

        self.kinetic_func = kinetic_func
        self.kinetic_func_params = kinetic_func_params

        self.stretch_y = None
        self.shift = None

    def get_name(self):
        """
        Returns the name of this blob.

        Returns:
            The name of this blob
        """
        try:
            name = self.blob.name
            return name
        except ReferenceError:
            return False

    def delete_model(self):
        """
        Deletes the blender model connected to this blob and referenced to by self.blob

        Returns:
            None
        """
        self.blob.select_set(True)
        bpy.ops.object.delete()

    def randomize_intensity(self):
        """
        Randomly shifts and scales the kinetic function for this blob.

        Returns:
            This Blob object (for sequencing)
        """
        self.peak_intensity_value = random.uniform(self.peak_intensity_value * 0.90, self.peak_intensity_value * 1.10)

        self.fit_kinetics()
        return self

    def randomize_kinetic_func_params(self):
        """
        Randomizes each value in self.kinetic_func_params by a maximum of 10% in either direction (experimental)

        Returns:
            None
        """
        self.kinetic_func_params = (random.uniform(param * 0.9, param * 1.10) for param in self.kinetic_func_params)
        return self

    def set_kinetic_function(self, kinetic_func):
        """
        Sets this blob's kinetic function

        Args:
            kinetic_func: Function which takes the current frame, the number of frames, and any other parameters

        Returns:
            None
        """
        self.kinetic_func = kinetic_func

    def set_kinetic_func_params(self, params: tuple):
        """
        Sets the parameters for the kinetic function and runs the kinetic function to ensure the parameters are valid.

        Args:
            params (tuple): List of parameters (in order) which should be passed to the kinetic function

        Returns:
            None
        """
        self.kinetic_func_params = params
        self.kinetic_func(0, self.num_frames, *params)

    def set_peak_intensity_value(self, intensity):
        """
        Sets the peak intensity of this blob. Make sure to call self.fit_kinetics() to update kinetic function.

        Args:
            intensity: Maximum relative intensity of this blob

        Returns:
            None
        """
        self.peak_intensity_value = intensity

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

        Args:
            x_mouse: X-coordinate of mouse position
            y_mouse: Y-coordinate of mouse position
            z_mouse: Z-coordinate of mouse position
        """
        # placement of blob (lighting source) -> randomized -> relative to mouse placement
        x_pos = random.uniform(-1.0, 1.0) * self.max_shift_blob_x + x_mouse
        y_pos = random.uniform(-1.0, 1.0) * self.max_shift_blob_y + y_mouse
        z_pos = 0 + z_mouse
        self.set_position(x_pos, y_pos, z_pos)

        return self

    def scale(self, x, y, z):
        scale_object(self.blob, x, y, z)

    def randomize_scale(self):
        """
        Scale blob by a maximum of 10% in either direction

        Returns:
            This Blob object (for sequencing)
        """
        self.blob.scale[0] = random.uniform(self.blob.scale[0] * 0.90, self.blob.scale[0] * 1.10)
        self.blob.scale[1] = random.uniform(self.blob.scale[1] * 0.90, self.blob.scale[1] * 1.10)
        self.blob.scale[2] = random.uniform(self.blob.scale[2] * 0.90, self.blob.scale[2] * 1.10)

        return self

    def fit_kinetics(self):
        """
        Sets stretch_y, a coefficient which is applied to self.kinetic_func and allows for custom peak
        intensity values.
        """
        # Get the maximum value of this blob's kinetic function
        max_r = max([self.kinetic_func(frame, self.num_frames, *self.kinetic_func_params) for frame in range(self.num_frames)])

        self.stretch_y = self.peak_intensity_value / max_r

    def interpolate(self, frame):
        """
        Sets this blob's intensity based on growth and decay functions

        Args:
            frame (int): Frame to interpolate for
        Returns:
            None
        """
        self.light_emit_mesh_blob = self.blob.active_material.node_tree.nodes["Emission"].inputs[1]

        # For now, we use pharma_func as the default kinetic function. This can be changed to self.kinetic_func.
        intensity = pharma_func(frame, self.num_frames, *self.kinetic_func_params) * self.stretch_y
        self.light_emit_mesh_blob.default_value = intensity
        return intensity


class Mouse:
    def __init__(self,
                 num_frames=100,
                 min_init_intensity_mouse=0.05,
                 max_init_intensity_mouse=0.06,
                 min_end_intensity_mouse=0.045,
                 max_end_intensity_mouse=0.05,
                 min_intensity_blob=0.2,
                 max_intensity_blob=0.4,
                 original_scaling_mouse=0.420,
                 min_scaling_mouse=0.9,
                 max_scaling_mouse=1.1,
                 max_shift_mouse_x=2,
                 max_shift_mouse_y=2,
                 ):
        """
        Args:
            num_frames: Number of frames to render (full kinematics will always be rendered)
            min_init_intensity_mouse: Minimum intensity of initial light emitted from mouse
            max_init_intensity_mouse: Maximum intensity of initial light emitted from mouse
            min_end_intensity_mouse: Minimum intensity of final light emitted from mouse (should be less than initial intensity)
            max_end_intensity_mouse: Maximum intensity of final light emitted from mouse (should be less than initial intensity)
            min_intensity_blob: Minimum intensity of blobs at highest-intensity frame
            max_intensity_blob: Minimum intensity of blobs at highest-intensity frame
            original_scaling_mouse: Original scale of mouse (deprecated)
            min_scaling_mouse: Minimum final scale of mouse
            max_scaling_mouse: Maximum final scale of mouse
            max_shift_mouse_x: Maximum shift in position on the x-axis (minimum is 0)
            max_shift_mouse_y: Maximum shift in position on the y-axis (minimum is 0)
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

        # TODO Allow the user to input their own mouse file
        time.sleep(0.5)
        self.mouse = bpy.data.objects["mouse"]
        self.light_emit_mesh = self.mouse.active_material.node_tree.nodes["Principled BSDF"].inputs[20]
        self.mouse.select_set(False)

    def add_blob(self,
                 kinetic_func=pharma_func,
                 kinetic_func_params=(0.1, 0.03),
                 original_scaling_blob=4.20,
                 min_scaling_blob=0.7,
                 max_scaling_blob=1.1,
                 max_shift_blob_x=1.5,
                 max_shift_blob_y=5.0,
                 min_intensity_blob=0.2,
                 max_intensity_blob=0.4,
                 ):
        """
        Add a light-emitting blob to this mouse.

        Args:
            kinetic_func: Function to find intensity of blob at a given frame
            kinetic_func_params: Params to pass to function
            original_scaling_blob: Original scale of Blob (deprecated)
            min_scaling_blob: Minimum scale of Blob
            max_scaling_blob: Maximum scale of Blob
            max_shift_blob_x: Maximum position change on the x axis (in the positive and negative directions)
            max_shift_blob_y: Maximum position change on the y axis (in the positive and negative directions)
            min_intensity_blob: Minimum peak intensity of Blob
            max_intensity_blob: Maximum peak intensity of Blob
        """
        global BLENDER_FILE_IMPORT_PATH

        bpy.ops.wm.append(directory=f"{BLENDER_FILE_IMPORT_PATH}/Object/", filename="blob")

        new_blob = bpy.context.selected_objects[0]
        new_blob.select_set(False)

        blob_obj = Blob(
            kinetic_func=kinetic_func,
            kinetic_func_params=kinetic_func_params,
            blob=new_blob,
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

        Returns:
            None
        """
        b = 0
        while b < len(self.blobs):
            if self.blobs[b].get_name():
                b += 1
            else:
                self.blobs.pop(b)

    def randomize_blobs(self):
        """
        Randomizes the position, scale and kinetics of all blobs connected to this mouse.

        Returns:
            None
        """
        for blob in self.blobs:
            blob.randomize_position(self.x_mouse, self.y_mouse, self.z_mouse) \
                .randomize_scale().randomize_intensity()

    def randomize_scale(self):
        """
        Randomizes the scale of this mouse.

        Returns:
            None
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

        Returns:
            None
        """
        self.x_mouse = random.uniform(-1.0, 1.0) * self.max_shift_mouse_x
        self.y_mouse = random.uniform(-1.0, 1.0) * self.max_shift_mouse_y
        self.z_mouse = 0

        self.mouse.location = [self.x_mouse, self.y_mouse, self.z_mouse]

        return self

    def delete_model(self):
        """
        Deletes the Blender model associated with this Mouse object from the scene.

        Returns:
            None
        """
        self.mouse.select_set(True)
        bpy.ops.object.delete()
        self.mouse = None

    def interpolate(self, frame):
        """
        Sets the intensity of the mouse based on linear decay.

        Args:
            frame: Number frame to set intensity for

        Returns:
            None
        """
        # illumination mouse -> decrease linearly (approx)
        intensity_delta = self.init_intensity_mouse - self.end_intensity_mouse
        intensity = self.init_intensity_mouse - intensity_delta * (frame / self.num_frames)
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

        Args:
            img_size: Pixel size of images to generate
            num_frames: Number of frames to render in each sample
            num_samples: Number of samples to render
            output_path: Path to output files
        """
        self.img_size = img_size
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.output_path = output_path

        self.mouse = None

    def set_mouse(self,
                  min_init_intensity_mouse=0.05,
                  max_init_intensity_mouse=0.06,
                  min_end_intensity_mouse=0.045,
                  max_end_intensity_mouse=0.05,
                  original_scaling_mouse=0.420,
                  min_scaling_mouse=0.9,
                  max_scaling_mouse=1.1,
                  max_shift_mouse_x=2,
                  max_shift_mouse_y=2,
                  ):
        """
        Checks if a mouse object has been created for this datagen, and adds one if it has not.

        Args:
            min_init_intensity_mouse: Minimum intensity of initial light emitted from mouse
            max_init_intensity_mouse: Maximum intensity of initial light emitted from mouse
            min_end_intensity_mouse: Minimum intensity of final light emitted from mouse (should be less than initial intensity)
            max_end_intensity_mouse: Maximum intensity of final light emitted from mouse (should be less than initial intensity)
            original_scaling_mouse: Original scale of mouse (deprecated)
            min_scaling_mouse: Minimum final scale of mouse
            max_scaling_mouse: Maximum final scale of mouse
            max_shift_mouse_x: Maximum shift in position on the x-axis (minimum is 0)
            max_shift_mouse_y: Maximum shift in position on the y-axis (minimum is 0)

        Returns:
            None
        """
        if not self.mouse:
            self.mouse = Mouse(
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

    def render_data(self, randomize_blobs: bool, panel):
        """
        Render the scene while interpolating intensity based on kinetic function. Save render to self.output_path

        Args:
            randomize_blobs (bool): Whether to randomize the position and kinetics of the blobs after each render
            panel: A Blender UI object which can be used for reporting erros and debug information.

        Returns:
            None
        """
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
            if randomize_blobs:
                self.mouse.randomize_blobs()

            for frame in range(self.num_frames):
                bpy.context.scene.render.filepath = f"{self.output_path}/{str(sample)}/frame{str(frame)}"

                for blob in self.mouse.get_blobs():
                    panel.report({"INFO"}, f"Frame: {frame}, Blob Intensity: {blob.interpolate(frame)}")

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

ADDON = "BBI_Kinetics"
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
    ('label', "Add Blob: "),
    ('import_path', bpy.props.StringProperty(name="Import Path: ", default='')),
    ('label', "To use default blob and mouse models, leave blank."),
    ('label', ''),
    ('label', "Set blob kinetics:"),
    ('absorption_rate', bpy.props.FloatProperty(name="Absorption Rate", default=0)),
    ('elimination_rate', bpy.props.FloatProperty(name="Elimination Rate", default=0)),
    ('label', "To use default kinetics, set all of these fields to zero."),
    ('label', ''),
    ('randomize_blob_intensity', bpy.props.BoolProperty(name='Randomize Blob Intensity?', default=True)),
    ('blob_highest_intensity', bpy.props.FloatProperty(name='Peak Intensity Val', default=0.3)),
    ('randomize_blob_position', bpy.props.BoolProperty(name='Randomize Blob Position?', default=True)),
    ('randomize_blob_scale', bpy.props.BoolProperty(name='Randomize Blob Scale?', default=True)),
    ('label', ""),
]

RENDER_SCENE_PROPS = [
    ('label', ""),
    ('label', "Render Scene Menu:"),
    ('num_frames', bpy.props.IntProperty(name='Num Frames', default=100)),
    ('num_samples', bpy.props.IntProperty(name='Num Samples', default=1)),
    ('randomize_blobs', bpy.props.BoolProperty(name='Randomize Blobs?')),
    ('output_path', bpy.props.StringProperty(name='Output Path',
                                             default="/Users/avbalsam/Desktop/blender_animations"
                                                     "/blender_addon_new_kinetics")),
]

PROPS = ADD_BLOB_PROPS + RENDER_SCENE_PROPS


def register():
    for (prop_name, prop_value) in PROPS:
        if prop_name != 'label':
            setattr(bpy.types.Scene, prop_name, prop_value)

    for _class in CLASSES:
        bpy.utils.register_class(_class)


def unregister():
    for (prop_name, _) in PROPS:
        if prop_name != 'label':
            delattr(bpy.types.Scene, prop_name)

    for _class in CLASSES:
        bpy.utils.unregister_class(_class)


if __name__ == "__main__":
    register()
