# bbi-kinetics
This Blender addon is a quick and easy way to generate large quanities of simulated Bioluminescence Imaging data. To install, download this repository as a zip file. Next, simply add the zip file as an add-on to any Blender file of your choice. For more information on installing Blender add-ons, see https://docs.blender.org/manual/en/latest/editors/preferences/addons.html#installing-add-ons

After installing and activating the add-on, navigate to the "Misc" tab in the Blender 3D viewport. You should see the following UI:

Click "Add Blob", and the addon will automatically set up the scene with a model mouse and a light-emitting blob representing an area of luciferin concentration. If you would like to add another blob, simply click "Add Blob" again to add another light-emitting blob to the mouse you have generated.

![Add-On UI](https://github.com/avbalsam/bbi-kinetics/blob/master/images/ui.png)

To customize the light intensity pattern of a blob, edit the parameters "Highest Intensity Frame" and "Peak Intensity Value" BEFORE clicking "Add Blob". Once "Add Blob" is clicked, it is impossible to change the intensity patterns of the blob that was added.

![Add Blob Menu](https://github.com/avbalsam/bbi-kinetics/blob/master/images/add_blob.png)

After generating light emitting blobs with satisfactory kinetics, feel free to move or scale those blobs within the mouse just as you would any other blender object. Do not change the material assigned to each of the blobs.

Finally, click "Render Scene" to generate a frame sequence with .tiff format representing the simulated mouse with simulated blobs. But before you do, make sure "Output Path" is set correctly. If you would like to generate multiple samples, change the "Num Samples" parameter. To customize the number of frames in each frame sequence, change the "Num Frames" parameter. Finally, if you would like to randomize the positions, scales and kinetics of the blobs after each iteration, check the checkbox "Randomize Blobs".

![Render Scene Menu](https://github.com/avbalsam/bbi-kinetics/blob/master/images/render_scene.png)

For technical questions or feature requests, contact Avi Balsam: avbalsam@gmail.com

This project is the product of a summer of work at the Weizmann Institute ISSI with Dr. Vyacheslav Kalchenko. Contact him at a.kalchenko@weizmann.ac.il

![Sample Mouse](https://github.com/avbalsam/bbi-kinetics/blob/master/images/new_mouse_gen.gif)
