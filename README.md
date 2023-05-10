## Installation Requirements

### Blender Installation

1. First, go to the official [Blender website](https://www.blender.org/download/) and download the version of Blender that matches your operating system (Windows, macOS, or Linux).
2. Once the download is complete, double-click the downloaded file to launch the installer.
3. Follow the installation wizard and accept the default settings. Note: if you are prompted to install additional software, such as Python, be sure to do so.
4. Once the installation is complete, launch Blender.

### bpycv Installation

1. Open a terminal or command prompt on your computer.
2. Navigate to the directory where Blender is installed. This will typically be something like `C:\Program Files\Blender Foundation\Blender`.
3. In the terminal or command prompt, run the following command to install `bpycv`:

   ```
   # For Windows user: ensure powershell has administrator permission

    # Ensure pip: equl to /blender-path/3.xx/python/bin/python3.10 -m ensurepip
    ./blender -b --python-expr "from subprocess import sys,call;call([sys.executable,'-m','ensurepip'])"
    # Update pip toolchain
    ./blender -b --python-expr "from subprocess import sys,call;call([sys.executable]+'-m pip install -U pip setuptools wheel'.split())"
    # pip install bpycv
    ./blender -b --python-expr "from subprocess import sys,call;call([sys.executable]+'-m pip install -U bpycv'.split())"
    # Check bpycv ready
    ./blender -b -E CYCLES --python-expr "import bpycv,cv2;d=bpycv.render_data();bpycv.tree(d);cv2.imwrite('/tmp/try_bpycv_vis(inst-rgb-depth).jpg', d.vis()[...,::-1])"
   ```
4. Once the installation is complete, you can test that `bpycv` is working properly by launching Blender and opening the Python Console. Type the following command:

   ```
   import bpycv
   ```

   If no errors are raised, then `bpycv` is installed correctly and ready to use.

## Running the code

1. Open Blender and make sure the default workspace is selected.
2. Click on the "Scripting" tab in the top header to open the scripting workspace.
3. In the left sidebar of the scripting workspace, navigate to the directory where your .py file is located. You can do this by clicking the folder icon in the top left corner of the sidebar and browsing to the appropriate folder.
4. Once you have found the .py file you want to open, click on it to select it.
5. The contents of the .py file will now be displayed in the main editor window of the scripting workspace.
6. Change the csv_name variable to match the name of the cvs you want to use.
7. Save the Blender file and ensure your csv file is located in the same directory
8. Click the run script button located at the top of the scripting window or press alt + p
9. Output images will begin to generate
