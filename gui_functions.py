import os
from tkinter import filedialog
import tkinter as tk


def get_file_path_gui(base_path="/DATA2/Data_CoCoS_HD/COCOS_ISM/"):
    root = tk.Tk()
    root.withdraw()
    root.geometry("800x600")
    rPadChars = 50 * " "
    root.ref_filename = filedialog.askopenfilename(
        initialdir=base_path,
        title="Select a red reference file (tif file) " + rPadChars,
        filetypes=(
            ("tif files", "*.tif"),
            ("tiff files", "*.tiff"),
            ("all files", "*.*"),
        ),
    )
    images_path = os.path.dirname(root.ref_filename)
    root.image_filename = filedialog.askopenfilename(
        initialdir=images_path,
        title="Select Spectral stack to analyse (tif file)",
        filetypes=(
            ("tif files", "*.tif"),
            ("tiff files", "*.tiff"),
            ("all files", "*.*"),
        ),
    )
    ref_red_name_im = os.path.split(root.ref_filename)[-1]
    im_rgb_name = os.path.split(root.image_filename)[-1]
    root.destroy()
    return images_path, ref_red_name_im, im_rgb_name


def get_selected_names_from_gui(names):
    """Create a GUI to select names and return the selected names."""

    # Inner function to handle the submission
    def submit_selection():
        nonlocal selected_names
        selected_names = [name for name, var in checkboxes.items() if var.get()]
        root.quit()
        root.destroy()  # Close the GUI

    # Initialize the tkinter window
    root = tk.Tk()
    root.title("Multi-Selection Checkbox Menu")

    # Dictionary to store IntVar objects for each checkbox
    checkboxes = {}
    selected_names = []

    # Create checkboxes for each name
    for name in names:
        var = tk.IntVar()
        chk = tk.Checkbutton(root, text=name, variable=var)
        chk.pack(anchor="w")
        checkboxes[name] = var

    # Add a submit button
    submit_button = tk.Button(root, text="Submit", command=submit_selection)
    submit_button.pack(pady=10)

    # Run the tkinter main loop
    root.mainloop()

    # Return the selected names
    return selected_names


def gui_get_fluorophores_to_analyse(fl_names):
    selected_fl = get_selected_names_from_gui(fl_names)
    selected_indices = []
    for selected_name in selected_fl:
        ind = fl_names.index(selected_name)
        selected_indices.append(ind)
    return selected_fl, selected_indices
