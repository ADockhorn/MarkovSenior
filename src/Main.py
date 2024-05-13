"""
Author: Mehmet Kayra Oguz and ChatGPT
Date: July 10, 2023
Description: This handles GUI
"""

import json
import multiprocessing
import random
import signal
import time
import tkinter as tk
from tkinter import Toplevel, filedialog
from PIL import Image, ImageTk
import psutil
from GrammarGym import gym_divcon_gui
from libraries.mario_level_visualizer import mlv
from markov_junior import Interpretor as intp
from multiprocessing import Pipe
import os
import GrammarStud as gs

root_path = os.path.abspath('..')
bin_path = os.path.join(root_path, 'bin')
output_path = os.path.join(bin_path, 'outputs')
resources_path = os.path.join(root_path, 'resources')
selected_file_name = None
gym_process = None
markovjun_process = None
training_aborted = True
argument_values = {}

# Info display
offspring_var = None
fitness_var = None
average_var = None
generation_var = None
load_percentage = None
index_var = None

# Images
image_label1 = None
image_label2 = None
canvasImg3 = None

def clear_directory(path):
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def clear_window():
    for widget in window.winfo_children():
        widget.destroy()

def selected_file_menu():
    global selected_file_name
    clear_window()
    selected_file_content = intp.parse_environment(os.path.join(resources_path, selected_file_name))

    mlv.world_string_to_png(selected_file_content,root_path, os.path.join(bin_path, 'level.png'))

    img = Image.open(os.path.join(bin_path, 'level.png'))
    photo = ImageTk.PhotoImage(img)

    # create a frame to hold the canvas and button
    content_frame = tk.Frame(window, padx=10, pady=10)
    content_frame.pack(expand=True)

    # create a frame for the canvas and label
    canvas_frame = tk.Frame(content_frame)
    canvas_frame.pack(pady=10)

    # create canvas and scrollbar
    canvas = tk.Canvas(canvas_frame, width=img.width, height=img.height)
    scrollbar_y = tk.Scrollbar(
        canvas_frame, orient='vertical', command=canvas.yview)
    scrollbar_x = tk.Scrollbar(
        canvas_frame, orient='horizontal', command=canvas.xview)

    # create a label to display the selected file name
    filename_label = tk.Label(canvas_frame, text="Selected Training Sample:\n" +
                              os.path.basename(selected_file_name), font=("Fixedsys", 14))
    filename_label.pack()

    # add image to canvas
    img_id = canvas.create_image(0, 0, anchor='center', image=photo)
    canvas.image = photo  # keep a reference!

    # update canvas so it knows about the image it just added
    canvas.update()

    # configure scrollable region to match image size
    canvas.configure(scrollregion=canvas.bbox(
        'all'), yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    # pack canvas and scrollbars
    scrollbar_y.pack(side='right', fill='y')
    scrollbar_x.pack(side='bottom', fill='x')
    canvas.pack()

    # create a frame for the buttons
    button_frame = tk.Frame(content_frame)
    button_frame.pack()

    # create the "Change File" button
    change_button = tk.Button(button_frame, text="Change",
                              command=select_file, width=15, height=2, font=("Fixedsys", 12))
    change_button.pack(side=tk.LEFT, pady=10, padx=5)

    # create the "Train Level" button
    train_button = tk.Button(button_frame, text="Continue",
                             command=train_menu, width=15, height=2, font=("Fixedsys", 12))
    train_button.pack(side=tk.LEFT, pady=10, padx=5)
    
    train_button = tk.Button(button_frame, text="Back to Main Menu",
                             command=main_menu, width=15, height=2, font=("Fixedsys", 12))
    train_button.pack(side=tk.BOTTOM, pady=10, padx=5)

    # center the content frame vertically
    content_frame.pack_configure(anchor='center')


def select_file():
    global selected_file_name
    selected_file_name = filedialog.askopenfilename(initialdir=resources_path, title="Select Sample")
    if selected_file_name:
        selected_file_menu()


def train_menu():
    window.resizable(True, True)
    window.geometry("900x700")
    window.resizable(False, False)
    
    global argument_values
    
    clear_window()
    global selected_file_name
    title_label = tk.Label(
        window, text="Training Parameters", font=("Fixedsys", 20))
    title_label.pack(pady=5)
    filename_label = tk.Label(window, text=os.path.basename(selected_file_name), font=("Fixedsys", 14))
    filename_label.pack()

    # Create a canvas for the settings
    settings_canvas = tk.Canvas(window)
    settings_canvas.pack(side='left', fill='both',
                         expand=True, anchor='center')

    # Add a scrollbar to the canvas
    scrollbar = tk.Scrollbar(window, command=settings_canvas.yview)
    scrollbar.pack(side='left', fill='y')

    # Configure the canvas
    settings_canvas.configure(yscrollcommand=scrollbar.set)
    settings_canvas.bind('<Configure>', lambda e: settings_canvas.configure(
        scrollregion=settings_canvas.bbox('all')))

    # Create a frame for the settings and add it to the canvas
    settings_frame = tk.Frame(settings_canvas, padx=20, pady=20)
    settings_canvas.create_window(
        (0, 0), window=settings_frame, anchor='center')

    # Create labels and entry fields for each argument
    arguments = [
        ("Name", "output"),
        ("Mode", ["divcon"]),
        ("Novelty Factor", 0.0),
        ("Coherency Factor", 1.0),
        ("Diversity Factor", 1),
        ("Tolerance Factor", 0.05),
        ("Population Size", 100),
        ("Mutation Rate", 0.01),
        ("Window Size", 4),
        ("Middle Window Size", 4),
        ("Micro Window Size", 2),
        ("Max Grammar Length", 10),
        ("Macro Window Size", 6),
        ("Shuffle Matches", False)
    ]

    argument_values = {}  # Store the argument values

    for row, (arg_name, arg_default) in enumerate(arguments):
        label = tk.Label(settings_frame, text=arg_name, font=("Fixedsys", 9))
        label.grid(row=row, column=0, sticky="w")

        if isinstance(arg_default, bool):
            var = tk.BooleanVar(value=arg_default)
            entry = tk.Checkbutton(settings_frame, variable=var, font=("Fixedsys", 9))
        elif isinstance(arg_default, list):
            var = tk.StringVar(value=arg_default)
            entry = tk.OptionMenu(settings_frame, var, *arg_default)
        else:
            var = tk.StringVar(value=arg_default)
            entry = tk.Entry(settings_frame, textvariable=var, font=("Fixedsys", 9))

        entry.grid(row=row, column=1, padx=10, pady=5, sticky="we")

        # Update the argument_values dictionary when entry value changes
        var.trace_add('write', lambda *args, var_name=arg_name, var=var: argument_values.update({var_name: var.get()}))

        # Initialize the argument_values dictionary
        argument_values[arg_name] = var.get()


    # Create a button frame to hold the buttons
    button_frame = tk.Frame(settings_frame)
    button_frame.grid(row=row+1, column=0, columnspan=2, pady=10)

    # Create a button to go back to the main screen
    back_button = tk.Button(button_frame, text="Back",
                            command=selected_file_menu, width=15, height=2, font=("Fixedsys", 12))
    back_button.pack(side=tk.LEFT, padx=5)

    # Create a button to train the level
    train_button = tk.Button(button_frame, text="Start Training", command=lambda: train_level(argument_values), width=15, height=2, font=("Fixedsys", 12))
    train_button.pack(side=tk.LEFT, padx=5)

    # Center the settings frame both vertically and horizontally
    settings_frame.pack(expand=True)
    settings_frame.pack_propagate(False)
    settings_frame.update_idletasks()
    window.update_idletasks()
    window_width = window.winfo_width()
    window_height = window.winfo_height()
    settings_width = settings_frame.winfo_width()
    settings_height = settings_frame.winfo_height()
    x_offset = (window_width - settings_width) // 2
    y_offset = (window_height - settings_height) // 2
    settings_canvas.configure(scrollregion=(
        0, 0, settings_width, settings_height))
    settings_canvas.xview_moveto(x_offset / settings_width)
    settings_canvas.yview_moveto(y_offset / settings_height)


def train_level(args: dict):
    global output_path, gym_process, training_aborted, selected_file_name
    # Create StringVars to store the values
    global offspring_var, fitness_var, average_var, generation_var, load_percentage, index_var, image_label1, image_label2, canvasImg3
    
    clear_directory(output_path)
    clear_window()

    window.resizable(True, True)
    window.geometry("950x950")
    window.resizable(False, False)

    title_label = tk.Label(window, text="Training", font=("Fixedsys", 14))
    title_label.grid(column=0, row=0, columnspan=2, pady=20)
    
    index_var = tk.StringVar(value="Processing the level")
    offspring_var = tk.StringVar()
    fitness_var = tk.StringVar()
    average_var = tk.StringVar()
    generation_var = tk.StringVar()
    load_percentage = tk.StringVar()

    # Create a frame for arguments
    arg_frame = tk.Frame(window)
    arg_frame.grid(column=0, row=1, padx=10, pady=10, columnspan=2)

    columns = 4
    horizontal_padding = 20  # Increase as needed
    small_padding = 3  # Smaller padding for the value
    for i, (arg, value) in enumerate(args.items()):
        row = i // columns
        col = i % columns
        arg_label = tk.Label(arg_frame, text=f"{arg}:", font=("Fixedsys", 8, "bold"))
        arg_label.grid(row=row, column=col*2, sticky='w', padx=(horizontal_padding, small_padding))

        value_label = tk.Label(arg_frame, text=f"{value}", font=("Fixedsys", 8))
        value_label.grid(row=row, column=col*2+1, sticky='w', padx=(small_padding, horizontal_padding))

    both_frames = tk.Frame(window)  
    both_frames.grid(column=0, row=2, padx=10, pady=10, sticky='n', columnspan=2) 

    window.grid_rowconfigure(2, weight=1)  # Add this line to make the 2nd row expandable

    stats_frame = tk.Frame(both_frames)
    stats_frame.grid(column=0, row=0, padx=40, sticky='w')

    # Create labels for the variable names
    index_label = tk.Label(stats_frame, text="Current Index: ", font=("Fixedsys", 10, "bold"))
    offspring_label = tk.Label(stats_frame, text="Offsprings generated: ", font=("Fixedsys", 10, "bold"))
    fitness_label = tk.Label(stats_frame, text="Highest Fitness: ", font=("Fixedsys", 10, "bold"))
    average_label = tk.Label(stats_frame, text="Average Fitness: ", font=("Fixedsys", 10, "bold"))
    generation_label = tk.Label(stats_frame, text="Current Generation: ", font=("Fixedsys", 10, "bold"))
    load_label = tk.Label(stats_frame, text="Level Processing: ", font=("Fixedsys", 10, "bold"))

    # Create labels to display the values
    index_value_label = tk.Label(stats_frame,textvariable=index_var, font=("Fixedsys", 8))
    offspring_value_label = tk.Label(stats_frame, textvariable=offspring_var, font=("Fixedsys", 8))
    fitness_value_label = tk.Label(stats_frame, textvariable=fitness_var, font=("Fixedsys", 8))
    average_value_label = tk.Label(stats_frame, textvariable=average_var, font=("Fixedsys", 8))
    generation_value_label = tk.Label(stats_frame, textvariable=generation_var, font=("Fixedsys", 8))
    load_value_label = tk.Label(stats_frame, textvariable=load_percentage, font=("Fixedsys", 8))
    
    # Arrange labels in a grid
    generation_label.grid(row=0, column=0, sticky='e')
    generation_value_label.grid(row=0, column=1)

    offspring_label.grid(row=1, column=0, sticky='e')
    offspring_value_label.grid(row=1, column=1)

    fitness_label.grid(row=2, column=0, sticky='e')
    fitness_value_label.grid(row=2, column=1)

    average_label.grid(row=3, column=0, sticky='e')
    average_value_label.grid(row=3, column=1)
    
    index_label.grid(row=4, column=0, sticky='e')
    index_value_label.grid(row=4, column=1) 

    load_label.grid(row=5, column=0, sticky='e')
    load_value_label.grid(row=5, column=1)

    images_frame = tk.Frame(both_frames)
    images_frame.grid(column=1, row=0, sticky='n') 

    # Image 1
    img1 = Image.open(os.path.join(bin_path, 'env.png'))
    photo1 = ImageTk.PhotoImage(img1)
    frame1 = tk.Frame(images_frame)
    frame1.pack(side='left')

    image_label1_text = tk.Label(frame1, text="Best Output", font=("Fixedsys", 12))
    image_label1_text.pack()

    image_label1 = tk.Label(frame1, image=photo1)
    image_label1.image = photo1
    image_label1.pack(pady=10)

    # Image 2
    img2 = Image.open(os.path.join(bin_path, 'env.png'))
    photo2 = ImageTk.PhotoImage(img2)
    frame2 = tk.Frame(images_frame)
    frame2.pack(side='left')

    image_label2_text = tk.Label(frame2, text="Current Sample", font=("Fixedsys", 12))
    image_label2_text.pack()

    image_label2 = tk.Label(frame2, image=photo2)
    image_label2.image = photo2
    image_label2.pack(pady=10)

    img3 = Image.open(os.path.join(bin_path, 'env.png'))
    photo3 = ImageTk.PhotoImage(img3)
    frame3 = tk.Frame(window)
    frame3.grid(column=0, row=4, columnspan=2, pady=10)

    # Image 3
    image_label3_text = tk.Label(frame3, text="Generated Level", font=("Fixedsys", 12))
    image_label3_text.pack()

    canvas_frame = tk.Frame(frame3, padx=10, pady=10)
    canvas_frame.pack(fill='both', expand=True)

    # Create a frame to hold the text display
    text_display_frame = tk.Frame(canvas_frame)
    text_display_frame.pack()

    canvasImg3 = tk.Canvas(canvas_frame, width=window.winfo_width(),height=img3.height, bg='grey')
    scrollbar_y = tk.Scrollbar(canvas_frame, orient='vertical', command=canvasImg3.yview)
    scrollbar_y.pack(side='right', fill='y')
    scrollbar_x = tk.Scrollbar(canvas_frame, orient='horizontal', command=canvasImg3.xview)
    scrollbar_x.pack(side='bottom', fill='x')
    canvasImg3.pack(side='left', fill='both', expand=True, pady=5)

    img_id = canvasImg3.create_image(0, 0, anchor='nw', image=photo3)
    canvasImg3.image = photo3  # keep a reference!
    canvasImg3.configure(scrollregion=canvasImg3.bbox('all'), yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    button_frame = tk.Frame(window)
    button_frame.grid(column=0, row=5, columnspan=2)
    button = tk.Button(button_frame, text="Abort Training",command=abort_training, width=15, height=2, font=("Fixedsys", 12))
    button.pack(pady=10)

    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=1)

    # Create communication pipes
    generation_conn_receiver, generation_conn_sender  = Pipe()
    createpop_conn_receiver, createpop_conn_sender = Pipe()
    gym_conn_receiver, gym_conn_sender = Pipe()
    print(selected_file_name)
    
    # Start gym_divcon_gui in a parallel process
    gym_process = multiprocessing.Process(target=gym_divcon_gui, args=(
        args["Name"], float(args["Novelty Factor"]), float(args["Coherency Factor"]), int(args["Population Size"]),
        float(args["Mutation Rate"]), int(args["Window Size"]), int(args["Middle Window Size"]), int(args["Micro Window Size"]),
        int(args["Max Grammar Length"]), float(args["Diversity Factor"]), int(args["Macro Window Size"]),float(args["Tolerance Factor"]),int(args["Shuffle Matches"]), generation_conn_sender,
        createpop_conn_sender, gym_conn_sender, output_path, os.path.join(resources_path, selected_file_name), os.path.join(resources_path, "env.txt")
    ))
    gym_process.start()
    training_aborted = False

    monitor_updates(generation_conn_receiver,createpop_conn_receiver, gym_conn_receiver)


def monitor_updates(generation_conn_receiver, createpop_conn_receiver, gym_conn_receiver):
    global offspring_var, fitness_var, average_var, generation_var, load_percentage, image_label1, image_label2, image_label3

    training_ended = False
    
    if not training_aborted:
        if generation_conn_receiver and createpop_conn_receiver:

            # Check if data received
            if createpop_conn_receiver.poll():
                try:
                    received_dict = createpop_conn_receiver.recv()
                    generation_var.set(received_dict["currGeneration"])
                    offspring_var.set("[" + received_dict["generatedOffsprings"] + "/" + received_dict["popSize"] + "]")
                    fitness_var.set(received_dict["highestFitness"]+"%")
                    average_var.set(received_dict["avgFitness"]+"%")
                    index_var.set(received_dict["index"])
                except EOFError:
                    pass

            if generation_conn_receiver.poll():
                try:
                    received_dict = generation_conn_receiver.recv()
                    if received_dict["currBestOut"]:
                        img1 = mlv.world_string_to_png(received_dict["currBestOut"], root_path,os.path.join(output_path, 'currBestOut.png'))
                    else:
                        img1 = Image.open(os.path.join(bin_path, 'env.png'))
                        
                    photo1 = ImageTk.PhotoImage(img1)
                    image_label1.config(image=photo1)
                    image_label1.image = photo1

                    if received_dict["currSample"]:
                        img2 = mlv.world_string_to_png(received_dict["currSample"], root_path,os.path.join(output_path, 'currSample.png'))
                    else:
                        img2 =Image.open(os.path.join(bin_path, 'env.png'))
                        
                    photo2 = ImageTk.PhotoImage(img2)
                    image_label2.config(image=photo2)
                    image_label2.image = photo2
                except EOFError:
                    pass

            if gym_conn_receiver:
                if gym_conn_receiver.poll():
                    try:
                        received_dict = gym_conn_receiver.recv()
                        load_percentage.set(str(received_dict["progress"]) + "%")
                        
                        if received_dict["currFinalOut"] :
                            img3 = mlv.world_string_to_png(received_dict["currFinalOut"], root_path, os.path.join(output_path, 'currFinalOut.png'))
                            photo3 = ImageTk.PhotoImage(img3)
                            canvasImg3.delete("all")  # Delete the current content of the canvas
                            img_id = canvasImg3.create_image(0, 0, anchor='nw', image=photo3)  # Create a new image on the canvas
                            canvasImg3.image = photo3  # Keep a reference to the image
                            canvasImg3.configure(scrollregion=canvasImg3.bbox('all'))  # Update the scrollregion
                        
                        training_ended = received_dict["end"]
                    except EOFError:
                        pass
        
        if not training_ended:
            # Schedule the next monitoring iteration after a delay
            window.after(50, lambda: monitor_updates(generation_conn_receiver, createpop_conn_receiver, gym_conn_receiver))
        else:  
            result_menu()

def result_menu():
    window.resizable(True, True)
    window.geometry("900x900")
    window.resizable(False, False)

    global selected_file_name, argument_values
    clear_window()

    img = Image.open(os.path.join(output_path, 'currFinalOut.png'))
    photo = ImageTk.PhotoImage(img)

    imgSample = Image.open(os.path.join(bin_path, 'level.png'))
    photoSample = ImageTk.PhotoImage(imgSample)

    padding = 20  # padding between the two images

    content_frame = tk.Frame(window, padx=10, pady=10)
    content_frame.pack(expand=True)

    # Creating a frame for arguments
    arg_frame = tk.Frame(content_frame)
    arg_frame.pack(pady=10)

    columns = 4
    horizontal_padding = 20  # Increase as needed
    small_padding = 3  # Smaller padding for the value
    for i, (arg, value) in enumerate(argument_values.items()):
        row = i // columns
        col = i % columns
        arg_label = tk.Label(arg_frame, text=f"{arg}:", font=("Fixedsys", 8, "bold"))
        arg_label.grid(row=row, column=col*2, sticky='w', padx=(horizontal_padding, small_padding))

        value_label = tk.Label(arg_frame, text=f"{value}", font=("Fixedsys", 8))
        value_label.grid(row=row, column=col*2+1, sticky='w', padx=(small_padding, horizontal_padding))

    canvas_frame = tk.Frame(content_frame)
    canvas_frame.pack(pady=10)

    canvas = tk.Canvas(canvas_frame, width=img.width, height=img.height + imgSample.height + padding)
    scrollbar_y = tk.Scrollbar(canvas_frame, orient='vertical', command=canvas.yview)
    scrollbar_x = tk.Scrollbar(canvas_frame, orient='horizontal', command=canvas.xview)

    filename_label = tk.Label(canvas_frame, text="Generated Level (top) vs Sample Level (bottom):", font=("Fixedsys", 14))
    filename_label.pack()

    img_id = canvas.create_image(0, 0, anchor='nw', image=photo)
    canvas.image = photo  # keep a reference to the image

    imgSample_id = canvas.create_image(0, img.height + padding, anchor='nw', image=photoSample)
    canvas.imageSample = photoSample  # keep a reference to the sample image

    canvas.update()

    canvas.configure(scrollregion=canvas.bbox('all'), yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    scrollbar_y.pack(side='right', fill='y')
    scrollbar_x.pack(side='bottom', fill='x')
    canvas.pack()

    button_frame = tk.Frame(content_frame)
    button_frame.pack()

    hint_label = tk.Label(button_frame, text="Generated content and grammars per sample are saved", font=("Fixedsys", 12,))
    hint_label.pack(side=tk.TOP, pady=5, anchor='center')

    change_button = tk.Button(button_frame, text="Back to Main Menu", command=main_menu, width=15, height=2, font=("Fixedsys", 12))
    change_button.pack(side=tk.LEFT, pady=10, padx=5, anchor='center')
    
    change_button = tk.Button(button_frame, text="Use Grammars", command=use_grammars, width=15, height=2, font=("Fixedsys", 12))
    change_button.pack(side=tk.RIGHT, pady=10, padx=5, anchor='center')

    content_frame.pack_configure(anchor='center')

def use_grammars():
    window.resizable(True, True)
    window.geometry("900x650")
    window.resizable(False, False)
    
    
    file_paths = []
    
    def file_dialog():
        nonlocal file_paths
        filepaths = filedialog.askopenfilenames(initialdir=output_path, filetypes=[('XML Files', '*.xml')])  # filter for XML files
        for filepath in filepaths:
            if filepath:
                try:
                    # create a pop-up window to show the file content
                    pop_up = tk.Toplevel(window)
                    with open(filepath, 'r') as file:
                        file_content = file.read()
                    text_widget = tk.Text(pop_up)
                    text_widget.insert(tk.END, file_content)
                    text_widget.pack(anchor='center')
                    button_frame = tk.Frame(pop_up)
                    button_frame.pack(anchor='center')

                    def continue_func():
                        nonlocal file_paths
                        file_paths.append(filepath)
                        pop_up.destroy()
                        show_file_paths()

                    def back_func():
                        pop_up.destroy()

                    continue_button = tk.Button(button_frame, text='Add Grammar', font=("Fixedsys", 12), command=continue_func)
                    back_button = tk.Button(button_frame, text='Back', font=("Fixedsys", 12),command=back_func)
                    continue_button.pack(side='left')
                    back_button.pack(side='right')

                except Exception as e:
                    tk.messagebox.showerror("Error", "Could not open file")

    def show_file_paths():
        nonlocal file_paths
        listbox.delete(0, tk.END)  # remove all current items
        for fp in file_paths:
            listbox.insert(tk.END, fp)
        update_buttons()


    def show_file_content(event):
        filepath = listbox.get(tk.ACTIVE)
        if filepath:
            try:
                pop_up = tk.Toplevel(window)
                with open(filepath, 'r') as file:
                    file_content = file.read()
                text_widget = tk.Text(pop_up)
                text_widget.insert(tk.END, file_content)
                text_widget.pack(anchor='center')
                tk.Button(pop_up, text='OK', font=("Fixedsys", 12),command=pop_up.destroy).pack(anchor='center')
            except Exception as e:
                tk.messagebox.showerror("Error", "Could not open file")

    def update_buttons():
        selected = listbox.curselection()
        move_up_button.config(state='disabled' if not selected or selected[0] == 0 else 'normal')
        move_down_button.config(state='disabled' if not selected or selected[0] == len(file_paths)-1 else 'normal')
        delete_button.config(state='disabled' if not selected else 'normal')

    def move_up():
        nonlocal file_paths
        selected = listbox.curselection()
        if selected and selected[0] > 0:
            index = selected[0]
            file_paths.insert(index-1, file_paths.pop(index))
            show_file_paths()
            listbox.select_set(index-1)

    def move_down():
        nonlocal file_paths
        selected = listbox.curselection()
        if selected and selected[0] < len(file_paths)-1:
            index = selected[0]
            file_paths.insert(index+1, file_paths.pop(index))
            show_file_paths()
            listbox.select_set(index+1)

    def delete():
        nonlocal file_paths
        selected = listbox.curselection()
        if selected:
            index = selected[0]
            file_paths.pop(index)
            show_file_paths()

    for widget in window.winfo_children():
        widget.destroy()

    title_label = tk.Label(window, text="Markov Junior Content Generation", font=("Fixedsys", 20))
    title_label.pack(pady=20)
    title_label = tk.Label(window, text="Selected grammars will be applied in the order they are placed from top to bottom", font=("Fixedsys", 12))
    title_label.pack(pady=10)
    
    main_frame = tk.Frame(window)
    main_frame.pack(expand=True)

    tk.Button(main_frame, text='Select File', font=("Fixedsys", 12),command=file_dialog).pack(anchor='center')

    listbox_frame = tk.Frame(main_frame)
    listbox_frame.pack(anchor='center')

    scrollbar_listbox_frame = tk.Frame(listbox_frame)
    scrollbar_listbox_frame.grid(row=0, column=0)

    scrollbar = tk.Scrollbar(scrollbar_listbox_frame)
    scrollbar.pack(side='right', fill='y')

    listbox = tk.Listbox(scrollbar_listbox_frame, height=20, width=80, yscrollcommand=scrollbar.set)
    listbox.pack(side='left')
    listbox.bind('<<ListboxSelect>>', lambda _: update_buttons())
    listbox.bind('<Double-1>', show_file_content)

    scrollbar.config(command=listbox.yview)

    button_frame = tk.Frame(listbox_frame)
    button_frame.grid(row=0, column=1)

    move_up_button = tk.Button(button_frame, text='Move Up', font=("Fixedsys", 12),command=move_up)
    move_up_button.pack()
    move_down_button = tk.Button(button_frame, text='Move Down', font=("Fixedsys", 12), command=move_down)
    move_down_button.pack()
    delete_button = tk.Button(button_frame, text='Delete',font=("Fixedsys", 12), command=delete)
    delete_button.pack()

    tk.Button(main_frame, text='Apply', font=("Fixedsys", 12), command=lambda:use_grammars_result(file_paths)).pack(anchor='center')  # replace lambda: None with your own command
    tk.Button(main_frame, text='Back', font=("Fixedsys", 12), command=main_menu).pack(anchor='center', pady=10)  # replace lambda: None with your own command
    update_buttons()

def use_grammars_result(grammarPaths: list):
    global markovjun_process
    img_path = os.path.join(output_path, 'result.png')

    def apply_grammars():
        if os.path.isfile(img_path):
            os.remove(img_path)
        gs.applyGrammars(root_path, img_path, grammarPaths)  # saves an image at img_path

    # Run apply_grammars in a separate process
    markovjun_process = multiprocessing.Process(target=apply_grammars)
    markovjun_process.start()

    def regenerate():
        display_image(os.path.join(bin_path,"env.png"))
        abort_markovjun()
        markovjun_process = multiprocessing.Process(target=apply_grammars)
        markovjun_process.start()
        train_button['state'] = tk.DISABLED  # Disable the Regenerate button
        select_grammars_button['state'] = tk.DISABLED
        
        check_image_existence()


    def display_image(path):
        if not os.path.isfile(path):
            print("Image file does not exist.")
        else:
            image = Image.open(path)
            photo = ImageTk.PhotoImage(image)

            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.config(width=image.width, height=image.height, scrollregion=canvas.bbox(tk.ALL))
            canvas.image = photo  # Keep a reference to prevent it from being garbage collected

    def check_image_existence():
        if os.path.isfile(img_path):
            train_button['state'] = tk.NORMAL  # Re-enable the Regenerate button
            select_grammars_button['state'] = tk.NORMAL
            display_image(img_path)
            os.remove(img_path)
        else:
            window.after(200, check_image_existence)  # Check again after 100ms

    def terminate():
        
        if markovjun_process:
            markovjun_process.kill()
        use_grammars()
    
    clear_window()

    for i in range(3):
        window.columnconfigure(i, weight=1, minsize=50)
        window.rowconfigure(i, weight=1, minsize=50)
    
    # Add title and explanation labels
    title_label = tk.Label(window, text="Markov Junior Content Generation", font=("Fixedsys", 20))
    title_label.grid(row=0, column=1, pady=10)

    explanation_label = tk.Label(window, text="Applying the selected grammars\nThe outputs are saved at the root", font=("Fixedsys", 12))
    explanation_label.grid(row=1, column=1, pady=5)

    # Create a frame for canvas and scrollbars
    frame = tk.Frame(window)
    frame.grid(row=2, column=1)

    canvas = tk.Canvas(frame, bg="white")
    canvas.grid(row=2, column=0, sticky="nsew")

    vsb = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    vsb.grid(row=2, column=1, sticky='ns')

    hsb = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
    hsb.grid(row=3, column=0, sticky='ew')

    canvas.config(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    # Create the button frame
    button_frame = tk.Frame(window)
    button_frame.grid(row=4, column=1)

    # Create another frame to center the buttons
    buttons_center_frame = tk.Frame(button_frame)
    buttons_center_frame.pack()

    # Create the buttons
    select_grammars_button = tk.Button(buttons_center_frame, text="New Grammar",
                             command=use_grammars, width=15, height=2, font=("Fixedsys", 12))
    select_grammars_button.pack(side=tk.LEFT, pady=10, padx=5)

    train_button = tk.Button(buttons_center_frame, text="Regenerate",
                             command=regenerate, width=15, height=2, font=("Fixedsys", 12))
    train_button.pack(side=tk.LEFT, pady=10, padx=5)

    back_button = tk.Button(buttons_center_frame, text="Abort",
                            command=terminate, width=15, height=2, font=("Fixedsys", 12))
    back_button.pack(side=tk.LEFT, pady=10, padx=5)

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    display_image(os.path.join(bin_path,"env.png"))
    
    train_button['state'] = tk.DISABLED  # Disable the Regenerate button
    select_grammars_button['state'] = tk.DISABLED
    window.update()
    check_image_existence()


def abort_training():
    global gym_process
    global training_aborted

    training_aborted = True
    # Terminate the parallel process if it's running
    if gym_process and gym_process.is_alive():
        # Kill all child processes first
        parent = psutil.Process(gym_process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.send_signal(signal.SIGKILL)
        
        gym_process.kill()
        gym_process.join()
        print("GYM PROCESS KILLED")

        selected_file_menu()

def abort_markovjun():
    global markovjun_process
    if markovjun_process and markovjun_process.is_alive():
        # Kill all child processes first
        parent = psutil.Process(markovjun_process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.send_signal(signal.SIGKILL)
        
        markovjun_process.kill()
        markovjun_process.join()
        print("markov PROCESS KILLED")
    


def saveStatus(num_offspring_generated, highest_fitness_until_now, average_fitness, current_generation, file_path, population_size):
    parameters = {
        "num_offspring_generated": num_offspring_generated,
        "highest_fitness_until_now": highest_fitness_until_now,
        "average_fitness": average_fitness,
        "current_generation": current_generation,
        "population_size": population_size
    }

    with open(os.path.join(file_path, "status.json"), 'w') as file:
        json.dump(parameters, file)


def save_levelload(load_percentage: float = 0):
    """
    Saves current percentage loading
    """
    parameters = {
        "load_percentage": round(load_percentage)
    }
    with open(os.path.join(output_path, "load.json"), 'w') as file:
        json.dump(parameters, file)


def main_menu():
    window.resizable(True, True)
    window.geometry("900x650")
    window.resizable(False, False)
    clear_window()
    title_label = tk.Label(window, text="Welcome to Markov Senior", font=("Fixedsys", 20))
    title_label.pack(pady=20)

    img = Image.open(os.path.join(bin_path, 'level.png'))
    photo = ImageTk.PhotoImage(img)
    image_label = tk.Label(window, image=photo)
    image_label.image = photo
    image_label.pack(pady=10)

    hint_label = tk.Label(window, text="Please select a training sample to learn a related grammar,\n or use existing markov junior grammars to generate content", font=("Fixedsys", 14))
    hint_label.pack(pady=20)
    
    # Create frame for the buttons
    select_button_frame = tk.Frame(window)
    select_button_frame.pack(pady=20)  # add some vertical space

    # Pack buttons inside the frame
    open_sample_button = tk.Button(select_button_frame, command=select_file, text="Select Sample", width=15, height=2, font=("Fixedsys", 12))
    open_sample_button.pack(side='left', padx=10)  # use side option and add some horizontal space
    open_grammar_button = tk.Button(select_button_frame, command=use_grammars, text="Select Grammar/s", width=15, height=2, font=("Fixedsys", 12))
    open_grammar_button.pack(side='right', padx=10)  # use side option and add some horizontal space


def close_window():
    clear_window()
    abort_markovjun()
    abort_training()
    exit(0)


#clear_directory(output_path)

window = tk.Tk()

window.geometry('900x600')
window.resizable(False, False)
window.title("Markov Senior")
window.protocol("WM_DELETE_WINDOW", close_window)
window.bind("<Control-c>", close_window)

# center the window on the screen
window.eval('tk::PlaceWindow . center')

# Display the main screen
main_menu()

window.mainloop()
