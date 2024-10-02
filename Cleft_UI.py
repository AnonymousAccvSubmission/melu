import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import sys
#sys.path.append('/Users/Daniel/Desktop/PhdWork/latest/journal/E2F-GAN_gui/')
#sys.path.append('./Inpainting/')
#from E2FNet_mine import E2FNet_Net  # Importing the E2FNet_Net class
from Inpainting.model_gui import Cleft_Inpainting_model
def kp_to_edges(keypoints, canvas, keypoint_ids):
    edges = {
        'lips': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                 (0, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 6),
                 (0, 19), (19, 18), (18, 17), (17, 6), (16, 6),
                 (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 0)]
    }

    canvas.delete("edge")  # Clear previous edges
    for feature, edge_list in edges.items():
        for pt1_idx, pt2_idx in edge_list:
            pt1_id = keypoint_ids[pt1_idx][0]
            pt2_id = keypoint_ids[pt2_idx][0]
            x1, y1 = canvas.coords(pt1_id)[:2]
            x2, y2 = canvas.coords(pt2_id)[:2]
            canvas.create_line(x1+5, y1+5, x2+5, y2+5, fill="white", width=2, tags="edge")


class ImageKeypointsApp:
    def __init__(self, root, model) :
        self.root = root
        self.root.title("Image and Keypoints Viewer")
        self.model_name = model
        # Initialize the E2FNet_Net model with the pretrained model path
        
          # Example path to the pretrained model
        if self.model_name == 'e2f':
            model_path = '/Users/Daniel/Desktop/PhdWork/latest/journal/E2F-GAN_gui/ckpt/ffhq_65k'
            self.model = E2FNet_Net(model_path)  # Load the pretrained model
        if self.model_name == 'agg':
            model_path = './Inpainting/pretrained/generator_best.pth'
            self.model = Cleft_Inpainting_model(model_path)
        # Set up the canvas for the image and keypoints
        self.canvas_image = tk.Canvas(root, width=256, height=256, bg='white')
        self.canvas_image.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas_keypoints = tk.Canvas(root, width=256, height=256, bg='white')
        self.canvas_keypoints.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas_inpainted = tk.Canvas(root, width=256, height=256, bg='white')  # Third canvas for inpainted image
        self.canvas_inpainted.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a frame to hold "Load Image" and "Load Mask" buttons in one line
        load_frame = tk.Frame(root)
        load_frame.pack(pady=5)

        self.load_image_button = tk.Button(load_frame, text="Load Image", command=self.load_image)
        self.load_image_button.pack(side=tk.LEFT, padx=5)

        self.load_mask_button = tk.Button(load_frame, text="Load Mask", command=self.load_mask)  # New button
        self.load_mask_button.pack(side=tk.LEFT, padx=5)

        self.load_keypoints_button = tk.Button(root, text="Load Keypoints", command=self.load_keypoints)
        self.load_keypoints_button.pack(pady=5)

        # Create a frame to hold "Display Keypoints" and "Reset Keypoints" in one line
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        # Display Keypoints and Reset Keypoints in the same line
        self.display_keypoints_button = tk.Button(button_frame, text="Display Keypoints", command=self.show_keypoints)
        self.display_keypoints_button.pack(side=tk.LEFT, padx=5)

        self.reset_keypoints_button = tk.Button(button_frame, text="Reset Keypoints", command=self.reset_keypoints)
        self.reset_keypoints_button.pack(side=tk.LEFT, padx=5)

        self.generate_image_button = tk.Button(root, text="Generate Edge Plot", command=self.generate_edge_map)
        self.generate_image_button.pack(pady=5)

        self.inpaint_image_button = tk.Button(root, text="Inpaint Image", command=self.inpaint_image)  # New button
        self.inpaint_image_button.pack(pady=5)

        self.save_image_button = tk.Button(root, text="Save Image", command=self.save_image)
        self.save_image_button.pack(pady=5)

        self.reset_all_button = tk.Button(root, text="Reset All", command=self.reset_all)  # New button for resetting everything
        self.reset_all_button.pack(pady=5)

        self.quit_button = tk.Button(root, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=5)

        self.image = None
        self.image_path = None  # To store the path of the loaded image
        self.mask = None
        self.keypoints = None
        self.keypoints_ids = []
        self.original_keypoints = []
        self.generated_edge = None
        self.drag_data = {"item": None, "x": 0, "y": 0}

        self.canvas_image.bind("<Button-1>", self.on_click)
        self.canvas_image.bind("<B1-Motion>", self.on_drag)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path  # Save the loaded image path
            self.image = Image.open(file_path)
            self.thumbnail = self.image.copy()
            self.thumbnail.thumbnail((256, 256))  # Adjust image size to 256x256
            self.thumbnail_tk = ImageTk.PhotoImage(self.thumbnail)
            self.canvas_image.create_image(0, 0, anchor=tk.NW, image=self.thumbnail_tk)
            self.canvas_image.image = self.thumbnail_tk

    def save_image(self):
        """Saves the inpainted image with the same basename as the loaded image."""
        if self.image_path and self.inpainted_img:
            # Get the basename of the loaded image
            base_name = os.path.basename(self.image_path)
            file_name, _ = os.path.splitext(base_name)
            save_img_name = f"{file_name}_inpainted.png"
            save_path = os.path.join('./inpainted_imgs', save_img_name)
            # Save the inpainted image if it exists
            self.inpainted_img.save(save_path)
            print(f"Inpainted image saved as {save_path}")
        else:
            print("No inpainted image to save.")


    def load_mask(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
            self.mask = cv2.resize(self.mask, (256, 256))  # Resize to match the image

            if self.image is not None:
                # Convert PIL image to OpenCV format
                image_cv = np.array(self.image.resize((256, 256)))
                if image_cv.shape[2] == 4:  # Remove alpha channel if present
                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2RGB)

                # Apply mask using cv2.addWeighted
                overlay = cv2.addWeighted(image_cv, 0.5, cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB), 0.5, 0)

                # Convert back to PIL format and display
                self.masked_image = Image.fromarray(overlay)
                self.masked_image_tk = ImageTk.PhotoImage(self.masked_image)
                self.canvas_image.create_image(0, 0, anchor=tk.NW, image=self.masked_image_tk)
                self.canvas_image.image = self.masked_image_tk
            else:
                print("Please load an image first.")

    def load_keypoints(self):
        if self.image is None:
            print("Please load an image first.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as f:
                lines = f.read()
                keypoints = np.array(list(map(int, lines.split()))).reshape(-1, 2)

            self.keypoints = {'points': keypoints}
            self.original_keypoints = keypoints.copy()  # Save original keypoints for reset
            print("Keypoints loaded.")

    def show_keypoints(self):
        if self.image is None or self.keypoints is None:
            print("Please load an image and keypoints first.")
            return

        # If mask is loaded, use the masked image
        if self.mask is not None:
            # Display the masked image with keypoints
            self.canvas_image.delete("all")
            self.image_with_keypoints = self.masked_image.copy()
        else:
            # If no mask is loaded, use the original image
            self.canvas_image.delete("all")
            self.image_with_keypoints = self.image.copy()

        self.image_with_keypoints.thumbnail((256, 256))  # Ensure the image is 256x256
        self.image_with_keypoints_tk = ImageTk.PhotoImage(self.image_with_keypoints)
        self.canvas_image.create_image(0, 0, anchor=tk.NW, image=self.image_with_keypoints_tk)

        self.keypoints_ids = []

        # Define the special keypoints that should be displayed in red
        special_keypoints = {1, 2, 3, 4, 5, 13, 14, 15, 16, 12}

        # Loop through keypoints and draw them with different colors
        for i, point in enumerate(self.keypoints['points']):
            x, y = point
            if i in special_keypoints:
                color = "red"  # Highlighted color for the specified keypoints
            else:
                color = "black"  # Default color for other keypoints

            keypoint_id = self.canvas_image.create_oval(x-5, y-5, x+5, y+5, fill=color, tags="keypoint")
            self.keypoints_ids.append((keypoint_id, point))

        # Draw edges after placing the keypoints
        kp_to_edges(self.keypoints['points'], self.canvas_image, self.keypoints_ids)


    def generate_edge_map(self):
        """Generates a blank image and plots the edges using keypoints."""
        if self.keypoints is None:
            print("Please load keypoints first.")
            return
        
        # Create a blank image (256x256) with all zeros (black background)
        generated_image = np.zeros((256, 256), dtype=np.uint8)
        
        # Define the edges to be drawn
        edges = {
            'lips': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                     (0, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 6),
                     (0, 19), (19, 18), (18, 17), (17, 6), (16, 6),
                     (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 0)]
        }

        # Draw edges on the blank image
        for feature, edge_list in edges.items():
            for pt1_idx, pt2_idx in edge_list:
                pt1 = self.keypoints['points'][pt1_idx]
                pt2 = self.keypoints['points'][pt2_idx]
                
                # Draw white lines (255) to represent edges
                cv2.line(generated_image, tuple(pt1), tuple(pt2), 255, 2)

        # Save the generated edge for later use
        self.generated_edge = generated_image

        # Convert the NumPy array back to an image using PIL
        generated_image_pil = Image.fromarray(generated_image)
        generated_image_tk = ImageTk.PhotoImage(generated_image_pil)

        # Display the generated image on the second canvas
        self.canvas_keypoints.create_image(0, 0, anchor=tk.NW, image=generated_image_tk)
        self.canvas_keypoints.image = generated_image_tk

    def inpaint_image(self):
        """Inpaint the image using the loaded model, image, mask, and generated edge."""
        if self.image is None or self.mask is None or self.generated_edge is None:
            print("Please load an image, mask, and generate an edge first.")
            return

        # Convert the image to the format expected by the model
        image_cv = np.array(self.image.resize((256, 256)))
        if image_cv.shape[2] == 4:  # Remove alpha channel if present
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2RGB)

        # Ensure the mask is of the correct size
        mask_resized = cv2.resize(self.mask, (256, 256))

        # Ensure the edge is in the correct format (if not already)
        if len(self.generated_edge.shape) == 2:
            edge_input = np.expand_dims(self.generated_edge, axis=-1)
        else:
            edge_input = self.generated_edge

        # Inpaint the image using the E2FNet_Net model
        inpainted_image_cv = self.model.predict(image_cv, mask_resized, edge_input)

        # Convert the output from the model back to PIL format for display
        inpainted_image_pil = Image.fromarray(inpainted_image_cv)
        inpainted_image_tk = ImageTk.PhotoImage(inpainted_image_pil)
        self.inpainted_img = inpainted_image_pil
        # Display the inpainted image on the third canvas
        self.canvas_inpainted.create_image(0, 0, anchor=tk.NW, image=inpainted_image_tk)
        self.canvas_inpainted.image = inpainted_image_tk

    def on_click(self, event):
        """Handles mouse click on a keypoint and checks if it can be dragged based on mask values."""
        for keypoint_id, point in self.keypoints_ids:
            x1, y1, x2, y2 = self.canvas_image.coords(keypoint_id)
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                # Get the current position of the keypoint on the canvas
                canvas_x = (x1 + x2) // 2
                canvas_y = (y1 + y2) // 2

                # Check if the keypoint falls in the region where the mask value is 255
                if self.mask is not None:
                    # Get the corresponding position in the mask (assuming both are 256x256)
                    mask_value = self.mask[int(canvas_y), int(canvas_x)]

                    if mask_value == 255 or mask_value == 0:
                        # Allow movement
                        self.drag_data["item"] = keypoint_id
                        self.drag_data["x"] = event.x
                        self.drag_data["y"] = event.y
                        print(f"Keypoint {keypoint_id} is movable.")
                    else:
                        # Do not allow movement for this keypoint
                        self.drag_data["item"] = None
                        print(f"Keypoint {keypoint_id} is fixed due to mask region.")
                else:
                    # If no mask is loaded, allow all keypoints to be moved
                    self.drag_data["item"] = keypoint_id
                    self.drag_data["x"] = event.x
                    self.drag_data["y"] = event.y
                break

    def on_drag(self, event):
        """Handles dragging of keypoints and restricts movement to mask regions with value 255."""
        if self.drag_data["item"]:  # Only move if the keypoint is movable
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]

            # Get the current position of the keypoint
            for i, (keypoint_id, (px, py)) in enumerate(self.keypoints_ids):
                if keypoint_id == self.drag_data["item"]:
                    new_x = px + dx
                    new_y = py + dy

                    # Ensure the new position is within the mask region where the value is 255
                    if self.mask is not None:
                        mask_value = self.mask[int(new_y), int(new_x)] if 0 <= int(new_x) < 256 and 0 <= int(new_y) < 256 else 0
                        if mask_value == 255 or mask_value == 0:
                            # Update the keypoint position only if within the allowed mask region
                            self.canvas_image.move(self.drag_data["item"], dx, dy)
                            self.drag_data["x"] = event.x
                            self.drag_data["y"] = event.y
                            self.keypoints['points'][i] = (new_x, new_y)
                            self.keypoints_ids[i] = (keypoint_id, (new_x, new_y))
                            print(f"Keypoint {keypoint_id} moved to ({new_x}, {new_y})")
                        else:
                            print(f"Keypoint {keypoint_id} cannot be moved outside the mask region.")
                    else:
                        # If no mask is loaded, allow movement freely
                        self.canvas_image.move(self.drag_data["item"], dx, dy)
                        self.drag_data["x"] = event.x
                        self.drag_data["y"] = event.y
                        self.keypoints['points'][i] = (new_x, new_y)
                        self.keypoints_ids[i] = (keypoint_id, (new_x, new_y))

                    break

            # Update edges in real-time
            kp_to_edges(self.keypoints['points'], self.canvas_image, self.keypoints_ids)

    def reset_keypoints(self):
        """Resets all keypoints and edges to their original positions and clears the second canvas."""
        if self.keypoints is None:
            return

        # Reset keypoints to their original positions
        self.keypoints['points'] = self.original_keypoints.copy()

        # Use the masked image if available, otherwise the original image
        self.canvas_image.delete("all")
        if self.mask is not None:
            self.image_with_keypoints = self.masked_image.copy()  # Keep the mask
        else:
            self.image_with_keypoints = self.image.copy()

        self.image_with_keypoints.thumbnail((256, 256))  # Ensure image is also 256x256
        self.image_with_keypoints_tk = ImageTk.PhotoImage(self.image_with_keypoints)
        self.canvas_image.create_image(0, 0, anchor=tk.NW, image=self.image_with_keypoints_tk)

        self.keypoints_ids = []

        # Define the special keypoints that should be displayed in red
        special_keypoints = {1, 2, 3, 4, 5, 13, 14, 15, 16, 12}

        # Loop through keypoints and draw them with different colors
        for i, point in enumerate(self.keypoints['points']):
            x, y = point
            if i in special_keypoints:
                color = "red"  # Highlighted color for the specified keypoints
            else:
                color = "black"  # Default color for other keypoints

            keypoint_id = self.canvas_image.create_oval(x-5, y-5, x+5, y+5, fill=color, tags="keypoint")
            self.keypoints_ids.append((keypoint_id, point))

        # Redraw edges after resetting the keypoints
        kp_to_edges(self.keypoints['points'], self.canvas_image, self.keypoints_ids)

        # Clear the second canvas
        self.canvas_keypoints.delete("all")

    def reset_all(self):
        """Resets everything: clears canvases, image, mask, and keypoints."""
        # Clear both canvases
        self.canvas_image.delete("all")
        self.canvas_keypoints.delete("all")
        self.canvas_inpainted.delete("all")

        # Reset internal data
        self.image = None
        self.mask = None
        self.keypoints = None
        self.keypoints_ids = []
        self.original_keypoints = []
        self.generated_edge = None

        print("All data has been reset.")

    def quit_app(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    model = "agg" #select between ([e2f, agg, deepfill])
    app = ImageKeypointsApp(root, model)
    
    root.mainloop()
       
