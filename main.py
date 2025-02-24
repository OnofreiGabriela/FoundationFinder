import cv2
import numpy as np
import json
import webbrowser
from tkinter import Tk, Label, Button, filedialog, Toplevel, Listbox
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def detect_skin(image):
    """Enhanced skin detection combining YCrCb and HSV color spaces with improved hair exclusion."""

    img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    height, width, _ = image.shape
    forehead_roi = image[int(height * 0.1):int(height * 0.25), width // 4: 3 * width // 4]
    avg_brightness = np.mean(cv2.cvtColor(forehead_roi, cv2.COLOR_BGR2GRAY))

    if avg_brightness < 100:
        HSV_lower, HSV_upper = (0, 15, 0), (17, 170, 255)
        YCrCb_lower, YCrCb_upper = (0, 135, 85), (255, 180, 135)
    else:
        HSV_lower, HSV_upper = (0, 20, 0), (20, 150, 255)
        YCrCb_lower, YCrCb_upper = (80, 135, 85), (255, 180, 135)

    HSV_mask = cv2.inRange(img_HSV, HSV_lower, HSV_upper)
    YCrCb_mask = cv2.inRange(img_YCrCb, YCrCb_lower, YCrCb_upper)

    combined_mask = cv2.bitwise_and(HSV_mask, YCrCb_mask)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.medianBlur(combined_mask, 3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

    edges = cv2.Canny(image, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    combined_mask[edges > 0] = 0

    return combined_mask


def get_region_mean(image, mask, roi):
    """Calculate mean color in the specified ROI."""
    x, y, w, h = roi
    roi_image = image[y:y + h, x:x + w]
    roi_mask = mask[y:y + h, x:x + w]

    if np.count_nonzero(roi_mask) == 0:
        return (0, 0, 0)

    mean_color = cv2.mean(roi_image, mask=roi_mask)
    return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))  # BGR format


def bgr_to_hex(bgr):
    """Convert BGR color to HEX."""
    return "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])


def hex_to_rgb(hex_code):
    """Convert hex color code to RGB format."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))


def color_distance_hex(hex1, hex2):
    """Calculate Euclidean distance between two hex colors."""
    rgb1 = np.array(hex_to_rgb(hex1))
    rgb2 = np.array(hex_to_rgb(hex2))
    return np.linalg.norm(rgb1 - rgb2)


def find_top_shades_hex(detected_hex, shades_file, top_n=3):
    """Find the top N closest matching shades based on hex color."""
    with open(shades_file, 'r') as file:
        shades = json.load(file)

    shade_distances = []

    for shade in shades:
        shade_hex = shade['hex']
        distance = color_distance_hex(detected_hex, shade_hex)
        shade_distances.append((shade, distance))

    # Sort by distance and return the top N
    shade_distances.sort(key=lambda x: x[1])
    return shade_distances[:top_n]


def process_frame(frame, shades_file, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces[:1]:  # Process only the first detected face
        face_region = frame[y:y + h, x:x + w]
        skin_mask = detect_skin(face_region)

        forehead_roi = (w // 4, h // 10, w // 2, h // 8)
        left_cheek_roi = (w // 8, h // 2, w // 4, h // 4)
        right_cheek_roi = (5 * w // 8, h // 2, w // 4, h // 4)

        forehead_mean = get_region_mean(face_region, skin_mask, forehead_roi)
        left_cheek_mean = get_region_mean(face_region, skin_mask, left_cheek_roi)
        right_cheek_mean = get_region_mean(face_region, skin_mask, right_cheek_roi)

        combined_mean = (
            (forehead_mean[0] + left_cheek_mean[0] + right_cheek_mean[0]) // 3,
            (forehead_mean[1] + left_cheek_mean[1] + right_cheek_mean[1]) // 3,
            (forehead_mean[2] + left_cheek_mean[2] + right_cheek_mean[2]) // 3,
        )
        combined_hex = bgr_to_hex(combined_mean)
        top_matches = find_top_shades_hex(combined_hex, shades_file)

        return combined_hex, top_matches

    return None, []


def on_shade_click(event, matches):
    """Open the URL for the selected foundation."""
    widget = event.widget
    selection = widget.curselection()
    if not selection:
        return  
    index = selection[0] 
    if index < len(matches):  
        shade = matches[index][0]  
        if "url" in shade:
            webbrowser.open(shade["url"])


def update_results_window(hex_color, matches, listbox):
    """Update the real-time results window with the latest shades."""
    listbox.delete(0, "end")
    listbox.insert("end", f"Detected Color: {hex_color}")
    listbox.insert("end", "")

    for match in matches:
        shade = match[0]
        listbox.insert(
            "end", f"{shade['shade']} ({shade['hex']}) - {shade['brand']} - {shade['product']}"
        )

    def on_item_select(event):
        """Open the corresponding URL for the selected shade."""
        selection = listbox.curselection()
        if not selection or selection[0] < 2: 
            return
        index = selection[0] - 2  
        if index < len(matches):
            shade = matches[index][0]  
            if "url" in shade:
                webbrowser.open(shade["url"])

    listbox.bind("<<ListboxSelect>>", on_item_select)


def start_webcam():
    """Start webcam processing with real-time shade updates."""
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        messagebox.showerror("Error", "Webcam not accessible.")
        return


    results_window = Toplevel(root)
    results_window.title("Top Matches (Real-Time)")
    results_window.geometry("400x300")

    listbox = Listbox(results_window, font=("Arial", 12), width=50, height=15)
    listbox.pack(pady=10)

    def process_webcam():
        ret, frame = webcam.read()
        if not ret or frame is None:
            return

        detected_hex, top_matches = process_frame(frame, 'shades.json', face_cascade)
        if detected_hex and top_matches:
            update_results_window(detected_hex, top_matches, listbox)

        cv2.imshow("Webcam - Skin Tone Detection", frame)

        if cv2.waitKey(1) & 0xFF != ord('q'):
            results_window.after(10, process_webcam)
        else:
            webcam.release()
            cv2.destroyAllWindows()
            results_window.destroy()

    process_webcam()


def open_image():
    """Open an image file."""
    filepath = askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not filepath:
        return

    image = cv2.imread(filepath)
    if image is None:
        messagebox.showerror("Error", "Could not load the image.")
        return

    detected_hex, top_matches = process_frame(image, 'shades.json', face_cascade)
    if detected_hex and top_matches:
        results_window = Toplevel(root)
        results_window.title("Top Matches")
        results_window.geometry("400x300")

        listbox = Listbox(results_window, font=("Arial", 12), width=50, height=15)
        listbox.pack(pady=10)
        update_results_window(detected_hex, top_matches, listbox)

    cv2.imshow("Processed Image", scale_image(image, 800, 800))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def scale_image(image, max_width = 800, max_height = 800):
    """Scale image to fit within the specified dimensions while maintaining aspect ratio."""
    height, width = image.shape[:2]

    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return scaled_image


def exclude_polygons_from_skin_mask(image):
    """Exclude specific facial features (eyes, eyebrows, lips, etc.) using updated Mediapipe polygons."""
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            print("No face detected.")
            return None  
        height, width, _ = image.shape
        skin_mask = np.ones((height, width), dtype=np.uint8) * 255

        for face_landmarks in results.multi_face_landmarks:
            
            polygons = {
                "lipsOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91,
                              146],
                "lipsInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
                "rightEye": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                                    144, 163, 7],
                "leftEye": [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                                   380, 381, 382, 362],
                "rightEyebrow": [70, 63, 105, 66, 107, 55, 222, 52, 53, 46],
                "leftEyebrow": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
            }

            
            def get_polygon_coords(indices):
                return np.array(
                    [(int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height)) for i in indices],
                    dtype=np.int32,
                )

        
            for key, indices in polygons.items():
                polygon = get_polygon_coords(indices)
                cv2.fillConvexPoly(skin_mask, polygon, 0)  # Fill the region with black (0)

        return skin_mask




# def exclude_outside_silhouette(image):
#     """Exclude regions outside the face silhouette from the skin mask."""
#     with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
#         results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         if not results.multi_face_landmarks:
#             return None  # No face detected

#         height, width, _ = image.shape
#         mask = np.zeros((height, width), dtype=np.uint8)

#         for face_landmarks in results.multi_face_landmarks:
#             # Define the silhouette polygon
#             silhouette_indices = [
#                 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
#                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
#                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
#             ]
#             silhouette_polygon = np.array([
#                 (int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height))
#                 for i in silhouette_indices
#             ], dtype=np.int32)

#             # Fill the silhouette region
#             cv2.fillPoly(mask, [silhouette_polygon], 255)

#         return mask



def select_foundation_from_matches(matches, image):
    """Allow the user to select a foundation shade and apply it to the image."""
    # Create a selection window
    selection_window = Toplevel(root)
    selection_window.title("Select Foundation")
    selection_window.geometry("400x350")

    listbox = Listbox(selection_window, font=("Arial", 12), width=50, height=15)
    listbox.pack(pady=10)

    for match in matches:
        shade = match[0]
        listbox.insert("end", f"{shade['shade']} ({shade['hex']}) - {shade['brand']}")

    selected_shade = []

    def on_select():
        selection = listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "No selection made.")
            return
        selected_shade.append(matches[selection[0]][0])  # Get the selected shade
        selection_window.destroy()

    Button(selection_window, text="Apply", command=on_select).pack(pady=10)
    root.wait_window(selection_window)

    if not selected_shade:
        return None

    # Apply the selected foundation shade
    return apply_selected_foundation(image, selected_shade[0])

def apply_selected_foundation(image, foundation_shade):
    """Apply the selected foundation shade while blending with natural textures only on detected skin areas, while excluding eyes, eyebrows, and lips."""
    
    foundation_rgb = np.array(hex_to_rgb(foundation_shade['hex']), dtype=np.uint8)
    foundation_bgr = foundation_rgb[::-1]  

    skin_mask = detect_skin(image)

    exclusion_mask = exclude_polygons_from_skin_mask(image)

    if skin_mask is None or exclusion_mask is None:
        print("No face detected.")
        return image

    # Combine both masks: Keep only skin areas while excluding eyes, eyebrows, lips
    final_mask = cv2.bitwise_and(skin_mask, exclusion_mask)

    smoothed_image = cv2.bilateralFilter(image, d=15, sigmaColor=75, sigmaSpace=75)

    foundation_layer = np.full_like(image, foundation_bgr, dtype=np.uint8)

    alpha_mask = (final_mask / 255.0)[:, :, None]  # Normalize mask to [0, 1] and expand dimensions
    beautified_image = cv2.convertScaleAbs(
        alpha_mask * cv2.addWeighted(smoothed_image, 0.8, foundation_layer, 0.2, 0) +
        (1 - alpha_mask) * image
    )
    return beautified_image



def apply_foundation_button():
    """Process and beautify an uploaded image with the ability to apply a selected foundation shade."""
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not filepath:
        return

    image = cv2.imread(filepath)
    if image is None:
        messagebox.showerror("Error", "Could not load the image.")
        return

    # Process the frame to get detected skin color and matches
    detected_hex, top_matches = process_frame(image, 'shades.json', face_cascade)
    if not detected_hex or not top_matches:
        messagebox.showinfo("Info", "No suitable matches found.")
        return

    # Allow the user to select and apply a foundation
    result_image = select_foundation_from_matches(top_matches, image)
    if result_image is None:
        return

    cv2.imshow("Foundation Applied Image", scale_image(result_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    root = Tk()
    root.title("Skin Tone Detection")
    root.geometry("400x200")

    Label(root, text="Skin Tone Detection", font=("Arial", 16)).pack(pady=20)
    Button(root, text="Open Image", command=open_image, width=20).pack(pady=10)
    Button(root, text="Start Webcam", command=start_webcam, width=20).pack(pady=10)
    Button(root, text="Apply Foundation", command=apply_foundation_button, width=20).pack(pady=10)

    root.mainloop()