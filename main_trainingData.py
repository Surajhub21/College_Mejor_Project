import cv2
import os

#.\envkernel\scripts\activate.ps1
# Create the main folder and subfolders
main_folder = 'training_data'
subfolders = [f'object{i}' for i in range(1, 6)]
os.makedirs(main_folder, exist_ok=True)
for subfolder in subfolders:
    os.makedirs(os.path.join(main_folder, subfolder), exist_ok=True)

# Initialize variables to track which folder and image number to use
current_folder_index = 0
current_image_count = 0
max_images_per_folder = 5

# Function to capture and save the image
def capture_image():
    global current_folder_index, current_image_count

    # Check if the current folder is full
    if current_image_count >= max_images_per_folder:
        current_folder_index += 1
        current_image_count = 0
        # If all folders are full, stop capturing
        if current_folder_index >= len(subfolders):
            print("All folders are full.")
            return

    # Capture the image
    ret, frame = cap.read()
    if ret:
        folder_path = os.path.join(main_folder, subfolders[current_folder_index])
        image_path = os.path.join(folder_path, f'image{current_image_count + 1}.jpg')
        cv2.imwrite(image_path, frame)
        current_image_count += 1
        print(f"Saved {image_path}")

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the camera feed
    cv2.imshow('Camera', frame)

    # Capture image on Enter key press
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key
        capture_image()

    # Break the loop if 'q' key is pressed
    if key == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
