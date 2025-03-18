import os
import shutil
import cv2
from retinaface import RetinaFace
from deepface import DeepFace

# Define folders
input_folder = "present"      # Group images
database_folder = "database"  # Known faces
people_folder = "people"      # Cropped faces

# ? Step 1: Delete and recreate 'people' folder
if os.path.exists(people_folder):
    shutil.rmtree(people_folder)
os.makedirs(people_folder, exist_ok=True)

# ? Step 2: Extract faces from group images (Process one classroom at a time)
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping {image_name}, unable to read image.")
        continue

    class_name = os.path.splitext(image_name)[0]  # Extract filename without extension
    class_folder = os.path.join(people_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)

    # Detect faces
    faces = RetinaFace.detect_faces(image_path)

    # Crop and save faces
    for i, key in enumerate(faces.keys()):
        face_area = faces[key]["facial_area"]
        x, y, x2, y2 = face_area
        cropped_face = image[y:y2, x:x2]

        face_filename = os.path.join(class_folder, f"face_{i+1}.jpg")
        cv2.imwrite(face_filename, cropped_face)

    # ✅ Confirmation message after processing each classroom
    print(f"✅ All images are saved in {class_name}!")

print("✅ All old files deleted and new faces saved in structured folders inside 'people'!\n")


# ? Step 3: Compare classroom-1 first, then classroom-2, etc.
def process_classroom(classroom):
    classroom_path = os.path.join(people_folder, classroom)
    if not os.path.exists(classroom_path):
        print(f"❌ {classroom} folder not found!")
        return

    # Get known faces from the database
    database_faces = {os.path.splitext(f)[0]: os.path.join(database_folder, f) for f in os.listdir(database_folder)}
    
    # Initialize attendance for all known people
    attendance = {name: "Absent" for name in database_faces.keys()}

    # Get cropped faces in the classroom folder
    people_images = [os.path.join(classroom_path, f) for f in os.listdir(classroom_path)]

    # Compare each cropped face with database
    for person_image in people_images:
        for name, db_image in database_faces.items():
            try:
                result = DeepFace.verify(img1_path=person_image, img2_path=db_image, model_name="ArcFace", enforce_detection=False)
                
                if result["verified"]:
                    attendance[name] = "Present"
                    break  # Stop checking if found
                
            except Exception as e:
                print(f"Error processing {person_image} with {db_image}: {e}")

    # ? Print attendance report
    print(f"\n📌 Attendance for {classroom}")
    for name, status in attendance.items():
        print(f"{name} -> {status}")


# Process classrooms one by one
classrooms = sorted(os.listdir(people_folder))  # Ensure order

for classroom in classrooms:
    process_classroom(classroom)  # Process one classroom at a time
