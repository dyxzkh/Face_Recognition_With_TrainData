import face_recognition
import cv2

# Dictionary to store known faces and their corresponding names
known_faces = {
    "Mark": "./train_face/m1.jpg",
    "Mark": "./train_face/m3.jpg",
    "Mark": "./train_face/m2.png",

    "Elon": "./train_face/e1.jpg",
    "Elon": "./train_face/e3.jpg",
    "Elon": "./train_face/e2.jpg",

    "JetLi": "./train_face/j1.jpg",
    "JetLi": "./train_face/j2.jpg",
    "JetLi": "./train_face/j3.jpg",
    "JetLi": "./train_face/j4.jpg",
    # Add more entries as needed
}

def recognize_face(image_path):
    # Load the image
    unknown_image = face_recognition.load_image_file(image_path)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(unknown_image)

    # Encode the faces in the image
    unknown_face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Loop through each face found in the image
    for (top, right, bottom, left), unknown_face_encoding in zip(face_locations, unknown_face_encodings):
        # Compare the unknown face with known faces
        for known_name, known_face_encoding in known_faces.items():
            # Load the known face image
            known_image = face_recognition.load_image_file(known_face_encoding)

            # Encode the known face
            known_face_encoding = face_recognition.face_encodings(known_image)[0]

            # Compare faces
            results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)

            # Check if the face matches
            if True in results:
                print(f"Face recognized: {known_name}")
                # You can perform further actions here, such as storing the recognized face's name, etc.

    # Display the image with rectangles around the recognized faces
    show_recognized_faces(image_path, face_locations)

def show_recognized_faces(image_path, face_locations):
    # Load the image
    image = cv2.imread(image_path)

    # Draw rectangles around the recognized faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Recognized Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "jetli_input.jpg"
recognize_face(image_path)
