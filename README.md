Project Title: AI-Powered Real-Time Virtual Try-On Website

Project Description:

This project is a web-based AI virtual try-on system that allows users to see how clothes would look on them in real time, directly through their webcam — similar to Snapchat filters or AR try-on apps.
Built using Flask (Python) for the backend, HTML/CSS/JavaScript for the frontend, and MediaPipe Pose for real-time human pose tracking, the system overlays user-selected clothing (upper wear, bottom wear, glasses, and shoes) on the user’s live video feed with realistic positioning.

Features Completed So Far:

 Real-Time Clothing Overlay
•	Users can input transparent .png image links of shirts, pants, glasses, and shoes.
•	These items are dynamically rendered on their body using MediaPipe Pose Landmarks via JavaScript.
 Webcam Integration
•	Live video feed is accessed and displayed using navigator.mediaDevices.getUserMedia.
•	Pose landmarks are tracked in real time using MediaPipe in the browser.
Flask Backend (Python)
•	Handles image validation, future support for static try-on, and ML-based recommendations.
•	Integrated with MediaPipe (Python) for server-side pose estimation (optional).
•	Supports clean modular architecture and logs errors clearly.
ML-Based Recommendations (Optional)
•	Basic outfit suggestion system using scikit-learn (can suggest styles like "casual", "formal", "sporty" based on inputs).

Clean and Modern UI
•	The UI has been redesigned with a professional layout using semantic HTML and elegant form styles.
•	Responsive layout ensures central alignment and visual clarity.


Upcoming Work (In Progress)

Automatic Background Removal for Uploaded Images
•	We are currently working on a feature that automatically removes backgrounds from uploaded clothing images (e.g., T-shirts with models).
•	This will eliminate the need for users to manually upload transparent .pngs, making the system easier to use and more flexible.
 Optimizing Accuracy and Speed
•	We aim to enhance the accuracy of clothing placement using refined pose points.
•	Performance will be improved by minimizing canvas redraw time and optimizing image loading.
•	Future improvements include caching clothing items and asynchronously preloading them to reduce latency.


 Technologies Used

Frontend	Backend	AI/ML	Other
HTML, CSS, JS	Flask (Python)	MediaPipe Pose (JS + Python)	scikit-learn, NumPy, OpenCV, Pillow
MediaPipe Pose	Flask-CORS	RandomForestClassifier (optional)	Transparent PNG logic


Demo Flow
1.	User selects gender and pastes clothing image URLs.
2.	Clicks “Start Virtual Try-On”.
3.	Camera opens, and clothes appear on the user's body in real time.
4.	Clothes move with body using pose tracking.

