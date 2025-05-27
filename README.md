# Fitness Trainer with Pose Estimation

An AI-powered web application that tracks exercises using computer vision and provides real-time feedback.

## Features

- Real-time pose estimation using MediaPipe
- Multiple exercise types: Squats, Push-ups, and Hammer Curls
- Customizable sets and repetitions
- Exercise form feedback
- Progress tracking
- Web interface for easy access

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fitness-trainer-pose-estimation.git
   cd fitness-trainer-pose-estimation
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the static folder structure:
   ```
   mkdir -p static/images
   ```

4. Add exercise images to the static/images folder:
   - squat.png
   - push_up.png
   - hammer_curl.png

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. Select an exercise type, set your desired number of repetitions and sets, then click "Start Workout"

4. Position yourself in front of your camera so that your full body is visible

5. Follow the on-screen guidance to perform the exercise correctly

## Project Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates
- `static/` - CSS, JavaScript, and images
- `pose_estimation/` - Pose estimation modules
- `exercises/` - Exercise tracking classes
- `feedback/` - User feedback modules
- `utils/` - Helper functions and utilities

## Technologies Used

- Flask - Web framework
- OpenCV - Computer vision
- MediaPipe - Pose estimation
- HTML/CSS/JavaScript - Frontend

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
