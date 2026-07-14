"""
VideoCapture loop with a graceful exit and a guard for "no camera available."

Why this matters: this repo is developed on a machine that may not have a
camera attached (and definitely doesn't have the Pi's camera). A script that
crashes with an unhandled exception when cv2.VideoCapture fails to open is
useless -- it needs to detect that case and exit cleanly, the same way
raspberry-pi/camera_inference.py has to.
"""

import cv2


def run_webcam_loop(camera_index=0, max_frames=None):
    """Open a webcam, show frames until 'q' is pressed, and exit gracefully.

    max_frames caps how many frames to process (used for automated/headless
    runs of this script so it doesn't hang forever waiting for a keypress).
    Returns True if a camera was successfully opened and used, False if not.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"No camera available at index {camera_index} -- this is expected on a headless")
        print("machine or CI runner. Exiting cleanly instead of crashing.")
        return False

    print("Camera opened. Press 'q' in the window to quit.")
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read a frame -- camera disconnected? Stopping.")
                break

            cv2.imshow("Webcam feed (press q to quit)", frame)
            frame_count += 1

            if max_frames is not None and frame_count >= max_frames:
                print(f"Reached max_frames={max_frames}, stopping automatically.")
                break

            # waitKey(1) processes GUI events and checks for a keypress; required for imshow to work.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit key pressed.")
                break
    finally:
        # Always release the camera and destroy windows, even if something above raised.
        cap.release()
        cv2.destroyAllWindows()
        print(f"Released camera after {frame_count} frames.")

    return True


if __name__ == "__main__":
    # On this development machine (no physical webcam attached in this environment),
    # this is expected to print the "no camera available" message and exit with code 0 --
    # that is the honest, correct behavior being tested here, not a failure.
    camera_was_available = run_webcam_loop(camera_index=0, max_frames=100)
    print(f"\ncamera_was_available: {camera_was_available}")
