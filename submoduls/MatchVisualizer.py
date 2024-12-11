import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import json
import os

class MatchVisualizer:
    def __init__(self, save_file="match_results.json"):
        self.correct_matches = []
        self.incorrect_matches = []
        self.save_file = save_file
        self.load_previous_results()
        
    def load_previous_results(self):
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file, "r") as file:
                    data = json.load(file)
                    self.correct_matches = data.get("correct_matches", [])
                    self.incorrect_matches = data.get("incorrect_matches", [])
            except json.JSONDecodeError:
                print("Error loading JSON: File is corrupted or empty.")
                self.correct_matches = []
                self.incorrect_matches = []

    def serialize_matches(self, matches):
        return [
            {"queryIdx": m.get("queryIdx"), "trainIdx": m.get("trainIdx"), "distance": m.get("distance", 0)}
            if isinstance(m, dict) else
            {"queryIdx": m.queryIdx, "trainIdx": m.trainIdx, "distance": m.distance}
            for m in matches
        ]

    def save_results(self):
        try:
            data = {
                "correct_matches": self.serialize_matches(self.correct_matches),
                "incorrect_matches": self.serialize_matches(self.incorrect_matches)
            }
            with open(self.save_file, "w") as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error saving results: {e}")

    def visualize_matches(self, img1, img2, keypoints1, keypoints2, matches):
        """
        Visualize matches with a clickable interface to verify matches.
        """
        img_matches = cv2.drawMatches(
            img1, keypoints1, img2, keypoints2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Create main application window
        self.window = tk.Tk()
        self.window.title("Match Verification")
        
        # Embed matplotlib figure in Tkinter
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Feature Matches (Click to Verify)")
        ax.imshow(img_matches)
        
        # Create canvas for embedding the matplotlib figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

        # Add buttons
        self.create_verification_buttons(matches)

        # Start the GUI event loop
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        try:
            self.window.mainloop()
        except Exception as e:
            print(f"Error during GUI execution: {e}")

    def create_verification_buttons(self, matches):
        """
        Create verification buttons in the Tkinter window.
        """
        def mark_correct():
            try:
                self.correct_matches.extend(matches)
                self.save_results()
                messagebox.showinfo("Info", "Marked as correct!")
                self.window.destroy()
            except Exception as e:
                print(f"Error marking as correct: {e}")

        def mark_incorrect():
            try:
                self.incorrect_matches.extend(matches)
                self.save_results()
                messagebox.showinfo("Info", "Marked as incorrect!")
                self.window.destroy()
            except Exception as e:
                print(f"Error marking as incorrect: {e}")

        tk.Label(self.window, text="Verify Matches").pack(pady=10)
        tk.Button(self.window, text="Correct", command=mark_correct, width=20).pack(pady=5)
        tk.Button(self.window, text="Incorrect", command=mark_incorrect, width=20).pack(pady=5)

    def on_close(self):
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit without saving?"):
            self.window.destroy()

    def get_evaluation_results(self):
        """
        Calculate and return the matching success rate.
        """
        total_matches = len(self.correct_matches) + len(self.incorrect_matches)
        if total_matches == 0:
            return 0
        success_rate = (len(self.correct_matches) / total_matches) * 100
        return success_rate

# Example usage
if __name__ == "__main__":
    img1 = cv2.imread("example1.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("example2.jpg", cv2.IMREAD_COLOR)

    # Dummy feature points and matches
    keypoints1 = [cv2.KeyPoint(x=50, y=50, _size=1), cv2.KeyPoint(x=150, y=150, _size=1)]
    keypoints2 = [cv2.KeyPoint(x=60, y=60, _size=1), cv2.KeyPoint(x=160, y=160, _size=1)]
    matches = [cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.1),
               cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=0.2)]

    visualizer = MatchVisualizer()
    visualizer.visualize_matches(img1, img2, keypoints1, keypoints2, matches)
    success_rate = visualizer.get_evaluation_results()
    print(f"Matching Success Rate: {success_rate:.2f}%")
