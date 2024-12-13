import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import json
import os
#FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
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
            {
                "queryIdx": m.queryIdx,
                "trainIdx": m.trainIdx,
                "distance": m.distance
            }
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
        if not matches:
            print("No matches to display.")
            return

        try:
            img_matches = cv2.drawMatches(
                img1, keypoints1, img2, keypoints2, matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        except cv2.error as e:
            print(f"Error drawing matches: {e}")
            return

        # Create main application window
        self.window = tk.Tk()
        self.window.title("Match Verification")

        # Embed matplotlib figure in Tkinter
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Feature Matches (Click to Verify)")
        ax.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct plotting
        ax.axis('off')  # Remove axes for a cleaner image view

        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

        # Add Matplotlib toolbar for zoom/pan functionalities
        toolbar = NavigationToolbar2Tk(canvas, self.window)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

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

        frame = tk.Frame(self.window)
        frame.pack(pady=10)
        
        tk.Label(frame, text="Verify Matches").pack(pady=10)
        tk.Button(frame, text="Correct", command=mark_correct, width=20).pack(pady=5)
        tk.Button(frame, text="Incorrect", command=mark_incorrect, width=20).pack(pady=5)

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
