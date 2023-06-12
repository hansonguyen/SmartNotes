import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from ocr import detect_document
from enhancer import enhance
 
ctk.set_appearance_mode("System")   
ctk.set_default_color_theme("dark-blue")   
appWidth, appHeight = 800, 1000

# App Class
class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("SmartNotes")  
        self.geometry(f"{appWidth}x{appHeight}")   
 
        # Filename Label
        self.filenameLabel = ctk.CTkLabel(self, text="Filename (Must be in images directory)")
        self.filenameLabel.grid(
            row=0, column=0,
            padx=20, pady=20,
            sticky="ew"
        )
 
        # Filename Entry Field
        self.filenameEntry = ctk.CTkEntry(self, placeholder_text="ex. note.jpg")
        self.filenameEntry.grid(
            row=0, column=1,
            columnspan=4, padx=20,
            pady=20, sticky="ew"
        )
        
        # Topics Label
        self.topicsLabel = ctk.CTkLabel(self, text="Related Topics (separate by '/')")
        self.topicsLabel.grid(
            row=1, column=0,
            padx=20, pady=20,
            sticky="ew"
        )
 
        # Topics Entry Field
        self.topicsEntry = ctk.CTkEntry(self, placeholder_text="ex. Topic 1/Topic 2")
        self.topicsEntry.grid(
            row=1, column=1,
            columnspan=4, padx=20,
            pady=20, sticky="ew"
        )
        
        # Enhance Button
        self.generateResultsButton = ctk.CTkButton(self, text="Enhance my notes", command=self.generateResults)
        self.generateResultsButton.grid(
            row=2, column=1,
            padx=20, pady=20,
            sticky="ew",
            columnspan=4
        )

        self.progressbar = ttk.Progressbar(mode="indeterminate")
        self.progressbar.grid(
            row=3, column=1,
            columnspan=4, padx=20, pady=20,
            sticky="ew"
        )

        # Output Label
        self.outputLabel = ctk.CTkLabel(self, text="Enhanced Notes")
        self.outputLabel.grid(
            row=4, column=0,
            padx=20, pady=20,
            sticky="ew"
        )
 
        # Output Box
        self.displayBox = ctk.CTkTextbox(self, width=500, height=600)
        self.displayBox.grid(
            row=5, column=0,
            columnspan=6, padx=20,
            pady=20, sticky="nsew"
        )
 
    def generateResults(self):
        self.generateResultsButton.configure(state="disabled")
        self.progressbar.start()
        self.displayBox.delete("0.0", "200.0")
        IMAGE_NAME = self.filenameEntry.get()
        TOPICS = self.topicsEntry.get()
        try:
            output = detect_document(IMAGE_NAME)
            enhanced_output = enhance(IMAGE_NAME, output, TOPICS)
            self.displayBox.insert("0.0", enhanced_output)
        except Exception as e:
            self.progressbar.stop()
            tk.messagebox.showerror("Error", str(e))
        finally:
            self.generateResultsButton.configure(state="normal")
            self.progressbar.stop()
 
if __name__ == "__main__":
    app = App()
    app.mainloop()