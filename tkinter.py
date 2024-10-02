import customtkinter as ctk
from tkinter import filedialog
import csv

class CSVViewer(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Visualiseur CSV")
        self.geometry("600x400")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.button = ctk.CTkButton(self, text="Sélectionner un fichier CSV", command=self.load_csv)
        self.button.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        self.text_box = ctk.CTkTextbox(self, width=560, height=300)
        self.text_box.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("Fichiers CSV", "*.csv")])
        if file_path:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                content = []
                for i, row in enumerate(csv_reader):
                    if i < 5:
                        content.append(", ".join(row))
                    else:
                        break
                
                self.text_box.delete("1.0", ctk.END)
                self.text_box.insert(ctk.END, "\n".join(content))

if __name__ == "__main__":
    app = CSVViewer()
    app.mainloop()