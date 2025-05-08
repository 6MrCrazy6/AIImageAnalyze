import os
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from detector import ObjectDetector, Classifier, Segmenter, ImageAnalyzer
from download_assets import download_all_assets

BG = "#F7F7F7"
PANEL = "#FFFFFF"
BORDER = "#E0E0E0"
TEXT_COLOR = "#333"
FONT_PLACEHOLDER = ("Roboto", 16, "bold")
FONT_TEXT = ("Roboto", 12, "bold")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.iconbitmap(os.path.join("UI", "icon.ico"))
        self.title("Image Analyzer")
        self.geometry("1100x600")
        self.configure(bg=BG)

        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.create_text(550, 40, text="Image Analyzer", fill="#222", font=("Roboto", 22, "bold"))

        self.canvas.create_rectangle(10, 80, 510, 590, outline=BORDER, fill=PANEL, width=2)
        self.canvas.create_rectangle(520, 80, 820, 590, outline=BORDER, fill=PANEL, width=2)
        self.canvas.create_rectangle(830, 80, 1090, 590, outline=BORDER, fill=PANEL, width=2)

        self.img_canvas = tk.Canvas(self, bg=PANEL, width=490, height=500, highlightthickness=0)
        self.img_canvas.place(x=15, y=85)
        self.img_canvas.create_text(245, 250, text="No image selected", fill="#AAA", font=FONT_PLACEHOLDER)

        self.result_panel = tk.Canvas(self, bg=PANEL, width=290, height=500, highlightthickness=0)
        self.result_panel.place(x=525, y=85)
        self.result_text_id = self.result_panel.create_text(
            145, 10, text="", fill=TEXT_COLOR, font=FONT_TEXT, anchor="n", width=270)

        self.status = self.canvas.create_text(960, 570, text="Loading models...", fill="#888", font=FONT_TEXT)

        self.button_files = [
            "choose_img_button.png",
            "advanced_analysis_button.png",
            "exit_button.png"
        ]
        self.actions = [self.choose, self.advanced, self.quit]
        self.buttons = []
        self.buttons_images = []
        self.original_images = []

        for i, (file, action) in enumerate(zip(self.button_files, self.actions)):
            path = os.path.join("UI", file)
            img = Image.open(path).resize((240, 60), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.original_images.append(img)
            self.buttons_images.append(img_tk)

            x, y = 960, 130 + i * 80
            btn = self.canvas.create_image(x, y, image=img_tk)
            self.canvas.tag_bind(btn, "<Button-1>", lambda e, idx=i: self.animate_button(idx))
            self.buttons.append(btn)

        self.disabled_images = {}
        self.disable_buttons([0, 1])
        self.save_button_img = None
        self.save_button = None

        threading.Thread(target=self.setup, daemon=True).start()

    def disable_buttons(self, indices):
        for idx in indices:
            img_path = os.path.join("UI", self.button_files[idx])
            im = Image.open(img_path).convert("RGBA").resize((240, 60), Image.Resampling.LANCZOS)
            alpha = im.split()[3].point(lambda p: int(p * 0.4))
            im.putalpha(alpha)
            ph = ImageTk.PhotoImage(im)
            self.disabled_images[idx] = ph
            self.canvas.itemconfig(self.buttons[idx], image=ph)
            self.canvas.itemconfig(self.buttons[idx], state="disabled")

    def enable_buttons(self, indices):
        for idx in indices:
            img_path = os.path.join("UI", self.button_files[idx])
            im = Image.open(img_path).resize((240, 60), Image.Resampling.LANCZOS)
            ph = ImageTk.PhotoImage(im)
            self.buttons_images[idx] = ph
            self.canvas.itemconfig(self.buttons[idx], image=ph)
            self.canvas.itemconfig(self.buttons[idx], state="normal")

    def setup(self):
        download_all_assets()
        self.det = ObjectDetector()
        self.clf = Classifier()
        self.seg = Segmenter()
        self.ia = ImageAnalyzer()
        self.canvas.after(0, lambda: self.canvas.itemconfig(self.status, text="Models loaded", fill="#4CAF50"))
        self.canvas.after(0, lambda: self.enable_buttons([0, 1]))

    def animate_button(self, index):
        btn = self.buttons[index]
        steps = 5
        scale_down = 0.92
        scale_up = 1.0

        def scale(step):
            factor = scale_down + (scale_up - scale_down) * step / steps
            img = self.original_images[index].resize(
                (int(240 * factor), int(60 * factor)), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.buttons_images[index] = img_tk
            self.canvas.itemconfig(btn, image=img_tk)
            x, y = 960, 130 + index * 80
            self.canvas.coords(btn, x, y)

        def animate_down(i=0):
            if i <= steps:
                scale(i)
                self.after(10, lambda: animate_down(i + 1))
            else:
                animate_up(steps)

        def animate_up(i):
            if i >= 0:
                scale(i)
                self.after(10, lambda: animate_up(i - 1))
            else:
                self.actions[index]()

        animate_down()

    def choose(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not path:
            return
        im = Image.open(path)
        im.thumbnail((480, 500))
        ph = ImageTk.PhotoImage(im)
        self.img_canvas.image = ph
        self.img_canvas.delete("all")
        self.img_canvas.create_image(0, 0, anchor="nw", image=ph)
        self.path = path
        det = self.det.detect(path)

        if isinstance(det, list) and det and isinstance(det[0], dict):
            # Собираем все лейблы без повторов
            unique_labels = sorted(set(d["label"] for d in det))
            txt = "Detected:\n" + ("\n".join(f"• {label}" for label in unique_labels) if unique_labels else "None")
        else:
            txt = "Detected:\n" + ("None" if not det else "• " + str(det))

        self.result_panel.itemconfig(self.result_text_id, text=txt)
        self.show_save_button()

    def advanced(self):
        if not hasattr(self, "path"):
            return

        cls = self.clf.classify(self.path)
        results = self.ia.perform_extended_analysis(self.path)

        det = results.get("detected_objects", [])
        object_counts = results.get("object_counts", {})

        object_boxes = [obj['coords'] for obj in det]
        unique_labels = set(d["label"] for d in det)
        txt = f"Classification: {cls}\n\n"
        txt += "Detected:\n" + "\n".join(f"• {label}" for label in unique_labels) + "\n\n"
        txt += "Object Counts:\n" + (
            "\n".join(f"- {k}: {v}" for k, v in object_counts.items()) if object_counts else "None")

        self.result_panel.itemconfig(self.result_text_id, text=txt)

        try:
            im = Image.open(self.path).convert("RGB")
            draw = ImageDraw.Draw(im)

            font_size = 10
            font = ImageFont.truetype("arial.ttf", font_size)

            for i, box in enumerate(object_boxes):
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                    if i < len(det):
                        label = det[i]["label"]
                        draw.text((x1 + 5, y1 + 5), label, fill="red", font=font)

            im.thumbnail((480, 500), Image.LANCZOS)
            ph = ImageTk.PhotoImage(im)

            self.img_canvas.delete("all")
            self.img_canvas.create_image(0, 0, anchor="nw", image=ph)
            self.img_canvas.image = ph
        except Exception as e:
            print("Ошибка при загрузке и рисовании:", e)

        self.show_save_button()

    def show_save_button(self):
        if self.save_button:
            return
        img_path = os.path.join("UI", "save_to_file_button.png")
        img = Image.open(img_path).resize((180, 40), Image.Resampling.LANCZOS)
        self.save_button_img = ImageTk.PhotoImage(img)

        x, y = 145, 470
        self.save_button = self.result_panel.create_image(x, y, image=self.save_button_img)
        self.result_panel.tag_bind(self.save_button, "<Button-1>", lambda e: self.save_result())

    def save_result(self):
        txt = self.result_panel.itemcget(self.result_text_id, "text")
        if not txt.strip():
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)

if __name__ == "__main__":
    App().mainloop()
