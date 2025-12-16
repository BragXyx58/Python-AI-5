import tkinter as tk
from tkinter import filedialog, scrolledtext, Label, simpledialog
from PIL import Image, ImageTk, ImageFilter
import colorsys
import numpy as np


class PlanetaryAI:
    def __init__(self, root):
        self.root = root
        self.root.title("Planetary AI v8.0 (Blob Detection & Color Temp)")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f0f0")

        # UI
        top_frame = tk.Frame(root, pady=10, bg="#f0f0f0")
        top_frame.pack(fill=tk.X)
        self.btn_load = tk.Button(top_frame, text="Загрузить фото планеты", command=self.process_image,
                                  font=("Arial", 12), bg="#e1e1e1", fg="black")
        self.btn_load.pack(side=tk.TOP, pady=5)

        center_frame = tk.Frame(root, bg="#f0f0f0")
        center_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.lbl_image = Label(center_frame, text="Нет изображения", bg="#e0e0e0", fg="black")
        self.lbl_image.pack(pady=10)
        self.lbl_result = Label(center_frame, text="Ожидание...", font=("Arial", 22, "bold"), bg="#f0f0f0", fg="black")
        self.lbl_result.pack(pady=5)
        self.lbl_details = Label(center_frame, text="", font=("Consolas", 10), justify=tk.LEFT, bg="#f0f0f0", fg="#444")
        self.lbl_details.pack(pady=5)

    def get_largest_blob_crop(self, img):
        """
        Главное исправление: Ищет самый большой СВЯЗНЫЙ объект (Blob).
        Это гарантированно отсекает звезды, которые не касаются планеты.
        """
        # 1. Работаем с уменьшенной копией для скорости
        scale_w, scale_h = 200, 200
        small = img.resize((scale_w, scale_h))
        gray = small.convert('L')
        # Порог 45 убирает тусклый фон
        binary = gray.point(lambda p: 255 if p > 45 else 0)
        width, height = binary.size
        pixels = binary.load()

        visited = set()
        largest_blob = []

        # 2. Поиск в ширину (BFS) для поиска островов
        for x in range(0, width, 2):  # Шаг 2 для скорости
            for y in range(0, height, 2):
                if pixels[x, y] == 255 and (x, y) not in visited:
                    # Нашли новый остров
                    current_blob = []
                    stack = [(x, y)]
                    visited.add((x, y))

                    min_x, max_x, min_y, max_y = x, x, y, y

                    while stack:
                        cx, cy = stack.pop()
                        current_blob.append((cx, cy))

                        # Обновляем границы текущего острова
                        if cx < min_x: min_x = cx
                        if cx > max_x: max_x = cx
                        if cy < min_y: min_y = cy
                        if cy > max_y: max_y = cy

                        # Соседи
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if pixels[nx, ny] == 255 and (nx, ny) not in visited:
                                    visited.add((nx, ny))
                                    stack.append((nx, ny))

                    # Если этот остров больше предыдущего лидера - запоминаем его границы
                    if len(current_blob) > len(largest_blob):
                        largest_blob = current_blob
                        best_bbox = (min_x, min_y, max_x, max_y)

        if not largest_blob:
            return img, 1.0

        # 3. Масштабируем координаты обратно к оригиналу
        orig_w, orig_h = img.size
        scale_x = orig_w / scale_w
        scale_y = orig_h / scale_h

        bx1, by1, bx2, by2 = best_bbox
        final_box = (
            int(bx1 * scale_x), int(by1 * scale_y),
            int(bx2 * scale_x), int(by2 * scale_y)
        )

        # Даем небольшой отступ
        pad = 5
        final_box = (
            max(0, final_box[0] - pad), max(0, final_box[1] - pad),
            min(orig_w, final_box[2] + pad), min(orig_h, final_box[3] + pad)
        )

        crop = img.crop(final_box)
        cw, ch = crop.size
        aspect = cw / ch if ch > 0 else 1.0
        return crop, aspect

    def detect_banding(self, img):
        gray = img.convert('L').resize((100, 100))
        arr = np.array(gray)
        vertical_var = np.std(np.mean(arr, axis=1))
        horizontal_var = np.std(np.mean(arr, axis=0))
        if horizontal_var == 0: return 0
        return vertical_var / horizontal_var

    def identify_pixel_group(self, h, s, v):
        if (0.28 <= h <= 0.75) and s > 0.15: return "Земля"
        if s < 0.20 and v > 0.65: return "Венера"

        # Разделяем Теплый спектр на Красный и Бежевый
        if ((0.0 <= h <= 0.18) or (0.94 <= h <= 1.0)):
            if s > 0.10:
                # Hue < 0.04 (около 14 градусов) = Марс
                # Hue > 0.04 = Юпитер/Сатурн
                if (0.0 <= h <= 0.045) or (0.98 <= h <= 1.0):
                    return "Red"
                else:
                    return "Beige"
        return None

    def analyze_image(self, image_path):
        try:
            original = Image.open(image_path).convert('RGB')
            # 1. BLOB DETECT (Убирает звезды)
            cropped_img, aspect_ratio = self.get_largest_blob_crop(original)

            # 2. Анализ полос
            banding_score = self.detect_banding(cropped_img)

            analysis_img = cropped_img.resize((100, 100))
            width, height = analysis_img.size

            counts = {"Земля": 0, "Венера": 0, "Red": 0, "Beige": 0}
            total_pixels = 0

            for x in range(width):
                for y in range(height):
                    r, g, b = analysis_img.getpixel((x, y))
                    if r + g + b > 40:
                        total_pixels += 1
                        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
                        group = self.identify_pixel_group(h, s, v)
                        if group:
                            counts[group] += 1

            if total_pixels == 0: return "Ошибка", 0.0, {}, aspect_ratio, 0

            res = {k: v / total_pixels for k, v in counts.items()}

            final_scores = {}

            # Сумма теплых (Марс + Юпитер + Сатурн)
            warm_sum = res["Red"] + res["Beige"]

            # Победа Земли/Венеры?
            if res["Земля"] > warm_sum and res["Земля"] > res["Венера"]:
                final_scores["Земля"] = res["Земля"]
            elif res["Венера"] > warm_sum and res["Венера"] > res["Земля"]:
                final_scores["Венера"] = res["Венера"]
            else:
                # --- ЛОГИКА ТЕПЛОЙ ГРУППЫ ---

                # 1. КОЛЬЦА (Сатурн)
                # Сатурн всегда > 1.3. Если кроп чистый (Blob), Марс будет ~1.0.
                if aspect_ratio > 1.30:
                    final_scores["Сатурн"] = warm_sum

                # 2. ПОЛОСЫ (Юпитер)
                # Порог поднят до 1.6, чтобы пятна Марса точно не прошли.
                elif banding_score > 1.60:
                    final_scores["Юпитер"] = warm_sum

                # 3. АНАЛИЗ ЦВЕТА ИЛИ СЛАБЫХ ПОЛОС
                else:
                    # Если полосы средние (1.2 - 1.6) - сомневаемся
                    # Смотрим на цвет: Если больше Красного -> Марс. Если больше Бежевого -> Юпитер.

                    if res["Red"] > res["Beige"]:
                        # Доминирует красный -> Скорее всего Марс
                        final_scores["Марс"] = warm_sum
                    else:
                        # Доминирует бежевый -> Юпитер (если полосы есть) или Сатурн (если нет)
                        if banding_score > 1.25:
                            final_scores["Юпитер"] = warm_sum
                        else:
                            # Бежевый шар без колец и полос?
                            # Это может быть Венера (желтая) или Сатурн без колец, или пыльный Марс.
                            # Отдадим приоритет Марсу, так как Сатурн без колец редок.
                            final_scores["Марс"] = warm_sum * 0.8
                            final_scores["Сатурн"] = warm_sum * 0.2

            sorted_res = sorted(final_scores.items(), key=lambda i: i[1], reverse=True)
            if not sorted_res: return "Неизвестно", 0.0, res, aspect_ratio, banding_score

            return sorted_res[0][0], sorted_res[0][1], final_scores, aspect_ratio, banding_score

        except Exception as e:
            print(e)
            return "Ошибка", 0.0, {}, 0.0, 0.0

    def process_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return

        try:
            display_img = Image.open(file_path)
            display_img.thumbnail((400, 350))
            photo = ImageTk.PhotoImage(display_img)
            self.lbl_image.config(image=photo, width=0, height=0)
            self.lbl_image.image = photo
        except:
            pass

        name, score, stats, aspect, banding = self.analyze_image(file_path)

        text = f"Аспект (Кольца): {aspect:.2f} (>1.3=Сатурн)\n"
        text += f"Полосы (Юпитер): {banding:.2f} (>1.6=Юпитер)\n"
        text += "-" * 30 + "\n"
        if stats:
            for p, s in stats.items():
                if s > 0.01: text += f"{p}: {s:.1%}\n"

        self.lbl_details.config(text=text)

        final_color = "black"
        verdict = f"Это {name.upper()}"

        if score < 0.40:
            self.lbl_result.config(text="Неизвестно", fg="gray")
            user_input = simpledialog.askstring("?", f"Сомнения ({score:.0%}). Это {name}?\nКто это?")
            if user_input:
                verdict = user_input + " (User)"
                final_color = "#9c27b0"
        else:
            colors = {"Марс": "#d32f2f", "Земля": "#1976d2", "Юпитер": "#e65100",
                      "Сатурн": "#fbc02d", "Венера": "#009688"}
            final_color = colors.get(name, "black")

        self.lbl_result.config(text=verdict, fg=final_color)


if __name__ == "__main__":
    root = tk.Tk()
    app = PlanetaryAI(root)
    root.mainloop()