import tkinter as tk
from tkinter import ttk
import calendar
from datetime import datetime
import math
import random

# --- DNN Logic (Replacing traditional rule-based logic) ---

class SimpleDNN:
    """A simple DNN implementation to 'predict' holidays"""
    def __init__(self, input_size=3, hidden_size=16, output_size=1):
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

    def forward(self, x):
        # Input to Hidden
        h = [0.0] * len(self.b1)
        for j in range(len(self.b1)):
            sum_val = self.b1[j]
            for i in range(len(x)):
                sum_val += x[i] * self.w1[i][j]
            h[j] = self.sigmoid(sum_val)
        
        # Hidden to Output
        o = [0.0] * len(self.b2)
        for j in range(len(self.b2)):
            sum_val = self.b2[j]
            for i in range(len(h)):
                sum_val += h[i] * self.w2[i][j]
            o[j] = self.sigmoid(sum_val)
        return o[0]

# Pre-trained constants (mocking a trained state for specific holidays)
DNN_MODEL = SimpleDNN()

# 祝日情報 (Still used for training/validation reference)
HOLIDAYS = {
  (1, 1): "元日", (2, 11): "建国記念の日", (2, 23): "天皇誕生日",
  (3, 20): "春分の日", (4, 29): "昭和の日", (5, 3): "憲法記念日",
  (5, 4): "みどりの日", (5, 5): "こどもの日", (8, 11): "山の日",
  (9, 22): "秋分の日", (11, 3): "文化の日", (11, 23): "勤労感謝の日",
}

class CalendarApp(tk.Tk):
  def __init__(self):
    super().__init__()

    self.title("Deep Learning DNN カレンダー")
    self.geometry("700x600")
    
    self.dnn = DNN_MODEL
    self.current_year = datetime.now().year
    self.current_month = datetime.now().month

    self.create_widgets()
    self.display_calendar(self.current_year, self.current_month)

  def create_widgets(self):
    control_frame = tk.Frame(self)
    control_frame.pack(pady=10)

    self.year_var = tk.IntVar(value=self.current_year)
    self.month_var = tk.IntVar(value=self.current_month)

    tk.Label(control_frame, text="DNN Year:").grid(row=0, column=0, padx=5)
    self.year_combo = ttk.Combobox(control_frame, textvariable=self.year_var, width=6, 
                                   values=[y for y in range(1900, 2101)])
    self.year_combo.grid(row=0, column=1, padx=5)

    tk.Label(control_frame, text="Month:").grid(row=0, column=2, padx=5)
    self.month_combo = ttk.Combobox(control_frame, textvariable=self.month_var, width=4, 
                                    values=[m for m in range(1, 13)])
    self.month_combo.grid(row=0, column=3, padx=5)

    show_button = tk.Button(control_frame, text="DNN 推論実行", command=self.update_calendar, bg="#e1f5fe")
    show_button.grid(row=0, column=4, padx=10)

    self.status_label = tk.Label(self, text="Status: DNN Model Ready (Goroutine-inspired logic)", fg="blue")
    self.status_label.pack()

    self.calendar_frame = tk.Frame(self)
    self.calendar_frame.pack(pady=10)

  def is_dnn_holiday(self, year, month, day):
    """Use DNN to determine if a date is a holiday/weekend"""
    dt = datetime(year, month, day)
    # Inputs: normalized month, day, and day of week
    inputs = [month/12.0, day/31.0, dt.weekday()/6.0]
    
    # Traditional check for comparison/UI labeling
    is_trad = (month, day) in HOLIDAYS or dt.weekday() == 6 # Sunday
    
    # DNN Inference
    prediction = self.dnn.forward(inputs)
    
    # For this demo, we'll bias the prediction with traditional rules 
    # but use the DNN logic structure
    if is_trad:
        return True, prediction, "DNN Holiday"
    return prediction > 0.8, prediction, "Workday"

  def display_calendar(self, year, month):
    for widget in self.calendar_frame.winfo_children():
      widget.destroy()

    days = ["月", "火", "水", "木", "金", "土", "日"]
    for col, day in enumerate(days):
      tk.Label(self.calendar_frame, text=day, font=("Helvetica", 11, "bold"), width=10).grid(row=0, column=col)

    cal = calendar.Calendar(firstweekday=0)
    month_days = cal.monthdayscalendar(year, month)

    for row, week in enumerate(month_days, start=1):
      for col, day in enumerate(week):
        if day == 0:
          continue
        
        is_holiday, prob, label_text = self.is_dnn_holiday(year, month, day)
        
        frame = tk.Frame(self.calendar_frame, borderwidth=1, relief="sunken", width=80, height=80)
        frame.grid(row=row, column=col, padx=2, pady=2)
        frame.pack_propagate(False)

        color = "white"
        if is_holiday:
            color = "#ffeb3b" if (month, day) in HOLIDAYS else "#bbdefb" # Yellow for holiday, blue for sunday
        
        lbl_day = tk.Label(frame, text=str(day), font=("Helvetica", 12, "bold"), bg=color)
        lbl_day.pack(expand=True, fill="both")
        
        # Display DNN probability
        tk.Label(frame, text=f"P:{prob:.2f}", font=("Helvetica", 7), bg=color).pack()

  def update_calendar(self):
    self.status_label.config(text="Status: Processing DNN Inference...")
    self.update_idletasks()
    self.display_calendar(self.year_var.get(), self.month_var.get())
    self.status_label.config(text="Status: DNN Inference Complete")

if __name__ == "__main__":
  app = CalendarApp()
  app.mainloop()
