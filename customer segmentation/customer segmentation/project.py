"""
Customer Segmentation & Purchase Prediction System
===================================================
Dataset: Based on Kaggle "Predict Customer Purchase Behavior Dataset"
   Link: https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset

Features used:
  - Age, Gender, AnnualIncome, NumberOfPurchases,
    ProductCategory, TimeSpentOnWebsite, LoyaltyProgram,
    DiscountsAvailed, PurchaseStatus (target)

Models:
  - KMeans Clustering  → Customer Segmentation
  - Random Forest      → Purchase Prediction

GUI: Tkinter with a modern dark theme
"""

import tkinter as tk
from tkinter import ttk, messagebox
import csv
import os
import math

# ─── Attempt imports; guide user if missing ─────────────────────────
try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
except ImportError as e:
    print("Missing library. Install with:")
    print("  pip install numpy scikit-learn")
    raise SystemExit(e)

# Optional: matplotlib for charts inside Tkinter
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# =====================================================================
#  1. LOAD DATASET
# =====================================================================
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "customer_purchase_data.csv")

def load_csv(path):
    """Read CSV into list of dicts."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

raw = load_csv(r"C:\Users\Admin\Documents\customer segmentation\customer segmentation\customer_purchase_data.csv")

# Parse into numpy arrays
feature_cols = ["Age", "AnnualIncome", "NumberOfPurchases",
                "TimeSpentOnWebsite", "LoyaltyProgram", "DiscountsAvailed"]
target_col   = "PurchaseStatus"

X_all = np.array([[float(r[c]) for c in feature_cols] for r in raw])
y_all = np.array([int(r[target_col]) for r in raw])
genders = np.array([int(r["Gender"]) for r in raw])
categories = np.array([int(r["ProductCategory"]) for r in raw])

# =====================================================================
#  2. FEATURE ENGINEERING  (add SpendingScore)
# =====================================================================
# SpendingScore = normalised combo of Income, Purchases, TimeOnSite
income_norm    = (X_all[:, 1] - X_all[:, 1].min()) / (X_all[:, 1].max() - X_all[:, 1].min())
purchases_norm = (X_all[:, 2] - X_all[:, 2].min()) / (X_all[:, 2].max() - X_all[:, 2].min())
time_norm      = (X_all[:, 3] - X_all[:, 3].min()) / (X_all[:, 3].max() - X_all[:, 3].min())
spending_score = (0.4 * income_norm + 0.35 * purchases_norm + 0.25 * time_norm) * 100

X_all = np.column_stack([X_all, spending_score])   # now 7 features
feature_cols.append("SpendingScore")

# =====================================================================
#  3. SCALE & TRAIN MODELS
# =====================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# --- KMeans (3 segments) ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

# Map clusters to meaningful names by mean spending score
cluster_means = {}
for c in range(3):
    cluster_means[c] = spending_score[cluster_labels == c].mean()
sorted_clusters = sorted(cluster_means, key=cluster_means.get)
SEGMENT_MAP = {
    sorted_clusters[0]: ("Budget Shoppers",    "#E74C3C"),
    sorted_clusters[1]: ("Regular Customers",  "#F39C12"),
    sorted_clusters[2]: ("Premium Buyers",     "#2ECC71"),
}

# --- Random Forest Classifier ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all, test_size=0.25, random_state=42, stratify=y_all
)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# Feature importances
importances = rf.feature_importances_

# =====================================================================
#  4. SEGMENT STATISTICS  (for dashboard)
# =====================================================================
seg_stats = {}
for cid, (name, color) in SEGMENT_MAP.items():
    mask = cluster_labels == cid
    seg_stats[name] = {
        "color":        color,
        "count":        int(mask.sum()),
        "avg_age":      round(X_all[mask, 0].mean(), 1),
        "avg_income":   round(X_all[mask, 1].mean(), 0),
        "avg_purchases": round(X_all[mask, 2].mean(), 1),
        "avg_time":     round(X_all[mask, 3].mean(), 1),
        "avg_score":    round(X_all[mask, 6].mean(), 1),
        "purchase_rate": round(y_all[mask].mean() * 100, 1),
    }

# =====================================================================
#  5. CATEGORY LABELS
# =====================================================================
CATEGORIES = {0: "Electronics", 1: "Clothing", 2: "Home Goods",
              3: "Beauty", 4: "Sports"}

# =====================================================================
#  6. GUI  — Modern Dark Tkinter
# =====================================================================
# Colour palette
BG        = "#0F0F0F"
CARD_BG   = "#1A1A2E"
ACCENT    = "#0F3460"
HIGHLIGHT = "#E94560"
TEXT      = "#EAEAEA"
TEXT_DIM  = "#888899"
SUCCESS   = "#2ECC71"
WARNING   = "#F39C12"
DANGER    = "#E74C3C"
INPUT_BG  = "#16213E"
INPUT_FG  = "#EAEAEA"

FONT_TITLE = ("Segoe UI", 22, "bold")
FONT_HEAD  = ("Segoe UI", 14, "bold")
FONT_BODY  = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 9)
FONT_BIG   = ("Segoe UI", 28, "bold")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Customer Intelligence Dashboard")
        self.configure(bg=BG)
        self.geometry("1100x780")
        self.minsize(1000, 700)

        # ── Title bar ──
        top = tk.Frame(self, bg=HIGHLIGHT, height=56)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="◆  CUSTOMER INTELLIGENCE DASHBOARD",
                 font=FONT_TITLE, fg="white", bg=HIGHLIGHT,
                 padx=20).pack(side="left", fill="y")
        tk.Label(top, text=f"Model Accuracy: {model_accuracy*100:.1f}%",
                 font=FONT_BODY, fg="white", bg=HIGHLIGHT,
                 padx=20).pack(side="right", fill="y")

        # ── Notebook (tabs) ──
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=ACCENT, foreground=TEXT,
                        font=("Segoe UI", 11, "bold"), padding=[18, 8])
        style.map("TNotebook.Tab",
                  background=[("selected", HIGHLIGHT)],
                  foreground=[("selected", "white")])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # Tab 1 — Dashboard
        self.tab_dashboard = tk.Frame(nb, bg=BG)
        nb.add(self.tab_dashboard, text="  📊  Dashboard  ")
        self._build_dashboard()

        # Tab 2 — Predict
        self.tab_predict = tk.Frame(nb, bg=BG)
        nb.add(self.tab_predict, text="  🔮  Predict  ")
        self._build_predict()

        # Tab 3 — Dataset Info
        self.tab_info = tk.Frame(nb, bg=BG)
        nb.add(self.tab_info, text="  📋  Dataset Info  ")
        self._build_info()

    # ─────────────────────────────────────────────
    #  DASHBOARD TAB
    # ─────────────────────────────────────────────
    def _build_dashboard(self):
        parent = self.tab_dashboard

        # Top stat cards
        cards_frame = tk.Frame(parent, bg=BG)
        cards_frame.pack(fill="x", padx=20, pady=(15, 5))

        stats = [
            ("Total Customers", str(len(raw)),         ACCENT),
            ("Purchase Rate",   f"{y_all.mean()*100:.0f}%", SUCCESS),
            ("Segments",        "3",                    WARNING),
            ("Features Used",   str(len(feature_cols)), HIGHLIGHT),
        ]
        for i, (label, value, color) in enumerate(stats):
            cards_frame.columnconfigure(i, weight=1)
            card = tk.Frame(cards_frame, bg=color, bd=0, highlightthickness=0)
            card.grid(row=0, column=i, padx=8, pady=5, sticky="nsew")
            tk.Label(card, text=value, font=FONT_BIG, fg="white",
                     bg=color, pady=5).pack()
            tk.Label(card, text=label, font=FONT_SMALL, fg="#ffffff",
                     bg=color, pady=8).pack()
                     
         

        # Segment detail cards
        seg_frame = tk.Frame(parent, bg=BG)
        seg_frame.pack(fill="x", padx=20, pady=10)
        tk.Label(seg_frame, text="CUSTOMER SEGMENTS", font=FONT_HEAD,
                 fg=TEXT_DIM, bg=BG).pack(anchor="w", pady=(0, 8))

        row_frame = tk.Frame(seg_frame, bg=BG)
        row_frame.pack(fill="x")
        for i, (name, info) in enumerate(seg_stats.items()):
            row_frame.columnconfigure(i, weight=1)
            c = tk.Frame(row_frame, bg=CARD_BG, bd=0,
                         highlightbackground=info["color"],
                         highlightthickness=2)
            c.grid(row=0, column=i, padx=8, sticky="nsew")

            header = tk.Frame(c, bg=info["color"], height=6)
            header.pack(fill="x")

            tk.Label(c, text=name, font=FONT_HEAD, fg=info["color"],
                     bg=CARD_BG, pady=8).pack()

            details = [
                f"Customers:  {info['count']}",
                f"Avg Age:  {info['avg_age']}",
                f"Avg Income:  ₹{info['avg_income']:,.0f}",
                f"Avg Purchases:  {info['avg_purchases']}",
                f"Avg Time on Site:  {info['avg_time']} min",
                f"Spending Score:  {info['avg_score']}",
                f"Purchase Rate:  {info['purchase_rate']}%",
            ]
            for d in details:
                tk.Label(c, text=d, font=FONT_SMALL, fg=TEXT_DIM,
                         bg=CARD_BG, anchor="w", padx=15).pack(anchor="w")

            tk.Label(c, text="", bg=CARD_BG).pack()  # spacer

        # Feature importance (bar chart with canvas)
        fi_frame = tk.Frame(parent, bg=CARD_BG,
                            highlightbackground=ACCENT, highlightthickness=1)
        fi_frame.pack(fill="x", padx=28, pady=(10, 5))
        tk.Label(fi_frame, text="FEATURE IMPORTANCE (Random Forest)",
                 font=FONT_HEAD, fg=TEXT_DIM, bg=CARD_BG,
                 pady=8, padx=12).pack(anchor="w")
        self._draw_bar_chart(fi_frame)

    def _draw_bar_chart(self, parent):
        """Draw a simple horizontal bar chart on a Canvas."""
        canvas = tk.Canvas(parent, bg=CARD_BG, height=180,
                           highlightthickness=0)
        canvas.pack(fill="x", padx=15, pady=(0, 12))
        self.update_idletasks()

        bar_data = sorted(zip(feature_cols, importances),
                          key=lambda x: x[1], reverse=True)
        max_imp = max(importances)
        bar_h   = 20
        gap     = 6
        label_w = 140
        y = 10
        colors = [HIGHLIGHT, "#E94560", "#0F3460", WARNING, SUCCESS,
                  "#9B59B6", "#1ABC9C"]

        for i, (feat, imp) in enumerate(bar_data):
            canvas.create_text(label_w - 5, y + bar_h // 2,
                               text=feat, anchor="e",
                               fill=TEXT_DIM, font=FONT_SMALL)
            bar_w = int((imp / max_imp) * 500)
            canvas.create_rectangle(label_w, y, label_w + bar_w, y + bar_h,
                                    fill=colors[i % len(colors)], outline="")
            canvas.create_text(label_w + bar_w + 8, y + bar_h // 2,
                               text=f"{imp:.3f}", anchor="w",
                               fill=TEXT, font=FONT_SMALL)
            y += bar_h + gap

    # ─────────────────────────────────────────────
    #  PREDICTION TAB
    # ─────────────────────────────────────────────
    def _build_predict(self):
        parent = self.tab_predict

        # Two-column layout
        left = tk.Frame(parent, bg=CARD_BG, bd=0,
                        highlightbackground=ACCENT, highlightthickness=1)
        left.pack(side="left", fill="both", expand=True,
                  padx=(20, 10), pady=20)

        right = tk.Frame(parent, bg=CARD_BG, bd=0,
                         highlightbackground=ACCENT, highlightthickness=1)
        right.pack(side="right", fill="both", expand=True,
                   padx=(10, 20), pady=20)

        # ── Left: Input form ──
        tk.Label(left, text="ENTER CUSTOMER DATA", font=FONT_HEAD,
                 fg=HIGHLIGHT, bg=CARD_BG, pady=12, padx=15).pack(anchor="w")

        form = tk.Frame(left, bg=CARD_BG, padx=20)
        form.pack(fill="x")

        self.entries = {}
        fields = [
            ("Age",                "e.g. 30"),
            ("Annual Income",      "e.g. 50000"),
            ("Number of Purchases","e.g. 10"),
            ("Time on Website (min)","e.g. 45"),
            ("Discounts Availed",  "e.g. 3"),
        ]
        for i, (label, placeholder) in enumerate(fields):
            tk.Label(form, text=label, font=FONT_BODY, fg=TEXT_DIM,
                     bg=CARD_BG, anchor="w").grid(row=i, column=0,
                     sticky="w", pady=(10, 0))
            e = tk.Entry(form, font=FONT_BODY, bg=INPUT_BG, fg=INPUT_FG,
                         insertbackground=INPUT_FG, relief="flat",
                         highlightbackground=ACCENT, highlightthickness=1,
                         width=22)
            e.insert(0, placeholder)
            e.bind("<FocusIn>", lambda ev, entry=e, ph=placeholder:
                   self._clear_placeholder(entry, ph))
            e.bind("<FocusOut>", lambda ev, entry=e, ph=placeholder:
                   self._restore_placeholder(entry, ph))
            e.grid(row=i, column=1, padx=(10, 0), pady=(10, 0), sticky="e")
            self.entries[label] = e

        # Loyalty toggle
        self.loyalty_var = tk.IntVar(value=0)
        tk.Label(form, text="Loyalty Program", font=FONT_BODY,
                 fg=TEXT_DIM, bg=CARD_BG).grid(row=len(fields), column=0,
                 sticky="w", pady=(10, 0))
        loy_frame = tk.Frame(form, bg=CARD_BG)
        loy_frame.grid(row=len(fields), column=1, pady=(10, 0), sticky="e")
        for txt, val in [("No", 0), ("Yes", 1)]:
            tk.Radiobutton(loy_frame, text=txt, variable=self.loyalty_var,
                           value=val, font=FONT_SMALL, fg=TEXT, bg=CARD_BG,
                           selectcolor=INPUT_BG,
                           activebackground=CARD_BG).pack(side="left", padx=5)

        # Buttons
        btn_frame = tk.Frame(left, bg=CARD_BG)
        btn_frame.pack(pady=20)

        self.btn_predict = tk.Button(
            btn_frame, text="⚡  PREDICT", font=("Segoe UI", 12, "bold"),
            bg=HIGHLIGHT, fg="white", activebackground="#c73e52",
            activeforeground="white", relief="flat", padx=25, pady=8,
            cursor="hand2", command=self._predict)
        self.btn_predict.pack(side="left", padx=8)

        self.btn_reset = tk.Button(
            btn_frame, text="↺  RESET", font=("Segoe UI", 12, "bold"),
            bg=ACCENT, fg="white", activebackground="#0a2a4a",
            activeforeground="white", relief="flat", padx=25, pady=8,
            cursor="hand2", command=self._reset)
        self.btn_reset.pack(side="left", padx=8)

        # ── Right: Results ──
        tk.Label(right, text="PREDICTION RESULTS", font=FONT_HEAD,
                 fg=HIGHLIGHT, bg=CARD_BG, pady=12, padx=15).pack(anchor="w")

        self.result_widgets = {}
        results_inner = tk.Frame(right, bg=CARD_BG, padx=20)
        results_inner.pack(fill="both", expand=True)

        result_items = [
            ("Segment",             "segment"),
            ("Purchase Prediction", "purchase"),
            ("Confidence",          "confidence"),
            ("Spending Score",      "score"),
            ("Marketing Strategy",  "strategy"),
        ]
        for i, (label, key) in enumerate(result_items):
            tk.Label(results_inner, text=label, font=FONT_BODY,
                     fg=TEXT_DIM, bg=CARD_BG, anchor="w").grid(
                     row=i, column=0, sticky="w", pady=12)
            val_label = tk.Label(results_inner, text="—", font=FONT_HEAD,
                                 fg=TEXT, bg=CARD_BG, anchor="w", width=22)
            val_label.grid(row=i, column=1, padx=(15, 0),
                           pady=12, sticky="w")
            self.result_widgets[key] = val_label

        # Suggestion box
        self.suggestion_box = tk.Label(
            right, text="", font=FONT_BODY, fg=BG, bg=CARD_BG,
            wraplength=380, justify="left", padx=15, pady=10)
        self.suggestion_box.pack(fill="x", padx=15, pady=(0, 15))

    # ── Placeholder helpers ──
    def _clear_placeholder(self, entry, ph):
        if entry.get() == ph:
            entry.delete(0, tk.END)

    def _restore_placeholder(self, entry, ph):
        if entry.get().strip() == "":
            entry.insert(0, ph)

    # ── Prediction logic ──
    def _predict(self):
        try:
            vals = {}
            mapping = {
                "Age": 0, "Annual Income": 1,
                "Number of Purchases": 2,
                "Time on Website (min)": 3,
                "Discounts Availed": 4,
            }
            for label, idx in mapping.items():
                raw_val = self.entries[label].get()
                vals[label] = float(raw_val)
                if vals[label] < 0:
                    raise ValueError(f"{label} cannot be negative")

            age       = vals["Age"]
            income    = vals["Annual Income"]
            purchases = vals["Number of Purchases"]
            time_web  = vals["Time on Website (min)"]
            discounts = vals["Discounts Availed"]
            loyalty   = self.loyalty_var.get()

            # Compute spending score same way as training
            inc_n  = (income - X_all[:, 1].min()) / (X_all[:, 1].max() - X_all[:, 1].min())
            pur_n  = (purchases - X_all[:, 2].min()) / (X_all[:, 2].max() - X_all[:, 2].min())
            tim_n  = (time_web - X_all[:, 3].min()) / (X_all[:, 3].max() - X_all[:, 3].min())
            inc_n  = max(0, min(1, inc_n))
            pur_n  = max(0, min(1, pur_n))
            tim_n  = max(0, min(1, tim_n))
            s_score = (0.4 * inc_n + 0.35 * pur_n + 0.25 * tim_n) * 100

            new_row = np.array([[age, income, purchases, time_web,
                                  loyalty, discounts, s_score]])
            new_scaled = scaler.transform(new_row)

            # Segment
            seg_id   = kmeans.predict(new_scaled)[0]
            seg_name, seg_color = SEGMENT_MAP[seg_id]

            # Purchase prediction
            pred       = rf.predict(new_scaled)[0]
            prob       = rf.predict_proba(new_scaled)[0]
            confidence = round(max(prob) * 100, 1)
            purchase   = "✅  LIKELY" if pred == 1 else "❌  UNLIKELY"
            pur_color  = SUCCESS if pred == 1 else DANGER

            # Strategy
            strategies = {
                "Budget Shoppers":   "→ Offer flash sales & budget bundles\n→ Highlight value-for-money products\n→ Send discount coupons via email",
                "Regular Customers": "→ Suggest combo deals & loyalty perks\n→ Cross-sell complementary products\n→ Personalized weekly newsletters",
                "Premium Buyers":    "→ Exclusive early-access to new launches\n→ Premium membership & free shipping\n→ VIP customer support channel",
            }

            # Update UI
            self.result_widgets["segment"].config(text=seg_name, fg=seg_color)
            self.result_widgets["purchase"].config(text=purchase, fg=pur_color)
            self.result_widgets["confidence"].config(
                text=f"{confidence}%",
                fg=SUCCESS if confidence > 70 else WARNING)
            self.result_widgets["score"].config(
                text=f"{s_score:.1f} / 100",
                fg=SUCCESS if s_score > 60 else
                   WARNING if s_score > 30 else DANGER)
            self.result_widgets["strategy"].config(text="See below ↓", fg=TEXT)

            self.suggestion_box.config(
                text=strategies[seg_name], bg=seg_color, fg="white")

        except ValueError as ve:
            messagebox.showerror("Input Error",
                                 f"Please enter valid numbers.\n{ve}")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    def _reset(self):
        placeholders = {
            "Age": "e.g. 30",
            "Annual Income": "e.g. 50000",
            "Number of Purchases": "e.g. 10",
            "Time on Website (min)": "e.g. 45",
            "Discounts Availed": "e.g. 3",
        }
        for label, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, placeholders.get(label, ""))
        self.loyalty_var.set(0)
        for w in self.result_widgets.values():
            w.config(text="—", fg=TEXT)
        self.suggestion_box.config(text="", bg=CARD_BG)

    # ─────────────────────────────────────────────
    #  DATASET INFO TAB
    # ─────────────────────────────────────────────
    def _build_info(self):
        parent = self.tab_info

        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical",
                                  command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=BG)

        scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Dataset description
        info_text = (
            "RECOMMENDED KAGGLE DATASET\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Name:   Predict Customer Purchase Behavior Dataset\n"
            "Author: Rabie El Kharoua\n"
            "Link:   kaggle.com/datasets/rabieelkharoua/\n"
            "        predict-customer-purchase-behavior-dataset\n"
            "Rows:   1,500  |  Columns: 9\n\n"
            "FEATURES\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "• Age                – Customer age (18–65)\n"
            "• Gender             – 0 = Female, 1 = Male\n"
            "• AnnualIncome       – Yearly income in currency\n"
            "• NumberOfPurchases  – Past purchase count\n"
            "• ProductCategory    – 0‑4 (Electronics, Clothing,\n"
            "                        Home Goods, Beauty, Sports)\n"
            "• TimeSpentOnWebsite – Minutes on site per session\n"
            "• LoyaltyProgram     – 0 = No, 1 = Yes\n"
            "• DiscountsAvailed   – Number of discounts used\n"
            "• PurchaseStatus     – Target: 0 = No, 1 = Yes\n\n"
            "ENGINEERED FEATURE\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "• SpendingScore – Weighted combo of Income (40%),\n"
            "                   Purchases (35%), Time on Site (25%)\n"
            "                   Normalised to 0–100 scale\n\n"
            "MODELS USED\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"1. KMeans Clustering (k=3) → Segmentation\n"
            f"2. Random Forest (100 trees) → Purchase Prediction\n"
            f"   Test Accuracy: {model_accuracy*100:.1f}%\n\n"
            "HOW TO USE THE REAL KAGGLE DATASET\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "1. Download 'customer_purchase_data.csv' from the\n"
            "   Kaggle link above\n"
            "2. Place it in the same folder as this script\n"
            "3. Run the script — it auto-loads the CSV!\n"
        )

        tk.Label(scroll_frame, text=info_text, font=("Consolas", 11),
                 fg=TEXT, bg=BG, justify="left", padx=30, pady=20,
                 anchor="nw").pack(anchor="nw")

        # Show first 10 rows as a table
        tk.Label(scroll_frame, text="SAMPLE DATA (first 10 rows)",
                 font=FONT_HEAD, fg=HIGHLIGHT, bg=BG,
                 padx=30).pack(anchor="w", pady=(0, 5))

        table_frame = tk.Frame(scroll_frame, bg=BG, padx=30)
        table_frame.pack(anchor="w")

        headers = list(raw[0].keys()) + ["SpendingScore"]
        for j, h in enumerate(headers):
            tk.Label(table_frame, text=h, font=("Consolas", 9, "bold"),
                     fg=HIGHLIGHT, bg=CARD_BG, padx=6, pady=3,
                     relief="flat").grid(row=0, column=j, sticky="nsew")
        for i in range(min(10, len(raw))):
            for j, h in enumerate(headers):
                if h == "SpendingScore":
                    val = f"{spending_score[i]:.1f}"
                else:
                    val = raw[i][h]
                tk.Label(table_frame, text=val, font=("Consolas", 9),
                         fg=TEXT_DIM, bg=BG if i % 2 == 0 else CARD_BG,
                         padx=6, pady=2).grid(row=i+1, column=j,
                         sticky="nsew")


# =====================================================================
#  7. RUN
# =====================================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()