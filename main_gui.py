import sys
import sqlite3
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTableWidget, QTableWidgetItem, QFileDialog, QHeaderView, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt

# --- DB HANDLER ---
DB_PATH = "E:/poppy/gee_agri_data.db"
TABLE_NAME = "crop_data"
def get_connection():
    return sqlite3.connect(DB_PATH)
def load_data(village_filter=None, year_filter=None):
    conn = get_connection()
    query = f"SELECT * FROM {TABLE_NAME}"
    filters = []
    if village_filter:
        filters.append(f"Village LIKE '%{village_filter}%' ")
    if year_filter:
        filters.append(f"Year={year_filter}")
    if filters:
        query += " WHERE " + " AND ".join(filters)
    df = pd.read_sql(query, conn)
    conn.close()
    return df
def export_to_csv(df, path):
    df.to_csv(path, index=False)

# --- GEE MODULE (stub) ---
def run_gee_extraction(village, lat, lon, date_start, date_end):
    # Replace with your actual GEE extraction logic
    features = {
        "NDVI": 0.65,
        "NIR": 0.23,
        "RED": 0.12,
        "VV": -10.5,
        "VH": -15.2
    }
    preview_img_path = None
    return features, preview_img_path

# --- MODEL PREDICT (stub) ---
def predict_yield(features_dict):
    # Replace with your actual model/scaler/imputer loading and prediction
    # Here, just a dummy formula for demo
    pred = 1000 * (0.5 * features_dict.get("NDVI", 0) + 0.1 * features_dict.get("NIR", 0))
    return float(pred)

# --- MAIN GUI ---
class CropYieldApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Crop Yield Prediction System")
        self.resize(1200, 800)
        self.selected_row_features = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        top_split = QHBoxLayout()

        # LEFT PANEL
        left_panel = QGroupBox("Database Viewer")
        left_layout = QVBoxLayout()
        self.village_search = QLineEdit()
        self.village_search.setPlaceholderText("Search Village")
        self.year_search = QLineEdit()
        self.year_search.setPlaceholderText("Year")
        search_btn = QPushButton("Load Data")
        search_btn.clicked.connect(self.load_db_data)
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(self.export_db_data)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.on_row_select)
        left_layout.addWidget(self.village_search)
        left_layout.addWidget(self.year_search)
        left_layout.addWidget(search_btn)
        left_layout.addWidget(export_btn)
        left_layout.addWidget(self.table)
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)

        # RIGHT PANEL
        right_panel = QGroupBox("GEE Extraction Interface")
        right_layout = QFormLayout()
        self.gee_village = QLineEdit()
        self.gee_lat = QLineEdit()
        self.gee_lon = QLineEdit()
        self.gee_date_start = QLineEdit()
        self.gee_date_end = QLineEdit()
        gee_btn = QPushButton("Run GEE Extraction")
        gee_btn.clicked.connect(self.run_gee)
        self.gee_result = QLabel("No extraction yet.")
        right_layout.addRow("Village/Region:", self.gee_village)
        right_layout.addRow("Latitude:", self.gee_lat)
        right_layout.addRow("Longitude:", self.gee_lon)
        right_layout.addRow("Date Start:", self.gee_date_start)
        right_layout.addRow("Date End:", self.gee_date_end)
        right_layout.addRow(gee_btn)
        right_layout.addRow(self.gee_result)
        right_panel.setLayout(right_layout)

        # LOWER PANEL
        lower_panel = QGroupBox("Prediction System")
        lower_layout = QHBoxLayout()
        self.input_ndvi = QLineEdit()
        self.input_nir = QLineEdit()
        self.input_red = QLineEdit()
        self.input_vv = QLineEdit()
        self.input_vh = QLineEdit()
        autofill_btn = QPushButton("Auto-fill from selected row")
        autofill_btn.clicked.connect(self.autofill_from_row)
        predict_btn = QPushButton("Predict Yield")
        predict_btn.clicked.connect(self.predict_yield_action)
        self.pred_result = QLabel("Prediction: -")
        for label, widget in [
            ("NDVI", self.input_ndvi), ("NIR", self.input_nir),
            ("RED", self.input_red), ("VV", self.input_vv), ("VH", self.input_vh)
        ]:
            lower_layout.addWidget(QLabel(label))
            lower_layout.addWidget(widget)
        lower_layout.addWidget(autofill_btn)
        lower_layout.addWidget(predict_btn)
        lower_layout.addWidget(self.pred_result)
        lower_panel.setLayout(lower_layout)

        # Assemble panels
        top_split.addWidget(left_panel, 3)
        top_split.addWidget(right_panel, 7)
        main_layout.addLayout(top_split)
        main_layout.addWidget(lower_panel)

    def load_db_data(self):
        village = self.village_search.text()
        year = self.year_search.text()
        df = load_data(village, year)
        self.current_df = df
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)
        for i, row in df.iterrows():
            for j, val in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(val)))
        self.table.resizeColumnsToContents()

    def export_db_data(self):
        if hasattr(self, 'current_df'):
            path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if path:
                export_to_csv(self.current_df, path)

    def on_row_select(self):
        selected = self.table.selectedItems()
        if selected:
            row = selected[0].row()
            # Try to get columns by name for robustness
            col_map = {self.table.horizontalHeaderItem(i).text(): i for i in range(self.table.columnCount())}
            try:
                self.selected_row_features = {
                    "NDVI": float(self.table.item(row, col_map["NDVI"]).text()),
                    "NIR": float(self.table.item(row, col_map["NIR"]).text()),
                    "RED": float(self.table.item(row, col_map["RED"]).text()),
                    "VV": float(self.table.item(row, col_map["VV"]).text()),
                    "VH": float(self.table.item(row, col_map["VH"]).text())
                }
            except Exception:
                self.selected_row_features = None

    def autofill_from_row(self):
        if self.selected_row_features:
            self.input_ndvi.setText(str(self.selected_row_features["NDVI"]))
            self.input_nir.setText(str(self.selected_row_features["NIR"]))
            self.input_red.setText(str(self.selected_row_features["RED"]))
            self.input_vv.setText(str(self.selected_row_features["VV"]))
            self.input_vh.setText(str(self.selected_row_features["VH"]))

    def run_gee(self):
        village = self.gee_village.text()
        lat = self.gee_lat.text()
        lon = self.gee_lon.text()
        date_start = self.gee_date_start.text()
        date_end = self.gee_date_end.text()
        features, preview_img = run_gee_extraction(village, lat, lon, date_start, date_end)
        self.gee_result.setText(str(features))

    def predict_yield_action(self):
        try:
            features = {
                "NDVI": float(self.input_ndvi.text()),
                "NIR": float(self.input_nir.text()),
                "RED": float(self.input_red.text()),
                "VV": float(self.input_vv.text()),
                "VH": float(self.input_vh.text())
            }
            pred = predict_yield(features)
            self.pred_result.setText(f"Prediction: {pred:.2f} Kg/Ha")
        except Exception as e:
            self.pred_result.setText(f"Error: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CropYieldApp()
    window.show()
    sys.exit(app.exec_())
