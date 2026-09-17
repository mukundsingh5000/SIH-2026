import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")


def init_db():
    """Initialize SQLite database table for storing prediction logs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            detected INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(class_id: int, class_name: str, confidence: float, detected: bool) -> dict:
    """Save a prediction result record into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    cursor.execute("""
        INSERT INTO predictions (timestamp, class_id, class_name, confidence, detected)
        VALUES (?, ?, ?, ?, ?)
    """, (timestamp, class_id, class_name, confidence, 1 if detected else 0))
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {
        "id": pred_id,
        "timestamp": timestamp,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "detected": detected
    }


def get_all_predictions(limit: int = 50) -> list:
    """Retrieve history of predictions ordered by timestamp descending."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, class_id, class_name, confidence, detected FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for row in rows:
        records.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "confidence": row["confidence"],
            "detected": bool(row["detected"])
        })
    return records


def get_analytics_summary() -> dict:
    """Calculate key analytics from actual SQLite prediction records."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Total Scans
    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    total_scans = cursor.fetchone()["total"]

    # 2. Total Objects Detected (excluding background / class_id 0)
    cursor.execute("SELECT COUNT(*) as detected FROM predictions WHERE detected = 1")
    total_detected = cursor.fetchone()["detected"]

    # 3. Average Confidence
    cursor.execute("SELECT AVG(confidence) as avg_conf FROM predictions")
    avg_row = cursor.fetchone()
    avg_confidence = round(float(avg_row["avg_conf"]), 4) if avg_row and avg_row["avg_conf"] is not None else 0.0

    # 4. Most Detected Object (excluding background)
    cursor.execute("""
        SELECT class_name, COUNT(*) as cnt 
        FROM predictions 
        WHERE detected = 1 
        GROUP BY class_name 
        ORDER BY cnt DESC 
        LIMIT 1
    """)
    most_row = cursor.fetchone()
    most_detected_object = most_row["class_name"] if most_row else "None"

    # 5. Objects by Type (Counts per class, excluding background)
    debris_classes = ["aircraft", "bottle", "cylinder", "human", "net", "pipe", "wreck"]
    object_counts = {cls: 0 for cls in debris_classes}
    
    cursor.execute("""
        SELECT class_name, COUNT(*) as cnt 
        FROM predictions 
        WHERE detected = 1 
        GROUP BY class_name
    """)
    for row in cursor.fetchall():
        if row["class_name"] in object_counts:
            object_counts[row["class_name"]] = row["cnt"]

    # 6. Average Confidence per Class
    all_classes = ["background", "aircraft", "bottle", "cylinder", "human", "net", "pipe", "wreck"]
    confidence_by_class = {cls: 0.0 for cls in all_classes}
    cursor.execute("""
        SELECT class_name, AVG(confidence) as avg_conf 
        FROM predictions 
        GROUP BY class_name
    """)
    for row in cursor.fetchall():
        if row["class_name"] in confidence_by_class and row["avg_conf"] is not None:
            confidence_by_class[row["class_name"]] = round(float(row["avg_conf"]), 4)

    # 7. Recent Detections (Top 10 descending)
    cursor.execute("""
        SELECT id, timestamp, class_id, class_name, confidence, detected 
        FROM predictions 
        ORDER BY id DESC 
        LIMIT 10
    """)
    recent_rows = cursor.fetchall()
    recent_predictions = [
        {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "class_id": r["class_id"],
            "class_name": r["class_name"],
            "confidence": r["confidence"],
            "detected": bool(r["detected"])
        }
        for r in recent_rows
    ]

    conn.close()

    return {
        "total_scans": total_scans,
        "total_detected": total_detected,
        "average_confidence": avg_confidence,
        "most_detected_object": most_detected_object,
        "object_counts": object_counts,
        "confidence_by_class": confidence_by_class,
        "recent_predictions": recent_predictions
    }


