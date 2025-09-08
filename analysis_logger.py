import csv
from datetime import datetime
import os

class AnalysisLogger:
    def __init__(self):
        os.makedirs('logs', exist_ok=True)
        
    def log_decision(self, symbol, prediction, confidence, reason):
        """Log every decision the bot makes"""
        with open('logs/decision_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                symbol,
                prediction,
                f"{confidence:.2f}",
                reason
            ])
            
    def log_missed_opportunity(self, symbol, confidence, reason):
        """Log specifically why opportunities were missed"""
        with open('logs/missed_opportunities.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                symbol,
                f"{confidence:.2f}",
                reason
            ])

# Initialize in your app.py
analysis_logger = AnalysisLogger()
